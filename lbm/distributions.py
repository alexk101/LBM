# distributions.py
"""LBM distribution functions for coupled multi-distribution simulations.

Supports single-distribution (isothermal), double-distribution (thermal:
momentum f + energy g), and multiphase models. All distributions consume
and produce macroscopic fields via a shared state dict so that e.g. the
thermal equilibrium g_eq can use rho, u from the momentum distribution and
T from the previous step (see Wikipedia: Lattice Boltzmann methods § Mathematical
equations for simulations).
"""
from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from .definitions import MacroState
from .equilibrium import EquilibriumModel, PolynomialEquilibrium
from .lattice import Lattice

_POLYNOMIAL_EQ = PolynomialEquilibrium()


class Distribution(eqx.Module):
    """Base class for LBM distribution functions.

    Each distribution represents a population (e.g. momentum f or energy g)
    moving in discrete velocity directions. For coupled systems (e.g. thermal
    LBM), equilibrium and lift use a shared macroscopic state dict so that
    one distribution can consume fields produced by another.

    Design (modular coupling):
        - equilibrium(macro, lattice): reads from macro (e.g. "rho", "u", "T")
        - lift(f, lattice, macro=None): writes fields into the state; may read
          macro for normalization (e.g. T = Σg_i/rho)
        - relaxation_time: BGK relaxation parameter (τ_f or τ_g)

    Parameters:
        name: Identifier string for this distribution type.
        label: Key for this distribution's array in state (e.g. "F", "G").
    """

    name: str = eqx.field(static=True)
    label: str = eqx.field(static=True)

    @abstractmethod
    def relaxation_time(self) -> float:
        """Relaxation time τ for BGK collision (ω = 1/τ)."""
        raise NotImplementedError()

    @abstractmethod
    def equilibrium(self, macro: MacroState, lattice: Lattice) -> Array:
        """Compute equilibrium distribution from current macroscopic state.

        Args:
            macro: Shared state with keys e.g. "rho", "u", "T". Each distribution
                   reads only what it needs (e.g. thermal reads rho, u, T).
            lattice: Lattice configuration.

        Returns:
            Equilibrium distribution with shape (..., N).
        """
        raise NotImplementedError()

    @abstractmethod
    def lift(
        self, f: Array, lattice: Lattice, macro: MacroState | None = None
    ) -> dict[str, Array]:
        """Extract macroscopic fields from this distribution (moments).

        Args:
            f: Post-streaming distribution (..., N).
            lattice: Lattice configuration.
            macro: Current macro state; used by some distributions (e.g. thermal
                   uses macro["rho"] for T = Σg_i/rho).

        Returns:
            Dict of field name -> array to merge into macro state (e.g. {"rho", "u"}
            or {"T"}).
        """
        raise NotImplementedError()


class ParticleDistribution(Distribution):
    """Standard single-distribution LBM for incompressible flow.

    The most common LBM model using the BGK collision operator with a
    single distribution function F. Suitable for low-Mach-number,
    isothermal incompressible flows.

    Physics:
        - BGK collision: f_i(x + e_i*dt, t+dt) = f_i(x,t) - ω(f_i - f_i^eq)
        - Equilibrium: Maxwell-Boltzmann expansion to second order (or swap via equilibrium_model)
        - Viscosity: ν = c_s²(τ - 0.5), where τ is relaxation time

    Parameters:
        name: Identifier for this distribution (default: "Particle")
        label: Symbolic label used in output fields (default: "F")
        tau: Relaxation time parameter controlling viscosity (default: 0.5)
        equilibrium_model: How to compute f_eq; None = built-in polynomial (default).

    Stability Notes:
        - τ must be > 0.5 for numerical stability (positive viscosity)
        - Typical range: 0.6 ≤ τ ≤ 1.0 balances stability and accuracy
        - For high Reynolds numbers, may need adaptive τ or MRT collision
    """

    name: str = "Particle"
    label: str = "F"
    tau: float = 0.5
    equilibrium_model: EquilibriumModel | None = None

    def relaxation_time(self) -> float:
        return self.tau

    def equilibrium(self, macro: MacroState, lattice: Lattice) -> Array:
        model = self.equilibrium_model if self.equilibrium_model is not None else _POLYNOMIAL_EQ
        return model.compute(macro, lattice)

    def lift(
        self, f: Array, lattice: Lattice, macro: MacroState | None = None
    ) -> dict[str, Array]:
        """Extracts macroscopic fields from distribution function via moment calculation."""
        rho = jnp.sum(f, axis=-1, keepdims=True)
        momentum = jnp.einsum("...i,ic->...c", f, lattice.e)
        u = momentum / (rho + 1e-10)
        return {"rho": rho, "u": u}


class ThermalDistribution(Distribution):
    """Energy distribution G for compressible thermal LBM (double-distribution F+G).

    The G distribution carries total energy density. Its zeroth moment is
    ``sum(g_i) = 2 * rho * E`` where ``E = T*Cv + |u|^2/2`` is the total
    energy per unit mass. Temperature is recovered via:
        ``T = (sum(g)/(2*rho) - |u|^2/2) / Cv``

    The equilibrium should be Levermore (Newton-solved) for compressible flows.
    Pass ``equilibrium_model=LevermoreEquilibrium(Cv=...)`` to use it; if None,
    falls back to a simple polynomial expansion (NOT recommended for production).

    Parameters:
        tau_T: Relaxation time for the energy field.
        gamma: Heat capacity ratio (default 1.4 for diatomic gas).
        Cv: Specific heat at constant volume in lattice units.
            For ideal gas: Cv = 1/(gamma - 1).  Must match the Cv used
            by the equilibrium model.
        equilibrium_model: Pluggable equilibrium (e.g. LevermoreEquilibrium).
    """

    name: str = "Thermal"
    label: str = "G"
    tau_T: float = 0.5
    gamma: float = 1.4
    Cv: float = 2.5
    equilibrium_model: EquilibriumModel | None = None

    def relaxation_time(self) -> float:
        return self.tau_T

    def equilibrium(self, macro: MacroState, lattice: Lattice) -> Array:
        if self.equilibrium_model is not None:
            return self.equilibrium_model.compute(macro, lattice)
        return _POLYNOMIAL_EQ.compute(macro, lattice)

    def lift(
        self, f: Array, lattice: Lattice, macro: MacroState | None = None
    ) -> dict[str, Array]:
        """Extract temperature from the energy distribution.

        T = (E - |u|^2/2) / Cv,  where E = sum(g_i) / (2 * rho).
        """
        sum_g = jnp.sum(f, axis=-1, keepdims=True)
        if macro is not None and "rho" in macro:
            rho = macro["rho"]
            if rho.ndim < 1 or rho.shape[-1] != 1:
                rho = rho[..., None]
        else:
            rho = jnp.ones_like(sum_g)

        u = macro["u"] if macro is not None and "u" in macro else None

        E = sum_g / (2.0 * rho + 1e-10)
        if u is not None:
            u_sq = jnp.sum(u ** 2, axis=-1, keepdims=True)
            T = (E - 0.5 * u_sq) / self.Cv
        else:
            T = E / self.Cv
        T = jnp.maximum(T, 1e-10)
        return {"T": T}


class MultiphaseDistribution(Distribution):
    """Density-dependent distribution for multiphase flow using Shan-Chen model.

    This implements the pseudo-potential (Shan-Chen) method for simulating
    immiscible fluid interfaces and surface tension effects without explicit
    interface tracking.

    Physics:
        - Fluid particles interact via short-range repulsive/attractive force
        - The interaction strength controls interfacial surface tension
        - Density variations naturally lead to phase separation

    Parameters:
        name: Identifier for this distribution (default: "Multiphase")
        label: Symbolic label used in output fields (default: "F")
        tau: Relaxation time parameter (default: 0.5)
        interaction_strength: Force coupling parameter σ (default: -1.0).
                              Negative values lead to phase separation.

    Phase Separation Notes:
        - |σ| < 0.25 for stable simulations without spurious currents
        - More negative = stronger surface tension and sharper interfaces
        - Density ratio between phases limited (~10:1 typical)
    """

    name: str = "Multiphase"
    label: str = "F"
    tau: float = 0.5
    interaction_strength: float = -1.0

    def relaxation_time(self) -> float:
        return self.tau

    def equilibrium(self, macro: MacroState, lattice: Lattice) -> Array:
        rho = macro["rho"]
        u = macro["u"]
        if rho.ndim >= 1 and rho.shape[-1] == 1:
            rho = jnp.squeeze(rho, axis=-1)
        psi = rho + self.interaction_strength * rho**2
        e_dot_u = jnp.einsum("...c,dc->...d", u, lattice.e)
        u_sq = jnp.sum(u**2, axis=-1, keepdims=True)
        f_eq = (
            psi[..., None]
            * lattice.expanded_weights[None, :]
            * (
                1.0
                + e_dot_u / lattice.c_s_sq
                + e_dot_u**2 / (2.0 * lattice.c_s_sq**2)
                - u_sq / (2.0 * lattice.c_s_sq)
            )
        )
        return f_eq

    def lift(
        self, f: Array, lattice: Lattice, macro: MacroState | None = None
    ) -> dict[str, Array]:
        """Density and velocity from multiphase distribution (same moments as standard)."""
        rho = jnp.sum(f, axis=-1, keepdims=True)
        momentum = jnp.einsum("...i,ic->...c", f, lattice.e)
        u = momentum / (rho + 1e-10)

        return {"rho": rho, "u": u}
