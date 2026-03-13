"""Compressible thermal LBM solver (double-distribution F + G).

Uses the extended equilibrium for F (mass/momentum) and the Levermore
equilibrium for G (energy).  The relaxation time for F is spatially varying:

    tau_f = viscosity / (rho * T) + 0.5

and for G it is derived from the Prandtl relation:

    tau_g = 0.5 + (tau_f - 0.5) / Pr

A heat-flux coupling term (Gis) is added to the G collision to enforce the
correct energy transport:

    G_collided += Gis * (1/tau_f - 1/tau_g)

where Gis = w_i(T) * (q · e_i) / T, and q is the non-equilibrium heat flux
computed from the momentum pressure tensor deficit.

Physical parameters:
    Pr    — Prandtl number.
    gamma — heat capacity ratio (adiabatic index).
    Cv    — specific heat at constant volume (lattice units).
    viscosity — dynamic viscosity (lattice units); used with rho,T for tau_f.
"""
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ..boundaries import apply_boundaries
from ..collision import BGKCollision
from ..definitions import LBMState, MacroState
from ..distributions import ParticleDistribution, ThermalDistribution, Distribution
from ..equilibrium import ExtendedEquilibrium, LevermoreEquilibrium
from ..lattice import Lattice
from .base import BaseSolver


class CompressibleSolver(BaseSolver):
    """Compressible thermal LBM with F (mass/momentum) + G (energy).

    Uses Extended equilibrium for F and Levermore equilibrium for G
    (following Frapolli, Chikatamarla & Karlin, 2015).

    The specific heat Cv is derived from gamma: Cv = 1/(gamma - 1).
    For diatomic gas (gamma=1.4): Cv = 2.5.

    Args:
        Pr: Prandtl number (default 0.71 for air).
        gamma: Heat capacity ratio (default 1.4 for diatomic gas).
        viscosity: Dynamic viscosity in lattice units.
    """

    Pr: float = 0.71
    gamma: float = 1.4
    viscosity: float = 0.01

    _Cv: float = eqx.field(init=False)
    _dist_f: ParticleDistribution = eqx.field(init=False)
    _dist_g: ThermalDistribution = eqx.field(init=False)

    def __post_init__(self) -> None:
        Cv = 1.0 / (self.gamma - 1.0)
        object.__setattr__(self, "_Cv", Cv)

        tau_f_ref = self.viscosity / (1.0 / 3.0) + 0.5
        tau_g_ref = 0.5 + (tau_f_ref - 0.5) / self.Pr

        object.__setattr__(
            self, "_dist_f",
            ParticleDistribution(
                tau=tau_f_ref,
                equilibrium_model=ExtendedEquilibrium(),
            ),
        )
        object.__setattr__(
            self, "_dist_g",
            ThermalDistribution(
                tau_T=tau_g_ref,
                gamma=self.gamma,
                Cv=Cv,
                equilibrium_model=LevermoreEquilibrium(Cv=Cv),
            ),
        )
        super().__post_init__()

    @property
    def distributions(self) -> tuple[Distribution, ...]:
        return (self._dist_f, self._dist_g)

    def system_variables(self) -> dict[str, float]:
        tau_f_ref = self.viscosity / (1.0 / 3.0) + 0.5
        tau_g_ref = 0.5 + (tau_f_ref - 0.5) / self.Pr
        return {
            "Pr": self.Pr,
            "gamma": self.gamma,
            "Cv": self._Cv,
            "viscosity": self.viscosity,
            "tau_f_ref": tau_f_ref,
            "tau_g_ref": tau_g_ref,
        }

    # ------------------------------------------------------------------
    # Spatially varying tau
    # ------------------------------------------------------------------

    def _tau_f(self, rho: Array, T: Array) -> Array:
        """tau_f(x) = viscosity / (rho * T) + 0.5."""
        if rho.ndim >= 1 and rho.shape[-1] == 1:
            rho = jnp.squeeze(rho, axis=-1)
        if T.ndim >= 1 and T.shape[-1] == 1:
            T = jnp.squeeze(T, axis=-1)
        denom = jnp.maximum(rho * T, 1e-10)
        return self.viscosity / denom + 0.5

    def _tau_g(self, tau_f: Array) -> Array:
        """tau_g from Prandtl relation: 0.5 + (tau_f - 0.5) / Pr."""
        return 0.5 + (tau_f - 0.5) / self.Pr

    # ------------------------------------------------------------------
    # Heat-flux coupling (Gis)
    # ------------------------------------------------------------------

    def _compute_heat_flux(
        self, f: Array, rho: Array, u: Array, T: Array,
    ) -> Array:
        """Non-equilibrium heat flux q from pressure tensor deficit.

        q_d = 2 * sum_ij delta_P_{d,j} * u_j
        where delta_P = P - P_eq, P_{ij} = sum_k f_k * e_{ki} * e_{kj},
        and P_eq_{ij} = rho * (T * delta_{ij} + u_i * u_j).
        """
        e = self.lattice.e  # (N, D) — shifted
        D = self.lattice.D

        if rho.ndim >= 1 and rho.shape[-1] == 1:
            rho = jnp.squeeze(rho, axis=-1)
        if T.ndim >= 1 and T.shape[-1] == 1:
            T = jnp.squeeze(T, axis=-1)

        # Pressure tensor P_{ij} = sum_k f_k * e_{ki} * e_{kj}  -> (..., D, D)
        P = jnp.einsum("...k,ki,kj->...ij", f, e, e)

        # Maxwellian equilibrium pressure tensor P_eq = rho * (T*I + u⊗u)
        eye = jnp.eye(D)
        P_eq = rho[..., None, None] * (
            T[..., None, None] * eye + jnp.einsum("...i,...j->...ij", u, u)
        )

        delta_P = P - P_eq  # (..., D, D)
        q = 2.0 * jnp.einsum("...ij,...j->...i", delta_P, u)  # (..., D)
        return q

    def _gis_term(
        self, q: Array, T: Array,
    ) -> Array:
        """Gis_i = w_i(T) * (q · e_i) / T."""
        e = self.lattice.e  # (N, D) — shifted
        if T.ndim >= 1 and T.shape[-1] == 1:
            T = jnp.squeeze(T, axis=-1)

        w = self.lattice.levermore_weights(T)  # (..., N)
        q_dot_e = jnp.einsum("...d,nd->...n", q, e)  # (..., N)
        T_safe = jnp.maximum(T, 1e-10)
        return w * q_dot_e / T_safe[..., None]

    # ------------------------------------------------------------------
    # Custom collide-and-stream
    # ------------------------------------------------------------------

    def _collide_and_stream(
        self,
        state: LBMState,
        macro: MacroState,
        boundary_velocity: dict[tuple[int, int], Array] | None = None,
        boundary_pressure: dict[tuple[int, int], Array] | None = None,
    ) -> dict[str, Array]:
        rho = macro["rho"]
        u = macro["u"]
        T = macro["T"]

        tau_f = self._tau_f(rho, T)
        tau_g = self._tau_g(tau_f)
        omega_f = 1.0 / tau_f
        omega_g = 1.0 / tau_g

        # --- F collision (spatially varying omega) ---
        f = state.dists[self._dist_f.label]
        f_eq = self._dist_f.equilibrium(macro, self.lattice)
        f_star = f + omega_f[..., None] * (f_eq - f)

        # --- G collision + Gis coupling ---
        g = state.dists[self._dist_g.label]
        g_eq = self._dist_g.equilibrium(macro, self.lattice)
        g_star = g + omega_g[..., None] * (g_eq - g)

        q = self._compute_heat_flux(f, rho, u, T)
        Gis = self._gis_term(q, T)
        g_star = g_star + Gis * (omega_f[..., None] - omega_g[..., None])

        # --- Stream + BCs ---
        from ..obstacles import apply_obstacles

        new_dists: dict[str, Array] = {}
        for label, f_coll in [
            (self._dist_f.label, f_star),
            (self._dist_g.label, g_star),
        ]:
            f_shifted = self._interpolate_shift(f_coll)
            f_streamed = self._stream(f_shifted)
            if self.boundary_spec is not None:
                f_streamed = apply_boundaries(
                    f_streamed,
                    self.lattice,
                    self.boundary_spec,
                    macro=macro,
                    boundary_velocity=boundary_velocity,
                    boundary_pressure=boundary_pressure,
                )
            if self.obstacles:
                f_streamed = apply_obstacles(f_streamed, self.lattice, self.obstacles)
            new_dists[label] = f_streamed
        return new_dists
