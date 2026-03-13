"""Equilibrium distribution models.

Available models:
    PolynomialEquilibrium   — second-order Maxwell-Boltzmann expansion (standard BGK)
    EntropicEquilibrium     — Ansumali-Karlin factorized product formula (closed-form)
    ExtendedEquilibrium     — factorized phi product for compressible flows
    LevermoreEquilibrium    — optimistix Newton root-find for energy (G) distribution
"""
from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from jax import Array

from .definitions import MacroState
from .lattice import Lattice


class EquilibriumModel(eqx.Module):
    """Abstract: computes equilibrium distribution f_eq from macro state and lattice."""

    @abstractmethod
    def compute(self, macro: MacroState, lattice: Lattice) -> Array:
        raise NotImplementedError()


# ---------------------------------------------------------------------------
# Polynomial (standard second-order Maxwell-Boltzmann expansion)
# ---------------------------------------------------------------------------


class PolynomialEquilibrium(EquilibriumModel):
    """Second-order Maxwell-Boltzmann expansion: f_i^eq = rho w_i (1 + u·e_i/c_s² + ...).

    Same equilibrium used by standard BGK, MRT, TRT, and regularized schemes.
    Expects macro['rho'] (with trailing 1-dim), macro['u'].
    """

    def compute(self, macro: MacroState, lattice: Lattice) -> Array:
        rho = macro["rho"]
        u = macro["u"]
        if rho.ndim >= 1 and rho.shape[-1] == 1:
            rho = jnp.squeeze(rho, axis=-1)
        e_dot_u = jnp.einsum("...c,dc->...d", u, lattice.e)
        u_sq = jnp.sum(u**2, axis=-1, keepdims=True)
        cs2 = lattice.c_s_sq
        return (
            rho[..., None]
            * lattice.expanded_weights
            * (1.0 + e_dot_u / cs2 + e_dot_u**2 / (2.0 * cs2**2) - u_sq / (2.0 * cs2))
        )


# ---------------------------------------------------------------------------
# Entropic (Ansumali-Karlin factorized product formula)
# ---------------------------------------------------------------------------


class EntropicEquilibrium(EquilibriumModel):
    """Entropic equilibrium via the Ansumali-Karlin factorized product formula.

    For each spatial dimension d independently:
        factor_d   = 2 - sqrt(1 + 3 u_d²)
        ratio_d    = (2 u_d + sqrt(1 + 3 u_d²)) / (1 - u_d)
        contrib_id = factor_d * ratio_d ^ e_{i,d}

    Then: f_i^eq = rho * w_i(T) * Π_d contrib_{i,d}

    This is a closed-form expression (no implicit solve). The weights w_i(T) are
    temperature-dependent Levermore weights.

    Expects macro['rho'], macro['u'], macro['T'].

    References:
        Ansumali & Karlin (2002), Entropic lattice Boltzmann method for large
        scale turbulence simulation. IJMPC 14, 1111-1123.
        Karlin, Ferrante & Öttinger (1999), Perfect entropy functions of the
        Lattice Boltzmann method. Europhys. Lett. 47, 182-188.
    """

    def compute(self, macro: MacroState, lattice: Lattice) -> Array:
        rho = macro["rho"]
        u = macro["u"]
        T = macro.get("T")
        if rho.ndim >= 1 and rho.shape[-1] == 1:
            rho = jnp.squeeze(rho, axis=-1)
        if T is None:
            T = jnp.full_like(rho, lattice.c_s_sq)
        elif T.ndim >= 1 and T.shape[-1] == 1:
            T = jnp.squeeze(T, axis=-1)

        eps = jnp.finfo(u.dtype).eps
        w = lattice.levermore_weights(T)

        u_expanded = u[..., None, :]
        e = lattice.e

        u_sq = u_expanded * u_expanded
        sqrt_term = jnp.sqrt(1.0 + 3.0 * u_sq)

        numerator = 2.0 * u_expanded + sqrt_term
        denominator = 1.0 - u_expanded
        denominator = jnp.where(jnp.abs(denominator) < eps, eps, denominator)

        ratio = numerator / denominator
        factor = 2.0 - sqrt_term

        power_term = jnp.pow(ratio, e)
        f_dims = factor * power_term

        f_product = jnp.prod(f_dims, axis=-1)
        return rho[..., None] * w * f_product


# ---------------------------------------------------------------------------
# Extended (factorized phi product formula for compressible flows)
# ---------------------------------------------------------------------------


class ExtendedEquilibrium(EquilibriumModel):
    """Extended equilibrium using the factorized phi product formula.

    For each spatial dimension d, three basis functions are defined
    (indexed by the integer velocity component e_{i,d} ∈ {-1, 0, 1}):
        phi[0](u_d, T) = 1 - (u_d² + T)           (rest: e=0)
        phi[1](u_d, T) = (u_d + u_d² + T) / 2     (positive: e=+1)
        phi[2](u_d, T) = (-u_d + u_d² + T) / 2    (negative: e=-1)

    Then: f_i^eq = rho * Π_d phi[e_{i,d}](u_d, T)

    Expects macro['rho'], macro['u'], macro['T'].

    References:
        Saadat, Dorschner & Karlin (2021), Extended lattice Boltzmann model.
        Entropy 23, 475.
        Frapolli, Chikatamarla & Karlin (2015), Entropic lattice Boltzmann
        model for compressible flows. Phys. Rev. E 92, 061301(R).
    """

    def compute(self, macro: MacroState, lattice: Lattice) -> Array:
        rho = macro["rho"]
        u = macro["u"]
        T = macro["T"]
        if rho.ndim >= 1 and rho.shape[-1] == 1:
            rho = jnp.squeeze(rho, axis=-1)
        if T.ndim >= 1 and T.shape[-1] == 1:
            T = jnp.squeeze(T, axis=-1)

        D = lattice.D
        vel_int = jnp.int32(jnp.round(lattice.velocities))
        vel_idx = vel_int % 3

        u_eff = u - lattice.shifts if lattice.is_shifted else u

        u_sq = u_eff**2
        phi_0 = 1.0 - (u_sq + T[..., None])
        phi_1 = (u_eff + u_sq + T[..., None]) * 0.5
        phi_2 = (-u_eff + u_sq + T[..., None]) * 0.5
        phi = jnp.stack([phi_0, phi_1, phi_2], axis=-2)

        d_range = jnp.arange(D)
        phi_selected = phi[..., vel_idx, d_range]

        return rho[..., None] * jnp.prod(phi_selected, axis=-1)


# ---------------------------------------------------------------------------
# Levermore (optimistix Newton root-find for energy distribution)
# ---------------------------------------------------------------------------


class LevermoreEquilibrium(EquilibriumModel):
    """Levermore equilibrium via Newton root-find (optimistix) for the energy distribution.

    Solves for Lagrange multipliers (χ, ζ) such that:
        g_i = w_i(T) * exp(χ + ζ · e_i)
    subject to moment constraints:
        Σ g_i = 2E,    Σ g_i * e_i = 2 u H
    where E = T Cv + |u|²/2 is total energy and H = E + T is enthalpy.

    Uses ``optimistix.root_find`` with Newton's method, vmapped over
    spatial points for an efficient per-point (D+1)×(D+1) solve.

    The Newton system has D+1 unknowns per spatial point (χ scalar, ζ D-vector)
    and is solved with jnp.linalg.solve for the (D+1)×(D+1) Jacobian. Uses 
    ``optimistix.root_find`` with Newton's method, vmapped over spatial points
    for an efficient per-point (D+1)×(D+1) solve.

    Expects macro['rho'], macro['u'], macro['T']. Cv must be supplied at init.

    References:
        Levermore (1996), Moment closure hierarchies for kinetic theories.
        J. Stat. Phys. 83, 1021-1065.
    """

    rtol: float = 1e-8
    atol: float = 1e-8
    max_steps: int = 64
    Cv: float = 1.0
    throw: bool = False

    def compute(self, macro: MacroState, lattice: Lattice) -> Array:
        rho = macro["rho"]
        u = macro["u"]
        T = macro["T"]
        if rho.ndim >= 1 and rho.shape[-1] == 1:
            rho = jnp.squeeze(rho, axis=-1)
        if T.ndim >= 1 and T.shape[-1] == 1:
            T = jnp.squeeze(T, axis=-1)

        T_safe = jnp.maximum(T, 1e-6)
        rho_safe = jnp.maximum(rho, 1e-6)

        D = lattice.D
        e = lattice.e                                # (N, D) — shifted
        w = lattice.levermore_weights(T_safe)         # (..., N)

        uu = jnp.sum(u**2, axis=-1)
        E = T_safe * self.Cv + 0.5 * uu
        H = E + T_safe

        spatial_shape = rho.shape
        flat_len = int(np.prod(spatial_shape)) if spatial_shape else 1

        # Flatten spatial dims → (B,) for vmap
        w_flat = w.reshape(flat_len, -1)             # (B, N)
        E_flat = E.reshape(flat_len)                 # (B,)
        u_flat = u.reshape(flat_len, D)              # (B, D)
        H_flat = H.reshape(flat_len)                 # (B,)
        rho_flat = rho_safe.reshape(flat_len)        # (B,)

        solver = optx.Newton(rtol=self.rtol, atol=self.atol)

        def _solve_point(w_pt: Array, E_pt: Array, u_pt: Array, H_pt: Array):
            """Root-find for a single spatial point."""

            def residual(y: tuple[Array, Array], _args: None):
                khi, zeta = y
                zeta_dot_e = zeta @ e.T                           # (N,)
                f = w_pt * jnp.exp(khi + zeta_dot_e)              # (N,)
                r_0 = jnp.sum(f) - 2.0 * E_pt                    # scalar
                r_d = f @ e - 2.0 * u_pt * H_pt                  # (D,)
                return (r_0, r_d)

            y0 = (jnp.float32(0.0), jnp.zeros(D, dtype=jnp.float32))
            sol = optx.root_find(
                residual, solver, y0, args=None,
                max_steps=self.max_steps, throw=self.throw,
            )
            return sol.value

        khi_flat, zeta_flat = jax.vmap(_solve_point)(w_flat, E_flat, u_flat, H_flat)

        khi = khi_flat.reshape(spatial_shape)
        zeta = zeta_flat.reshape(*spatial_shape, D)

        zeta_dot_e = jnp.einsum("...d,nd->...n", zeta, e)
        return rho_safe[..., None] * w * jnp.exp(khi[..., None] + zeta_dot_e)


# np needed at module level for the prod() call in LevermoreEquilibrium
import numpy as np  # noqa: E402
