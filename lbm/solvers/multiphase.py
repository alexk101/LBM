"""Multiphase LBM solver using the Shan-Chen pseudo-potential model.

Single distribution F with a density-dependent interaction force that drives
phase separation.  The Shan-Chen force is computed from the pseudo-potential
ψ(rho) and incorporated as a velocity shift in the equilibrium:

    u_eq = u + τ * F_SC / rho

where F_SC is the inter-particle force:

    F_SC(x) = -G * ψ(x) * Σ_i w_i * ψ(x + e_i) * e_i

Parameters:
    interaction_strength (G) — coupling constant. Negative values cause
        attractive interactions and phase separation.
    tau — relaxation time.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..boundaries import apply_boundaries
from ..definitions import LBMState, MacroState
from ..distributions import ParticleDistribution, Distribution
from ..lattice import Lattice
from .base import BaseSolver


class MultiphaseSolver(BaseSolver):
    """Shan-Chen multiphase LBM solver.

    Args:
        interaction_strength: Shan-Chen coupling G (negative = phase separation).
        tau: Relaxation time.
    """

    interaction_strength: float = -1.0
    tau: float = 0.8

    _dist_f: ParticleDistribution = eqx.field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "_dist_f", ParticleDistribution(tau=self.tau),
        )
        super().__post_init__()

    @property
    def distributions(self) -> tuple[Distribution, ...]:
        return (self._dist_f,)

    def system_variables(self) -> dict[str, float]:
        cs2 = self.lattice.c_s_sq
        nu = cs2 * (self.tau - 0.5)
        return {
            "interaction_strength": self.interaction_strength,
            "tau": self.tau,
            "nu": nu,
        }

    # ------------------------------------------------------------------
    # Shan-Chen interaction force
    # ------------------------------------------------------------------

    def _pseudo_potential(self, rho: Array) -> Array:
        """ψ(rho) = rho₀ * (1 - exp(-rho/rho₀)) for numerical stability, or simply rho."""
        return rho

    def _shan_chen_force(self, rho: Array) -> Array:
        """F_SC(x) = -G * ψ(x) * Σ_i w_i * ψ(x+e_i) * e_i.

        Streams ψ along each lattice direction to evaluate neighbors, then
        contracts with weights and velocity vectors. Returns shape (..., D).
        """
        if rho.ndim >= 1 and rho.shape[-1] == 1:
            rho = jnp.squeeze(rho, axis=-1)

        psi = self._pseudo_potential(rho)           # (...)
        e_int = self.lattice.velocities              # (N, D) — integer for grid rolls
        e_phys = self.lattice.e                      # (N, D) — shifted for physics
        w = self.lattice.expanded_weights            # (N,)
        D = self.lattice.D

        def _shift_one(psi_field: Array, e_i: Array) -> Array:
            """Roll psi by -e_i to get ψ(x + e_i)."""
            rolled = psi_field
            for dim in range(D):
                rolled = jnp.roll(rolled, -e_i[dim], axis=dim)
            return rolled

        psi_neighbors = jax.vmap(_shift_one, in_axes=(None, 0))(psi, e_int)

        force = jnp.einsum("n...,n,nd->...d", psi_neighbors, w, e_phys)

        force = -self.interaction_strength * psi[..., None] * force
        return force

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
        dist = self._dist_f
        rho = macro["rho"]
        u = macro["u"]

        # Shan-Chen velocity shift: u_eq = u + tau * F_SC / rho
        F_SC = self._shan_chen_force(rho)
        rho_squeezed = rho
        if rho.ndim >= 1 and rho.shape[-1] == 1:
            rho_squeezed = jnp.squeeze(rho, axis=-1)
        u_eq = u + self.tau * F_SC / (rho_squeezed[..., None] + 1e-10)

        # Collision with shifted equilibrium
        shifted_macro: MacroState = {**macro, "u": u_eq}
        f = state.dists[dist.label]
        f_eq = dist.equilibrium(shifted_macro, self.lattice)
        omega = 1.0 / self.tau
        f_star = f + omega * (f_eq - f)

        # Stream + BCs + obstacles
        f_shifted = self._interpolate_shift(f_star)
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
            from ..obstacles import apply_obstacles
            f_streamed = apply_obstacles(f_streamed, self.lattice, self.obstacles)
        return {dist.label: f_streamed}
