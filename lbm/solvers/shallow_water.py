"""Shallow-water LBM solver.

Single distribution F where density represents water height *h*.  A gravity
forcing term is added after collision:

    F_i += w_i * (e_{i,y} - u_y) * g * h / c_s^2

Physical parameters:
    gravity  — gravitational acceleration in lattice units.
    height_0 — reference (equilibrium) water height.
    Fr       — Froude number Fr = V0 / sqrt(g * h0).  If provided, V0 is
               derived from it; otherwise V0 defaults to 0.1.
    Re       — Reynolds number (used to compute tau via nu = V0 * L / Re).

Stability:
    Wave celerity c = sqrt(g * h0) must be < lattice sound speed 1/sqrt(3).
"""
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ..boundaries import apply_boundaries
from ..definitions import DTYPE, LBMState, MacroState
from ..distributions import ParticleDistribution, Distribution
from ..lattice import Lattice
from .base import BaseSolver


class ShallowWaterSolver(BaseSolver):
    """Shallow-water LBM with gravity source term.

    Args:
        gravity: Gravitational acceleration (lattice units).
        height_0: Reference water height.
        Fr: Froude number (optional; determines V0 = Fr * sqrt(g * h0)).
        Re: Reynolds number (determines viscosity / tau).
        char_length: Characteristic length for Re (default 1.0).
        gravity_axis: Spatial axis index for the gravity direction (default 1 = y).
    """

    gravity: float = 0.01
    height_0: float = 1.0
    Fr: float | None = None
    Re: float = 100.0
    char_length: float = 1.0
    gravity_axis: int = 1

    _dist_f: ParticleDistribution = eqx.field(init=False)

    def __post_init__(self) -> None:
        cs2 = self.lattice.c_s_sq
        celerity = (self.gravity * self.height_0) ** 0.5
        lattice_cs = cs2 ** 0.5
        if celerity >= lattice_cs:
            raise ValueError(
                f"ShallowWater stability: wave celerity {celerity:.4f} >= "
                f"lattice sound speed {lattice_cs:.4f}. "
                "Reduce gravity or height_0."
            )
        V0 = self.Fr * celerity if self.Fr is not None else 0.1
        if V0 > 0.2:
            import warnings
            warnings.warn(
                f"ShallowWater: V0={V0:.3f} > 0.2; compressibility errors likely.",
                stacklevel=2,
            )
        nu = V0 * self.char_length / self.Re
        tau = nu / cs2 + 0.5
        object.__setattr__(self, "_dist_f", ParticleDistribution(tau=tau))
        super().__post_init__()

    @property
    def distributions(self) -> tuple[Distribution, ...]:
        return (self._dist_f,)

    def system_variables(self) -> dict[str, float]:
        cs2 = self.lattice.c_s_sq
        celerity = (self.gravity * self.height_0) ** 0.5
        V0 = self.Fr * celerity if self.Fr is not None else 0.1
        nu = V0 * self.char_length / self.Re
        tau = nu / cs2 + 0.5
        return {
            "gravity": self.gravity,
            "height_0": self.height_0,
            "celerity": celerity,
            "Fr": self.Fr if self.Fr is not None else V0 / max(celerity, 1e-12),
            "Re": self.Re,
            "V0": V0,
            "nu": nu,
            "tau": tau,
        }

    def _gravity_source(self, f: Array, macro: MacroState) -> Array:
        """Gravity forcing: S_i = w_i * (e_{i,g} - u_g) * g * h / c_s^2."""
        rho = macro["rho"]
        u = macro["u"]
        if rho.ndim >= 1 and rho.shape[-1] == 1:
            rho = jnp.squeeze(rho, axis=-1)

        g_ax = self.gravity_axis
        e_g = self.lattice.e[:, g_ax]                        # (N,)
        u_g = u[..., g_ax]                                  # (...,)
        cs2 = self.lattice.c_s_sq
        w = self.lattice.expanded_weights                   # (N,)

        force = (
            w
            * (e_g - u_g[..., None])
            * (self.gravity * rho[..., None])
            / cs2
        )
        return force

    def _collide_and_stream(
        self,
        state: LBMState,
        macro: MacroState,
        boundary_velocity: dict[tuple[int, int], Array] | None = None,
        boundary_pressure: dict[tuple[int, int], Array] | None = None,
    ) -> dict[str, Array]:
        dist = self._dist_f
        f = state.dists[dist.label]

        f_star = self.collision.collide(f, macro, dist, self.lattice)
        f_star = f_star + self._gravity_source(f, macro)
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
