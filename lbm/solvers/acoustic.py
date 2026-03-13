"""Linearized acoustic LBM solver with Guo source terms.

Single distribution F linearized around a background state (rho_0, u=0).
Source terms (monopole, dipole, quadrupole) are injected via Guo's forcing
scheme after collision.

Macroscopic quantities are corrected for the source:
    rho   = sum(f) + 0.5 * q             (density with monopole correction)
    u     = (sum(f * e) + 0.5 * F_src) / rho_0
    p     = c_s^2 * (rho - rho_0)         (acoustic pressure perturbation)

Source specification:
    Sources are ``AcousticSource`` modules stored in ``sources``.  Each source
    carries a type (monopole / dipole / quadrupole), spatial amplitude field,
    and a callable ``time_fn(t) -> scalar`` for temporal modulation.
"""
from __future__ import annotations

from typing import Callable, Literal

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ..boundaries import apply_boundaries
from ..definitions import LBMState, MacroState
from ..distributions import ParticleDistribution, Distribution
from ..lattice import Lattice
from .base import BaseSolver


class AcousticSource(eqx.Module):
    """Specification for a single acoustic source term.

    Attributes:
        kind: ``"monopole"``, ``"dipole"``, or ``"quadrupole"``.
        amplitude: Spatial amplitude field.
            - monopole:    scalar field (...,)
            - dipole:      vector field (..., D)
            - quadrupole:  tensor — longitudinal (..., D) and/or lateral (..., D)
        lateral_amplitude: For quadrupole lateral component.
        time_fn: Maps simulation time *t* (float) to a scalar modulation factor.
            Must be JIT-compatible (no Python-side effects).
        direction_fn: For rotating dipoles — maps *t* to a direction vector.
            If None, direction is static (embedded in amplitude).
        position_fn: For moving sources — maps *t* to a position tuple.
            If None, position is static (embedded in amplitude).
    """

    kind: Literal["monopole", "dipole", "quadrupole"] = eqx.field(static=True)
    amplitude: Array
    lateral_amplitude: Array | None = None
    time_fn: Callable[[float], float] = eqx.field(static=True, default=lambda t: 1.0)
    direction_fn: Callable[[float], Array] | None = eqx.field(static=True, default=None)
    position_fn: Callable[[float], tuple] | None = eqx.field(static=True, default=None)
    base_amplitude: float = 1.0
    spatial_shape: tuple[int, ...] = eqx.field(static=True, default=())


def _resolve_source(src: AcousticSource, t: float) -> tuple[Array, Array | None]:
    """Resolve dynamic amplitude for moving/rotating sources at time t."""
    amp = src.amplitude
    lat_amp = src.lateral_amplitude

    if src.direction_fn is not None and src.kind == "dipole":
        direction = src.direction_fn(t)
        amp = jnp.zeros_like(src.amplitude)
        if src.position_fn is not None:
            pos = src.position_fn(t)
            amp = amp.at[pos].set(src.base_amplitude * direction)
        else:
            nz = jnp.abs(src.amplitude) > 0
            nz_any = jnp.any(nz, axis=-1)
            amp = jnp.where(nz_any[..., None], src.base_amplitude * direction, 0.0)

    if src.position_fn is not None and src.direction_fn is None:
        pos = src.position_fn(t)
        if src.kind == "monopole":
            amp = jnp.zeros(src.spatial_shape)
            amp = amp.at[pos].set(src.base_amplitude)
        elif src.kind == "dipole":
            amp = jnp.zeros_like(src.amplitude)
            direction = src.amplitude[tuple(p for p in jnp.unravel_index(0, src.spatial_shape))]
            amp = amp.at[pos].set(direction)
        elif src.kind == "quadrupole":
            amp = jnp.zeros_like(src.amplitude)
            amp = amp.at[pos].set(src.base_amplitude)

    return amp, lat_amp


class AcousticSolver(BaseSolver):
    """Linearized acoustic LBM with Guo source terms.

    Args:
        rho_0: Background density.
        c_0: Speed of sound (for reporting; lattice c_s = 1/sqrt(3) is used internally).
        tau: Constant relaxation time.
        sources: Tuple of :class:`AcousticSource` specifications.
    """

    rho_0: float = 1.0
    c_0: float | None = None
    tau: float = 0.6
    sources: tuple[AcousticSource, ...] = ()

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
        cs = self.lattice.c_s_sq ** 0.5
        return {
            "rho_0": self.rho_0,
            "c_0": self.c_0 if self.c_0 is not None else cs,
            "tau": self.tau,
            "c_s": cs,
        }

    # ------------------------------------------------------------------
    # Guo source terms
    # ------------------------------------------------------------------

    def _guo_monopole(self, macro: MacroState, q: Array, t: float) -> Array:
        """S_i^mono = w_i * (1 - 1/(2τ)) * vel_factor * q(x,t)."""
        u = macro["u"]
        cs2 = self.lattice.c_s_sq
        cs4 = cs2 * cs2
        w = self.lattice.expanded_weights               # (N,)
        e = self.lattice.e                               # (N, D)
        prefactor = 1.0 - 0.5 / self.tau

        cu = jnp.einsum("nd,...d->...n", e, u)           # (..., N)
        u_sq = jnp.sum(u ** 2, axis=-1, keepdims=True)   # (..., 1)
        vel_factor = 1.0 + cu / cs2 + (cu ** 2 - cs2 * u_sq) / (2.0 * cs4)

        return w * prefactor * vel_factor * q[..., None]

    def _guo_dipole(self, macro: MacroState, F_src: Array, t: float) -> Array:
        """S_i^dip = w_i * (1 - 1/(2τ)) * force_factor."""
        u = macro["u"]
        cs2 = self.lattice.c_s_sq
        cs4 = cs2 * cs2
        w = self.lattice.expanded_weights
        e = self.lattice.e
        prefactor = 1.0 - 0.5 / self.tau

        cF = jnp.einsum("nd,...d->...n", e, F_src)
        cu = jnp.einsum("nd,...d->...n", e, u)
        uF = jnp.sum(u * F_src, axis=-1, keepdims=True)
        force_factor = cF / cs2 + (cu * cF - cs2 * uF) / cs4

        return w * prefactor * force_factor

    def _guo_quadrupole(
        self, macro: MacroState, B_src: Array | None, b_src: Array | None,
    ) -> Array:
        """S_i^quad via central-difference spatial derivatives."""
        cs2 = self.lattice.c_s_sq
        w = self.lattice.expanded_weights
        e = self.lattice.e
        D = self.lattice.D
        prefactor = 1.0 - 0.5 / self.tau

        def _deriv(field: Array, dim: int) -> Array:
            fwd = jnp.roll(field, -1, axis=dim)
            bwd = jnp.roll(field, 1, axis=dim)
            return (fwd - bwd) / 2.0

        force_components: list[Array] = []
        for i in range(D):
            F_i = jnp.zeros_like(macro["u"][..., 0:1])
            if B_src is not None:
                F_i = F_i + _deriv(B_src[..., i : i + 1], dim=i)
            if b_src is not None:
                for j in range(D):
                    if j != i:
                        F_i = F_i + _deriv(b_src[..., i : i + 1], dim=j)
            force_components.append(F_i)

        force_vec = jnp.concatenate(force_components, axis=-1)  # (..., D)
        c_dot_F = jnp.einsum("nd,...d->...n", e, force_vec)

        return w * prefactor * c_dot_F / cs2

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
        f = state.dists[dist.label]

        # BGK collision
        f_eq = dist.equilibrium(macro, self.lattice)
        omega = 1.0 / self.tau
        f_star = f - omega * (f - f_eq)

        t = 0.0
        for src in self.sources:
            amp, lat_amp = _resolve_source(src, t)
            modulation = src.time_fn(t)
            if src.kind == "monopole":
                q = amp * modulation
                f_star = f_star + self._guo_monopole(macro, q, t)
            elif src.kind == "dipole":
                F_src = amp * modulation
                f_star = f_star + self._guo_dipole(macro, F_src, t)
            elif src.kind == "quadrupole":
                B = amp * modulation if amp is not None else None
                b = lat_amp * modulation if lat_amp is not None else None
                f_star = f_star + self._guo_quadrupole(macro, B, b)

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

    def _lift_all(
        self, new_dists: dict[str, Array], macro: MacroState,
    ) -> MacroState:
        """Acoustic lift with source corrections (Eqs. 17-18 from Guo)."""
        f = new_dists[self._dist_f.label]
        rho = jnp.sum(f, axis=-1, keepdims=True)
        momentum = jnp.einsum("...i,ic->...c", f, self.lattice.e)

        for src in self.sources:
            modulation = src.time_fn(0.0)
            if src.kind == "monopole":
                q = src.amplitude * modulation
                if q.ndim < rho.ndim:
                    q = q[..., None]
                rho = rho + 0.5 * q
            elif src.kind == "dipole":
                F_src = src.amplitude * modulation
                momentum = momentum + 0.5 * F_src

        u = momentum / self.rho_0
        return {"rho": rho, "u": u}
