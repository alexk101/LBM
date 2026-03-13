"""Boundary conditions for LBM.

Zou-He (1997, Phys. Fluids 9, 1591): boundaries are based on *bounceback of
the non-equilibrium* part of the distribution:

  f_i = f_i^eq(rho, u) + (f_{opp(i)} - f_{opp(i)}^eq(rho, u))

Both velocity and pressure BCs follow from this plus mass/momentum consistency.

Streaming convention (pull):  f_i(x, t+dt) = f_i*(x - e_i, t).
After streaming at a boundary face, directions that point *into the domain
from outside* carry invalid (wrapped) data and must be overwritten.  At
side=0 (low face, e.g. x=0) the unknown set is ``e_i[axis] > 0``; at
side=1 (high face) it is ``e_i[axis] < 0``.

Supported BCs:
    periodic   — handled implicitly by ``jnp.roll`` (no-op here).
    no_slip    — full bounce-back: f_unknown = f_opposite.
    free_slip  — specular reflection: mirror normal component.
    outflow    — zero-gradient: copy from interior neighbor.
    velocity   — Zou-He: prescribed u; rho from mass; f via non-eq bounceback.
    pressure   — Zou-He: prescribed rho; u from momentum; f via non-eq bounceback.
    absorbing  — sponge layer: damp f toward f_eq(rho₀, 0) near boundaries.
    inlet_eq   — equilibrium inlet: set f = f_eq(rho, u) at face.
"""
from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .definitions import MacroState
from .equilibrium import PolynomialEquilibrium
from .lattice import Lattice

BoundaryKind = Literal[
    "periodic", "no_slip", "free_slip", "velocity", "pressure",
    "outflow", "absorbing", "inlet_eq",
]

_POLY_EQ = PolynomialEquilibrium()


class BoundarySpec(eqx.Module):
    """Boundary type for each face.  ``get(axis, side)`` returns the kind."""

    _items: tuple[tuple[int, int, str], ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        x: tuple[BoundaryKind, BoundaryKind] | None = None,
        y: tuple[BoundaryKind, BoundaryKind] | None = None,
        z: tuple[BoundaryKind, BoundaryKind] | None = None,
    ) -> None:
        items: list[tuple[int, int, str]] = []
        for axis, pair in enumerate([x, y, z]):
            if pair is None:
                continue
            for side, kind in enumerate(pair):
                items.append((axis, side, kind))
        self._items = tuple(items)

    def get(self, axis: int, side: int) -> BoundaryKind:
        for a, s, k in self._items:
            if a == axis and s == side:
                return k  # type: ignore[return-value]
        return "periodic"

    def has_any_non_periodic(self) -> bool:
        return any(k != "periodic" for (_, _, k) in self._items)

    def has_no_slip(self) -> bool:
        return any(k == "no_slip" for (_, _, k) in self._items)


# ======================================================================
# Helpers
# ======================================================================

def _wall_slice(axis: int, index: int, D: int):
    """Indexing tuple selecting a boundary face (all directions)."""
    sl: list[int | slice] = [slice(None)] * D
    sl[axis] = index
    return tuple(sl) + (slice(None),)


def _unknown_mask(lattice: Lattice, axis: int, side: int) -> Array:
    """Boolean mask (N,) — True for directions that are unknown at this face.

    With standard pull streaming, at side 0 the unknown directions have
    e_i[axis] > 0 (streamed from outside), and at side 1 they have
    e_i[axis] < 0.
    """
    vel = lattice.velocities[:, axis]
    return vel > 0 if side == 0 else vel < 0


def _f_eq(rho: Array, u: Array, lattice: Lattice) -> Array:
    """Polynomial equilibrium from (rho, u)."""
    if rho.ndim >= 1 and rho.shape[-1] != 1:
        rho = rho[..., None]
    if u.shape[-1] != lattice.D:
        u = jnp.reshape(u, (*u.shape[:-1], lattice.D))
    return _POLY_EQ.compute({"rho": rho, "u": u}, lattice)


# ======================================================================
# No-slip (full bounce-back)
# ======================================================================

def _apply_no_slip(
    f: Array, lattice: Lattice, axis: int, side: int, spatial_shape: tuple[int, ...],
) -> Array:
    D = len(spatial_shape)
    opp = lattice.opposite_indices
    wall_index = 0 if side == 0 else spatial_shape[axis] - 1
    unknown = _unknown_mask(lattice, axis, side)
    ws = _wall_slice(axis, wall_index, D)
    f_wall = f[ws]
    new_f_wall = jnp.where(unknown, f_wall[..., opp], f_wall)
    return f.at[ws].set(new_f_wall)


# ======================================================================
# Free-slip (specular reflection)
# ======================================================================

def _apply_free_slip(
    f: Array, lattice: Lattice, axis: int, side: int, spatial_shape: tuple[int, ...],
) -> Array:
    D = len(spatial_shape)
    mirror_axis = lattice.mirror_indices_per_axis[axis]
    wall_index = 0 if side == 0 else spatial_shape[axis] - 1
    unknown = _unknown_mask(lattice, axis, side)
    ws = _wall_slice(axis, wall_index, D)
    f_wall = f[ws]
    new_f_wall = jnp.where(unknown, f_wall[..., mirror_axis], f_wall)
    return f.at[ws].set(new_f_wall)


# ======================================================================
# Outflow (zero-gradient extrapolation)
# ======================================================================

def _apply_outflow(
    f: Array, lattice: Lattice, axis: int, side: int, spatial_shape: tuple[int, ...],
) -> Array:
    D = len(spatial_shape)
    wall_index = 0 if side == 0 else spatial_shape[axis] - 1
    neighbor_index = wall_index + (1 if side == 0 else -1)
    unknown = _unknown_mask(lattice, axis, side)
    ws = _wall_slice(axis, wall_index, D)
    ns_list: list[int | slice] = [slice(None)] * D
    ns_list[axis] = neighbor_index
    ns = tuple(ns_list) + (slice(None),)
    f_wall = f[ws]
    f_neighbor = f[ns]
    new_f_wall = jnp.where(unknown, f_neighbor, f_wall)
    return f.at[ws].set(new_f_wall)


# ======================================================================
# Zou-He velocity BC  (general, any D2Q9/D3Q19 face)
# ======================================================================

def _apply_zou_he_velocity(
    f: Array,
    lattice: Lattice,
    axis: int,
    side: int,
    spatial_shape: tuple[int, ...],
    u_face: Array,
) -> Array:
    """Zou-He velocity BC: prescribed u, derive rho from mass, set unknowns
    via non-equilibrium bounceback f_i = f_i^eq + (f_opp - f_opp^eq).

    Works for any lattice; no hard-coded per-face formulas.
    """
    D = len(spatial_shape)
    opp = lattice.opposite_indices
    vel = lattice.velocities                      # (N, D)
    wall_index = 0 if side == 0 else spatial_shape[axis] - 1
    ws = _wall_slice(axis, wall_index, D)
    f_wall = f[ws]                                # (...face_shape, N)
    unknown = _unknown_mask(lattice, axis, side)   # (N,)

    # Normal velocity component (signed: positive = into domain)
    u_n = u_face[..., axis]
    if side == 0:
        sign = 1.0
    else:
        sign = -1.0

    # Density from mass conservation:
    # rho = [Σ_zero f_i + 2 * Σ_known_nonzero f_i] / (1 ∓ u_n)
    # where "zero" means e_i[axis]==0, "known_nonzero" = e_i[axis] pointing
    # from interior (opposite sign of unknown).
    e_ax = vel[:, axis]                            # (N,)
    is_zero = e_ax == 0
    is_known_nz = (~unknown) & (~is_zero)

    S_zero = jnp.sum(f_wall * is_zero, axis=-1)
    S_known_nz = jnp.sum(f_wall * is_known_nz, axis=-1)

    rho = (S_zero + 2.0 * S_known_nz) / (1.0 - sign * u_n)

    # Equilibrium at (rho, u) and opposite
    f_eq_wall = _f_eq(rho, u_face, lattice)
    f_eq_opp = f_eq_wall[..., opp]

    # Non-equilibrium bounceback: f_i = f_i^eq + (f_opp - f_opp^eq)
    f_neqbb = f_eq_wall + (f_wall[..., opp] - f_eq_opp)

    new_f_wall = jnp.where(unknown, f_neqbb, f_wall)
    return f.at[ws].set(new_f_wall)


# ======================================================================
# Zou-He pressure BC  (general, any D2Q9/D3Q19 face)
# ======================================================================

def _apply_zou_he_pressure(
    f: Array,
    lattice: Lattice,
    axis: int,
    side: int,
    spatial_shape: tuple[int, ...],
    rho_face: Array,
    macro: MacroState | None,
) -> Array:
    """Zou-He pressure BC: prescribed rho, derive u from momentum, set unknowns
    via non-equilibrium bounceback.

    The normal velocity is computed from the mass equation:
        u_n = 1 - (Σ_zero f + 2 * Σ_known_nz f) / rho     (side 0)
        u_n = -1 + (Σ_zero f + 2 * Σ_known_nz f) / rho     (side 1)

    Transverse velocity is taken from the interior neighbor (zero-gradient
    extrapolation), which is standard practice and matches the approach used
    in many validated LBM codes.
    """
    D = len(spatial_shape)
    opp = lattice.opposite_indices
    vel = lattice.velocities
    wall_index = 0 if side == 0 else spatial_shape[axis] - 1
    neighbor_index = wall_index + (1 if side == 0 else -1)
    ws = _wall_slice(axis, wall_index, D)
    f_wall = f[ws]
    unknown = _unknown_mask(lattice, axis, side)

    if rho_face.ndim >= 1 and rho_face.shape[-1] == 1:
        rho = jnp.squeeze(rho_face, axis=-1)
    else:
        rho = rho_face
    rho_safe = jnp.maximum(rho, 1e-10)

    e_ax = vel[:, axis]
    is_zero = e_ax == 0
    is_known_nz = (~unknown) & (~is_zero)

    S_zero = jnp.sum(f_wall * is_zero, axis=-1)
    S_known_nz = jnp.sum(f_wall * is_known_nz, axis=-1)

    # Normal velocity from mass conservation
    if side == 0:
        u_n = 1.0 - (S_zero + 2.0 * S_known_nz) / rho_safe
    else:
        u_n = -1.0 + (S_zero + 2.0 * S_known_nz) / rho_safe

    # Transverse velocity from interior neighbor
    ns_list: list[int | slice] = [slice(None)] * D
    ns_list[axis] = neighbor_index
    ns = tuple(ns_list)
    if macro is not None and "u" in macro:
        u_interior = macro["u"][ns]
    else:
        u_interior = jnp.zeros((*rho.shape, D))

    # Build full velocity: replace normal component with the derived one
    u_face_full = u_interior.at[..., axis].set(u_n)

    # Non-equilibrium bounceback
    f_eq_wall = _f_eq(rho, u_face_full, lattice)
    f_eq_opp = f_eq_wall[..., opp]
    f_neqbb = f_eq_wall + (f_wall[..., opp] - f_eq_opp)

    new_f_wall = jnp.where(unknown, f_neqbb, f_wall)
    return f.at[ws].set(new_f_wall)


# ======================================================================
# Absorbing layer (sponge)
# ======================================================================

class AbsorbingLayerSpec(eqx.Module):
    """Specification for absorbing (sponge) layer on one or more faces.

    The damping coefficient increases quadratically from 0 at the inner
    edge of the buffer to ``sigma_max`` at the domain boundary:

        σ(d) = sigma_max * ((width - d) / width)²   for d < width

    The distribution is relaxed toward equilibrium at rest:
        f_damped = f - σ * (f - f_eq(rho₀, 0))

    Attributes:
        sigma: Pre-computed damping field, shape ``spatial_shape``.
               Build via :meth:`build`.
        rho_0: Reference (background) density for the equilibrium target.
    """

    sigma: Array
    rho_0: float = 1.0

    @staticmethod
    def build(
        spatial_shape: tuple[int, ...],
        *,
        rho_0: float = 1.0,
        faces: dict[tuple[int, int], dict] | None = None,
    ) -> AbsorbingLayerSpec:
        """Construct from per-face specifications.

        Args:
            spatial_shape: Domain shape, e.g. (Nx, Ny).
            rho_0: Background density.
            faces: Mapping ``(axis, side) -> {"width": int, "sigma_max": float}``.
        """
        D = len(spatial_shape)
        sigma = jnp.zeros(spatial_shape)
        if faces is None:
            return AbsorbingLayerSpec(sigma=sigma, rho_0=rho_0)

        for (ax, sd), params in faces.items():
            width = params["width"]
            sigma_max = params["sigma_max"]
            coords = jnp.arange(spatial_shape[ax], dtype=jnp.float32)
            if sd == 1:
                coords = spatial_shape[ax] - 1.0 - coords
            shape = [1] * D
            shape[ax] = spatial_shape[ax]
            dist = jnp.reshape(coords, shape)
            dist = jnp.broadcast_to(dist, spatial_shape)
            in_buffer = dist < width
            layer = sigma_max * ((width - dist) / width) ** 2
            layer = jnp.where(in_buffer, layer, 0.0)
            sigma = jnp.maximum(sigma, layer)

        return AbsorbingLayerSpec(sigma=sigma, rho_0=rho_0)


def _apply_absorbing(
    f: Array, lattice: Lattice, absorbing: AbsorbingLayerSpec,
) -> Array:
    """Apply sponge damping: f -= σ * (f - f_eq(rho₀, 0))."""
    D = lattice.D
    spatial_shape = f.shape[:D]
    rho = jnp.full(spatial_shape + (1,), absorbing.rho_0)
    u = jnp.zeros(spatial_shape + (D,))
    f_eq_rest = _POLY_EQ.compute({"rho": rho, "u": u}, lattice)
    sigma = absorbing.sigma[..., None]  # (..., 1) for broadcast over N
    return f - sigma * (f - f_eq_rest)


# ======================================================================
# Equilibrium inlet
# ======================================================================

def _apply_inlet_eq(
    f: Array,
    lattice: Lattice,
    axis: int,
    side: int,
    spatial_shape: tuple[int, ...],
    inlet_state: MacroState,
) -> Array:
    """Equilibrium inlet: set f = f_eq(rho, u) at the boundary face."""
    D = len(spatial_shape)
    wall_index = 0 if side == 0 else spatial_shape[axis] - 1
    ws = _wall_slice(axis, wall_index, D)
    rho = inlet_state["rho"]
    u = inlet_state["u"]
    if rho.ndim < D:
        rho = rho[..., None]
    f_eq_face = _f_eq(rho, u, lattice)

    # Select the face slice (rho/u might be full-domain or already face-shaped)
    ns_list: list[int | slice] = [slice(None)] * D
    ns_list[axis] = wall_index
    face_idx = tuple(ns_list)
    try:
        f_eq_at_face = f_eq_face[face_idx]
    except IndexError:
        f_eq_at_face = f_eq_face

    f_wall = f[ws]
    unknown = _unknown_mask(lattice, axis, side)
    new_f_wall = jnp.where(unknown, f_eq_at_face, f_wall)
    return f.at[ws].set(new_f_wall)


# ======================================================================
# Main dispatcher
# ======================================================================

def apply_boundaries(
    f_streamed: Array,
    lattice: Lattice,
    spec: BoundarySpec,
    macro: MacroState | None = None,
    boundary_velocity: dict[tuple[int, int], Array] | None = None,
    boundary_pressure: dict[tuple[int, int], Array] | None = None,
    absorbing_spec: AbsorbingLayerSpec | None = None,
    inlet_states: dict[tuple[int, int], MacroState] | None = None,
) -> Array:
    """Apply all boundary conditions to a streamed distribution.

    Periodic faces need no action (handled by ``jnp.roll`` in streaming).
    """
    if not spec.has_any_non_periodic() and absorbing_spec is None:
        return f_streamed

    D = lattice.D
    N = lattice.N
    shape = f_streamed.shape
    spatial_shape = tuple(shape[:D])

    f = f_streamed
    boundary_velocity = boundary_velocity or {}
    boundary_pressure = boundary_pressure or {}
    inlet_states = inlet_states or {}
    macro = macro or {}

    for axis in range(D):
        for side in (0, 1):
            kind = spec.get(axis, side)
            if kind == "periodic":
                continue
            elif kind == "no_slip":
                f = _apply_no_slip(f, lattice, axis, side, spatial_shape)
            elif kind == "free_slip":
                f = _apply_free_slip(f, lattice, axis, side, spatial_shape)
            elif kind == "outflow":
                f = _apply_outflow(f, lattice, axis, side, spatial_shape)
            elif kind == "velocity":
                u_face = boundary_velocity.get((axis, side))
                if u_face is None:
                    raise ValueError(
                        f"Boundary (axis={axis}, side={side}) is 'velocity' but no "
                        "boundary_velocity[(axis, side)] was provided."
                    )
                f = _apply_zou_he_velocity(
                    f, lattice, axis, side, spatial_shape, u_face,
                )
            elif kind == "pressure":
                rho_face = boundary_pressure.get((axis, side))
                if rho_face is None:
                    raise ValueError(
                        f"Boundary (axis={axis}, side={side}) is 'pressure' but no "
                        "boundary_pressure[(axis, side)] was provided."
                    )
                f = _apply_zou_he_pressure(
                    f, lattice, axis, side, spatial_shape, rho_face, macro,
                )
            elif kind == "absorbing":
                pass  # handled globally below
            elif kind == "inlet_eq":
                inlet_state = inlet_states.get((axis, side))
                if inlet_state is None:
                    raise ValueError(
                        f"Boundary (axis={axis}, side={side}) is 'inlet_eq' but no "
                        "inlet_states[(axis, side)] was provided."
                    )
                f = _apply_inlet_eq(
                    f, lattice, axis, side, spatial_shape, inlet_state,
                )

    if absorbing_spec is not None:
        f = _apply_absorbing(f, lattice, absorbing_spec)

    return f
