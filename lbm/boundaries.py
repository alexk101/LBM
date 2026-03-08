# boundaries.py
"""Boundary conditions for LBM.

Original Zou–He (1997, Phys. Fluids 9, 1591): boundaries are based on *bounceback of
the non-equilibrium* part of the distribution:

  f_i - f_i^eq = f_opposite - f_opposite^eq   =>   f_i = f_i^eq + (f_opposite - f_opposite^eq)

So the unknown population is equilibrium at the boundary (ρ, u) plus the non-equilibrium
of the opposite (known) population. Both velocity and pressure BCs in the paper follow
from this plus mass/momentum consistency.

- *Periodic*: default from jnp.roll (no op in apply_boundaries).
- *No-slip*: full bounce-back (swap with opposite direction).
- *Free-slip*: specular reflection (mirror direction across the wall).
- *Velocity* (Zou–He): prescribed u; rho from mass, unknown f from non-eq bounceback.
  Our formulas (e.g. f1 = f3 + (2/3)*ρ*u_x for west) are the D2Q9 equivalent of the above.
- *Pressure*: prescribed ρ. Original Zou–He derives u from momentum + non-eq bounceback.
  We use a simpler stable approximation: unknown f = f^eq(ρ, u_interior) (no non-eq part).
- *Outflow*: zero-gradient (copy unknown from interior neighbor).
"""
from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .definitions import MacroState
from .lattice import Lattice

BoundaryKind = Literal[
    "periodic", "no_slip", "free_slip", "velocity", "pressure", "outflow"
]


class BoundarySpec(eqx.Module):
    """Specifies boundary type for each face of the domain.

    Use get(axis, side) to look up the kind. For "velocity" and "pressure" faces,
    pass prescribed values into apply_boundaries via boundary_velocity and
    boundary_pressure (dicts keyed by (axis, side)).
    """

    _items: tuple[tuple[int, int, str], ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        x: tuple[BoundaryKind, BoundaryKind] | None = None,
        y: tuple[BoundaryKind, BoundaryKind] | None = None,
        z: tuple[BoundaryKind, BoundaryKind] | None = None,
    ) -> None:
        """Specify BCs per axis: (low_side, high_side)."""
        items: list[tuple[int, int, str]] = []
        for axis, pair in enumerate([x, y, z]):
            if pair is None:
                continue
            for side, kind in enumerate(pair):
                items.append((axis, side, kind))
        self._items = tuple(items)

    def get(self, axis: int, side: int) -> BoundaryKind:
        """Return boundary kind for given axis and side (0=low, 1=high)."""
        for (a, s, k) in self._items:
            if a == axis and s == side:
                return k  # type: ignore[return-value]
        return "periodic"

    def has_any_non_periodic(self) -> bool:
        """True if any face is not periodic."""
        return any(k != "periodic" for (_, _, k) in self._items)

    def has_no_slip(self) -> bool:
        """True if any face uses no_slip (kept for backward compatibility)."""
        return any(k == "no_slip" for (_, _, k) in self._items)


def _apply_no_slip(
    f: Array,
    lattice: Lattice,
    axis: int,
    side: int,
    spatial_shape: tuple[int, ...],
) -> Array:
    D = len(spatial_shape)
    N = lattice.N
    vel = lattice.velocities
    opp = lattice.opposite_indices
    wall_index = 0 if side == 0 else spatial_shape[axis] - 1
    into_wall = (vel[:, axis] < 0) if side == 0 else (vel[:, axis] > 0)
    wall_slice_list: list[int | slice] = [slice(None)] * D
    wall_slice_list[axis] = wall_index
    wall_slice = tuple(wall_slice_list) + (slice(None),)
    f_wall = f[wall_slice]
    new_f_wall = jnp.where(
        into_wall[None, ...], f_wall[..., opp], f_wall
    )
    return f.at[wall_slice].set(new_f_wall)


def _apply_free_slip(
    f: Array,
    lattice: Lattice,
    axis: int,
    side: int,
    spatial_shape: tuple[int, ...],
) -> Array:
    D = len(spatial_shape)
    N = lattice.N
    vel = lattice.velocities
    mirror = lattice.mirror_indices_per_axis  # (D, N)
    mirror_axis = mirror[axis]
    wall_index = 0 if side == 0 else spatial_shape[axis] - 1
    into_wall = (vel[:, axis] < 0) if side == 0 else (vel[:, axis] > 0)
    wall_slice_list = [slice(None)] * D
    wall_slice_list[axis] = wall_index
    wall_slice = tuple(wall_slice_list) + (slice(None),)
    f_wall = f[wall_slice]
    new_f_wall = jnp.where(
        into_wall[None, ...], f_wall[..., mirror_axis], f_wall
    )
    return f.at[wall_slice].set(new_f_wall)


def _apply_outflow(
    f: Array,
    lattice: Lattice,
    axis: int,
    side: int,
    spatial_shape: tuple[int, ...],
) -> Array:
    D = len(spatial_shape)
    vel = lattice.velocities
    wall_index = 0 if side == 0 else spatial_shape[axis] - 1
    neighbor_index = wall_index + (1 if side == 0 else -1)
    into_wall = (vel[:, axis] < 0) if side == 0 else (vel[:, axis] > 0)
    wall_slice_list = [slice(None)] * D
    wall_slice_list[axis] = wall_index
    wall_slice = tuple(wall_slice_list) + (slice(None),)
    neighbor_slice_list = [slice(None)] * D
    neighbor_slice_list[axis] = neighbor_index
    neighbor_slice = tuple(neighbor_slice_list) + (slice(None),)
    f_wall = f[wall_slice]
    f_neighbor = f[neighbor_slice]
    new_f_wall = jnp.where(into_wall[None, ...], f_neighbor, f_wall)
    return f.at[wall_slice].set(new_f_wall)


def _zou_he_d2q9_west(
    f_wall: Array, u_x: Array, u_y: Array, rho: Array | None = None
) -> Array:
    """Unknown: 1, 5, 8. If rho is None, compute from mass; else use prescribed rho."""
    f0, f1, f2, f3, f4, f5, f6, f7, f8 = (
        f_wall[..., 0], f_wall[..., 1], f_wall[..., 2], f_wall[..., 3],
        f_wall[..., 4], f_wall[..., 5], f_wall[..., 6], f_wall[..., 7], f_wall[..., 8],
    )
    if rho is None:
        rho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - u_x)
    f1_new = f3 + (2.0 / 3.0) * rho * u_x
    f5_new = f7 - 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * u_x + (1.0 / 3.0) * rho * u_y
    f8_new = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rho * u_x - (1.0 / 3.0) * rho * u_y
    f_wall_new = f_wall.at[..., 1].set(f1_new).at[..., 5].set(f5_new).at[..., 8].set(f8_new)
    return f_wall_new


def _zou_he_d2q9_east(
    f_wall: Array, u_x: Array, u_y: Array, rho: Array | None = None
) -> Array:
    """Unknown: 3, 6, 7."""
    f0, f1, f2, f3, f4, f5, f6, f7, f8 = (
        f_wall[..., 0], f_wall[..., 1], f_wall[..., 2], f_wall[..., 3],
        f_wall[..., 4], f_wall[..., 5], f_wall[..., 6], f_wall[..., 7], f_wall[..., 8],
    )
    if rho is None:
        rho = (f0 + f1 + f2 + f4 + 2.0 * (f5 + f8)) / (1.0 + u_x)
    f3_new = f1 - (2.0 / 3.0) * rho * u_x
    f6_new = f8 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho * u_x - (1.0 / 3.0) * rho * u_y
    f7_new = f5 + 0.5 * (f2 - f4) - (1.0 / 6.0) * rho * u_x + (1.0 / 3.0) * rho * u_y
    f_wall_new = f_wall.at[..., 3].set(f3_new).at[..., 6].set(f6_new).at[..., 7].set(f7_new)
    return f_wall_new


def _zou_he_d2q9_south(
    f_wall: Array, u_x: Array, u_y: Array, rho: Array | None = None
) -> Array:
    """Unknown: 4, 7, 8 (e_y < 0)."""
    f0, f1, f2, f3, f4, f5, f6, f7, f8 = (
        f_wall[..., 0], f_wall[..., 1], f_wall[..., 2], f_wall[..., 3],
        f_wall[..., 4], f_wall[..., 5], f_wall[..., 6], f_wall[..., 7], f_wall[..., 8],
    )
    if rho is None:
        rho = (f0 + f1 + f2 + f3 + 2.0 * (f5 + f6)) / (1.0 - u_y)
    f4_new = f2 - (2.0 / 3.0) * rho * u_y
    f7_new = f5 - 0.5 * (f1 - f3) - (1.0 / 3.0) * rho * u_x - (1.0 / 6.0) * rho * u_y
    f8_new = f6 + 0.5 * (f1 - f3) - (1.0 / 3.0) * rho * u_x + (1.0 / 6.0) * rho * u_y
    f_wall_new = f_wall.at[..., 4].set(f4_new).at[..., 7].set(f7_new).at[..., 8].set(f8_new)
    return f_wall_new


def _zou_he_d2q9_north(
    f_wall: Array, u_x: Array, u_y: Array, rho: Array | None = None
) -> Array:
    """Unknown: 2, 5, 6 (e_y > 0)."""
    f0, f1, f2, f3, f4, f5, f6, f7, f8 = (
        f_wall[..., 0], f_wall[..., 1], f_wall[..., 2], f_wall[..., 3],
        f_wall[..., 4], f_wall[..., 5], f_wall[..., 6], f_wall[..., 7], f_wall[..., 8],
    )
    if rho is None:
        rho = (f0 + f1 + f3 + f4 + 2.0 * (f7 + f8)) / (1.0 + u_y)
    f2_new = f4 + (2.0 / 3.0) * rho * u_y
    f5_new = f7 + 0.5 * (f1 - f3) + (1.0 / 3.0) * rho * u_x + (1.0 / 6.0) * rho * u_y
    f6_new = f8 - 0.5 * (f1 - f3) + (1.0 / 3.0) * rho * u_x - (1.0 / 6.0) * rho * u_y
    f_wall_new = f_wall.at[..., 2].set(f2_new).at[..., 5].set(f5_new).at[..., 6].set(f6_new)
    return f_wall_new


def _apply_zou_he_velocity_2d(
    f: Array,
    lattice: Lattice,
    axis: int,
    side: int,
    spatial_shape: tuple[int, ...],
    u_face: Array,
) -> Array:
    """Zou–He velocity BC for D2Q9. u_face shape (..., 2) over the face."""
    if lattice.N != 9:
        raise NotImplementedError("Zou–He velocity BC is implemented only for D2Q9")
    D = len(spatial_shape)
    wall_index = 0 if side == 0 else spatial_shape[axis] - 1
    wall_slice_list = [slice(None)] * D
    wall_slice_list[axis] = wall_index
    wall_slice = tuple(wall_slice_list) + (slice(None),)
    f_wall = f[wall_slice]
    u_x = u_face[..., 0]
    u_y = u_face[..., 1]
    if axis == 0 and side == 0:
        f_wall_new = _zou_he_d2q9_west(f_wall, u_x, u_y)
    elif axis == 0 and side == 1:
        f_wall_new = _zou_he_d2q9_east(f_wall, u_x, u_y)
    elif axis == 1 and side == 0:
        f_wall_new = _zou_he_d2q9_south(f_wall, u_x, u_y)
    else:
        f_wall_new = _zou_he_d2q9_north(f_wall, u_x, u_y)
    return f.at[wall_slice].set(f_wall_new)


def _equilibrium_d2q9(rho: Array, u: Array, lattice: Lattice) -> Array:
    """Equilibrium distribution for D2Q9 from (rho, u). Same as ParticleDistribution.equilibrium."""
    if u.shape[-1] != 2:
        u = jnp.reshape(u, (*u.shape[:-1], 2))
    e_dot_u = jnp.einsum("...c,dc->...d", u, lattice.velocities)
    u_sq = jnp.sum(u**2, axis=-1, keepdims=True)
    rho_exp = rho[..., None] if rho.ndim < u.ndim else rho
    f_eq = (
        rho_exp
        * lattice.weights[lattice.indices][None, :]
        * (
            1.0
            + e_dot_u / lattice.c_s_sq
            + e_dot_u**2 / (2.0 * lattice.c_s_sq**2)
            - u_sq / (2.0 * lattice.c_s_sq)
        )
    )
    return f_eq


# Unknown population indices per (axis, side) for D2Q9: directions that stream from outside.
_D2Q9_UNKNOWN: dict[tuple[int, int], tuple[int, ...]] = {
    (0, 0): (1, 5, 8),   # west
    (0, 1): (3, 6, 7),   # east
    (1, 0): (4, 7, 8),   # south
    (1, 1): (2, 5, 6),   # north
}


def _apply_zou_he_pressure_2d(
    f: Array,
    lattice: Lattice,
    axis: int,
    side: int,
    spatial_shape: tuple[int, ...],
    rho_face: Array,
    u_interior: Array | None,
) -> Array:
    """Pressure BC: prescribed rho; set unknown populations to f_eq(rho, u_interior).

    Uses equilibrium at (rho_prescribed, u from interior neighbor). Stable; may
    show a small mass drift until the flow is developed.
    """
    if lattice.N != 9:
        raise NotImplementedError("Pressure BC is implemented only for D2Q9")
    D = len(spatial_shape)
    wall_index = 0 if side == 0 else spatial_shape[axis] - 1
    neighbor_index = wall_index + (1 if side == 0 else -1)
    wall_slice_list = [slice(None)] * D
    wall_slice_list[axis] = wall_index
    wall_slice = tuple(wall_slice_list) + (slice(None),)
    neighbor_slice_list = [slice(None)] * D
    neighbor_slice_list[axis] = neighbor_index
    neighbor_slice = tuple(neighbor_slice_list) + (slice(None),)
    f_wall = f[wall_slice]
    if rho_face.ndim >= 1 and rho_face.shape[-1] == 1:
        rho = jnp.squeeze(rho_face, axis=-1)
    else:
        rho = rho_face
    if u_interior is None:
        u_face = jnp.zeros((*rho.shape, 2))
    else:
        u_face = u_interior[neighbor_slice]
    if u_face.ndim == 1:
        u_face = u_face[None, :]
    f_eq = _equilibrium_d2q9(rho, u_face, lattice)
    unknown_idx = _D2Q9_UNKNOWN[(axis, side)]
    f_wall_new = f_wall
    for i in unknown_idx:
        f_wall_new = f_wall_new.at[..., i].set(f_eq[..., i])
    return f.at[wall_slice].set(f_wall_new)


def apply_boundaries(
    f_streamed: Array,
    lattice: Lattice,
    spec: BoundarySpec,
    macro: MacroState | None = None,
    boundary_velocity: dict[tuple[int, int], Array] | None = None,
    boundary_pressure: dict[tuple[int, int], Array] | None = None,
) -> Array:
    """Apply boundary conditions to a streamed distribution.

    Periodic faces are left unchanged. For "velocity" and "pressure" faces you must
    pass boundary_velocity or boundary_pressure keyed by (axis, side). Arrays must
    have shape matching the face (e.g. (Y,) or (Y, 2) for 2D x-face).
    """
    if not spec.has_any_non_periodic():
        return f_streamed

    D = lattice.D
    N = lattice.N
    shape = f_streamed.shape
    spatial_shape = tuple(shape[:D])
    assert shape == spatial_shape + (N,), f"{shape} vs {spatial_shape} + ({N},)"

    f = f_streamed
    boundary_velocity = boundary_velocity or {}
    boundary_pressure = boundary_pressure or {}
    macro = macro or {}

    for axis in range(D):
        for side in (0, 1):
            kind = spec.get(axis, side)
            if kind == "periodic":
                continue
            if kind == "no_slip":
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
                f = _apply_zou_he_velocity_2d(
                    f, lattice, axis, side, spatial_shape, u_face
                )
            elif kind == "pressure":
                rho_face = boundary_pressure.get((axis, side))
                if rho_face is None:
                    raise ValueError(
                        f"Boundary (axis={axis}, side={side}) is 'pressure' but no "
                        "boundary_pressure[(axis, side)] was provided."
                    )
                u = macro.get("u")
                f = _apply_zou_he_pressure_2d(
                    f, lattice, axis, side, spatial_shape, rho_face, u
                )
    return f
