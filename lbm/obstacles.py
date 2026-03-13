"""Obstacle definitions for LBM: immersed bodies and thin walls.

Obstacles interact with the solver by providing a boolean mask and a
``bounce_back`` method that modifies post-streaming distributions in-place
(functionally, returning the updated array).

Supported obstacle types:
    Cylinder    — solid circle (2D) or cylinder (3D) with full-way bounce-back.
    RigidWall   — infinitely thin wall using mid-link bounce-back.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .lattice import Lattice


class Obstacle(eqx.Module, ABC):
    """Base class for LBM obstacles."""

    mask: Array  # boolean, shape = spatial_shape

    @abstractmethod
    def bounce_back(self, f: Array, lattice: Lattice) -> Array:
        """Apply bounce-back to post-streaming distribution *f*.

        Args:
            f: Distribution array, shape ``(*spatial_shape, N)``.
            lattice: Lattice definition (provides ``opposite_indices``).

        Returns:
            Modified distribution with the same shape.
        """
        ...


class Cylinder(Obstacle):
    """Solid circular (2D) or cylindrical (3D) obstacle.

    Uses full-way bounce-back: populations inside the obstacle are swapped
    with their opposite directions, and interior nodes are zeroed.

    Args:
        center: Center coordinates, length D.
        radius: Radius in lattice units.
        spatial_shape: Domain shape.
        length: Axial half-length for 3D cylinders (None for 2D).
    """

    mask: Array
    center: tuple[int, ...]
    radius: int

    def __init__(
        self,
        center: tuple[int, ...] | list[int],
        radius: int,
        spatial_shape: tuple[int, ...],
        length: int | None = None,
    ) -> None:
        center_t = tuple(int(c) for c in center)
        D = len(spatial_shape)

        grids = jnp.meshgrid(
            *[jnp.arange(s) for s in spatial_shape], indexing="ij",
        )
        azimuthal = jnp.stack(grids[:2], axis=-1)
        center_2d = jnp.array(center_t[:2], dtype=jnp.float32)
        if D == 3:
            dist = jnp.sqrt(jnp.sum((azimuthal - center_2d) ** 2, axis=-1))
        else:
            dist = jnp.sqrt(jnp.sum((azimuthal - center_2d) ** 2, axis=-1))
        base_mask = dist <= radius

        row_count = jnp.sum(base_mask, axis=1, keepdims=True)
        col_count = jnp.sum(base_mask, axis=0, keepdims=True)
        base_mask = base_mask & (row_count > 1) & (col_count > 1)

        if D == 3 and length is not None:
            z_grid = grids[2]
            z_center = center_t[2]
            base_mask = base_mask & (jnp.abs(z_grid - z_center) <= length / 2)

        object.__setattr__(self, "mask", base_mask)
        object.__setattr__(self, "center", center_t)
        object.__setattr__(self, "radius", radius)

    def bounce_back(self, f: Array, lattice: Lattice) -> Array:
        opp = lattice.opposite_indices
        f_bounced = f[..., opp]
        mask_expanded = self.mask[..., None]
        return jnp.where(mask_expanded, f_bounced, f)


class RigidWall(Obstacle):
    """Thin rigid wall using mid-link bounce-back.

    The wall sits between two rows of fluid nodes (at ``position-1`` and
    ``position``) along the given ``axis``, spanning ``[start, end)`` on
    the perpendicular axis.  After streaming, populations that crossed the
    wall are swapped with their opposites.

    Args:
        axis: Axis perpendicular to the wall (0=x, 1=y).
        position: Coordinate along ``axis`` where the wall is located.
        start: Start coordinate on the perpendicular axis.
        end: End coordinate on the perpendicular axis.
        spatial_shape: Domain shape.
    """

    mask: Array  # empty for thin walls
    axis: int = eqx.field(static=True)
    position: int = eqx.field(static=True)
    start: int = eqx.field(static=True)
    end: int = eqx.field(static=True)

    _mask_below: Array
    _mask_above: Array

    def __init__(
        self,
        axis: int,
        position: int,
        start: int,
        end: int,
        spatial_shape: tuple[int, ...],
    ) -> None:
        D = len(spatial_shape)
        grids = jnp.meshgrid(
            *[jnp.arange(s) for s in spatial_shape], indexing="ij",
        )

        axis_grid = grids[axis]
        perp_axis = 1 - axis
        perp_grid = grids[perp_axis]

        in_wall_extent = (perp_grid >= start) & (perp_grid < end)
        mask_below = (axis_grid == position - 1) & in_wall_extent
        mask_above = (axis_grid == position) & in_wall_extent

        object.__setattr__(self, "mask", jnp.zeros(spatial_shape, dtype=jnp.bool_))
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "_mask_below", mask_below)
        object.__setattr__(self, "_mask_above", mask_above)

    def bounce_back(self, f: Array, lattice: Lattice) -> Array:
        opp = lattice.opposite_indices
        vel_ax = lattice.velocities[:, self.axis]
        is_positive = vel_ax > 0  # (N,) boolean mask

        f_opp = f[..., opp]

        crossed_up = f * self._mask_above[..., None]
        crossed_down = f_opp * self._mask_below[..., None]

        new_f = jnp.where(
            is_positive & self._mask_below[..., None],
            crossed_down,
            f,
        )
        new_f_opp = jnp.where(
            is_positive & self._mask_above[..., None],
            crossed_up,
            new_f[..., opp],
        )
        new_f = new_f.at[..., opp].set(new_f_opp)
        return new_f


def apply_obstacles(
    f: Array, lattice: Lattice, obstacles: tuple[Obstacle, ...],
) -> Array:
    """Apply bounce-back for all obstacles to a post-streaming distribution."""
    for obs in obstacles:
        f = obs.bounce_back(f, lattice)
    return f
