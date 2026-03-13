from __future__ import annotations

from enum import Enum

import equinox as eqx
import jax.numpy as jnp
from jax import Array

# Internal type for macro fields passed to equilibrium models.
# Kept as a plain dict so equilibrium models don't need to depend on LBMState.
MacroState = dict[str, Array]


class LBMState(eqx.Module):
    """Full simulation state as a JAX pytree.

    Typed fields give IDE autocompletion and catch key errors at construction.
    Because this is an eqx.Module, jax.grad / jax.tree.map / jax.lax.fori_loop
    all work out of the box.

    Attributes:
        rho: Density field, shape (..., 1).
        u: Velocity field, shape (..., D).
        T: Temperature field, shape (..., 1), or None for athermal simulations.
        dists: Distribution arrays keyed by label (e.g. {"F": ..., "G": ...}).
    """

    rho: Array
    u: Array
    T: Array | None = None
    dists: dict[str, Array] = eqx.field(default_factory=dict)

    @property
    def macro(self) -> MacroState:
        """Extract macro dict for equilibrium / collision computation."""
        m: MacroState = {"rho": self.rho, "u": self.u}
        if self.T is not None:
            m["T"] = self.T
        return m

    # --- dict-like access for plotting / logging code ---

    def __getitem__(self, key: str) -> Array:
        if key == "rho":
            return self.rho
        if key == "u":
            return self.u
        if key == "T":
            if self.T is None:
                raise KeyError("T")
            return self.T
        if key in self.dists:
            return self.dists[key]
        raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        if key in ("rho", "u"):
            return True
        if key == "T":
            return self.T is not None
        return key in self.dists


class Level(Enum):
    """Field level indicates the physical abstraction layer."""

    MACROSCOPIC = "macroscopic"  # Density, velocity, temperature
    MICROSCOPIC = "microscopic"  # Distribution functions f_i


PLOT_DPI: int = 150
TAU_MIN: float = 0.5

DTYPE = jnp.float32
DTYPE_LOW = jnp.float32

R = 8.31446261  # J/(mol*K)
