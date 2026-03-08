# collision.py
"""Collision operators for LBM. Swap schemes without changing distributions.

Distributions remain responsible for equilibrium, lift, and relaxation_time();
the collision scheme decides how to relax f toward equilibrium (BGK, MRT, TRT, etc.).
"""
from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import Array

from .definitions import MacroState
from .distributions import Distribution
from .lattice import Lattice


class CollisionScheme(eqx.Module):
    """Abstract collision operator: post-collision f from (f, macro, distribution, lattice)."""

    @abstractmethod
    def collide(
        self,
        f: Array,
        macro: MacroState,
        distribution: Distribution,
        lattice: Lattice,
    ) -> Array:
        """Compute post-collision distribution (same shape as f).

        Args:
            f: Pre-collision distribution (..., N).
            macro: Current macroscopic state (rho, u, T, ...).
            distribution: Provides equilibrium and relaxation (τ or rates).
            lattice: Lattice configuration.

        Returns:
            Post-collision f with shape (..., N).
        """
        raise NotImplementedError()


class BGKCollision(CollisionScheme):
    """Single-relaxation-time (BGK) collision: f_star = f + ω(f_eq - f), ω = 1/τ."""

    def collide(
        self,
        f: Array,
        macro: MacroState,
        distribution: Distribution,
        lattice: Lattice,
    ) -> Array:
        f_eq = distribution.equilibrium(macro, lattice)
        omega = 1.0 / distribution.relaxation_time()
        return f + omega * (f_eq - f)
