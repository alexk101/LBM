# equilibrium.py
"""Equilibrium distribution models. Swap equilibrium formulas without new distribution classes.

Schemes like MRT/TRT keep the same polynomial equilibrium and only change collision;
entropic, Levermore, and extended use different equilibrium formulas. This module
abstracts the equilibrium computation so one distribution (e.g. momentum) can use
polynomial, entropic, or extended equilibrium by passing a different EquilibriumModel.
"""
from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .definitions import MacroState
from .lattice import Lattice


class EquilibriumModel(eqx.Module):
    """Abstract: computes equilibrium distribution f_eq from macro state and lattice."""

    @abstractmethod
    def compute(self, macro: MacroState, lattice: Lattice) -> Array:
        """Equilibrium distribution (..., N). Reads required keys from macro (e.g. rho, u)."""
        raise NotImplementedError()


class PolynomialEquilibrium(EquilibriumModel):
    """Second-order Maxwell–Boltzmann expansion: f_i^eq = ρ w_i (1 + u·e_i/c_s² + ...).

    Same equilibrium used by standard BGK, MRT, TRT, and regularized schemes.
    Expects macro['rho'], macro['u'].
    """

    def compute(self, macro: MacroState, lattice: Lattice) -> Array:
        rho = macro["rho"]
        u = macro["u"]
        if rho.ndim >= 1 and rho.shape[-1] == 1:
            rho = jnp.squeeze(rho, axis=-1)
        e_dot_u = jnp.einsum("...c,dc->...d", u, lattice.velocities)
        u_sq = jnp.sum(u**2, axis=-1, keepdims=True)
        f_eq = (
            rho[..., None]
            * lattice.weights[lattice.indices][None, :]
            * (
                1.0
                + e_dot_u / lattice.c_s_sq
                + e_dot_u**2 / (2.0 * lattice.c_s_sq**2)
                - u_sq / (2.0 * lattice.c_s_sq)
            )
        )
        return f_eq
