"""Per-experiment initial condition functions.

Each function takes ``(cfg, solver)`` and returns an ``LBMState``.
Register new experiments by adding a function and mapping its name in ``REGISTRY``.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array

from lbm.definitions import LBMState, MacroState

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from lbm.solvers.base import BaseSolver


def _make_state(macro: MacroState, solver: BaseSolver) -> LBMState:
    dists = {d.label: d.equilibrium(macro, solver.lattice) for d in solver.distributions}
    return LBMState(
        rho=macro["rho"], u=macro["u"], T=macro.get("T"), dists=dists,
    )


# ======================================================================
# Double shear layer
# ======================================================================

def double_shear_layer(cfg: DictConfig, solver: BaseSolver) -> LBMState:
    shape = tuple(cfg.simulation.spatial_shape)
    Nx, Ny = shape[0], shape[1]
    Re = cfg.solver.Re
    Ma = cfg.solver.Ma
    cs = solver.lattice.c_s_sq ** 0.5
    V0 = Ma * cs

    lam = cfg.experiment.get("lam", 80.0)
    epsilon = cfg.experiment.get("epsilon", 0.05)

    x = jnp.linspace(0, 1, Nx, endpoint=False)
    y = jnp.linspace(0, 1, Ny, endpoint=False)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    ux = jnp.where(
        Y < 0.5,
        V0 * jnp.tanh(lam * (Y - 0.25)),
        V0 * jnp.tanh(lam * (0.75 - Y)),
    )
    uy = epsilon * V0 * jnp.sin(2.0 * math.pi * (X + 0.25))

    rho = jnp.ones(shape)[..., None]
    u = jnp.stack([ux, uy], axis=-1)
    return _make_state({"rho": rho, "u": u}, solver)


# ======================================================================
# Shallow water dam break
# ======================================================================

def shallow_water_dam_break(cfg: DictConfig, solver: BaseSolver) -> LBMState:
    shape = tuple(cfg.simulation.spatial_shape)
    Nx, Ny = shape[0], shape[1]
    h0 = cfg.solver.height_0 if cfg.solver.height_0 is not None else 1.0
    perturbation_radius = cfg.experiment.get("perturbation_radius", 5)
    perturbation_amp = cfg.experiment.get("perturbation_amplitude", 0.5)

    cx, cy = Nx // 2, Ny // 2
    x = jnp.arange(Nx)
    y = jnp.arange(Ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    in_square = (jnp.abs(X - cx) <= perturbation_radius) & (jnp.abs(Y - cy) <= perturbation_radius)
    rho = jnp.where(in_square, h0 + perturbation_amp, h0)[..., None]
    u = jnp.zeros((*shape, 2))
    return _make_state({"rho": rho, "u": u}, solver)


# ======================================================================
# Taylor-Green vortex 3D
# ======================================================================

def taylor_green_3d(cfg: DictConfig, solver: BaseSolver) -> LBMState:
    shape = tuple(cfg.simulation.spatial_shape)
    Nx, Ny, Nz = shape
    Ma = cfg.solver.Ma
    cs = solver.lattice.c_s_sq ** 0.5
    V0 = Ma * cs

    x = jnp.linspace(0, 2 * math.pi, Nx, endpoint=False)
    y = jnp.linspace(0, 2 * math.pi, Ny, endpoint=False)
    z = jnp.linspace(0, 2 * math.pi, Nz, endpoint=False)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    ux = V0 * jnp.sin(X) * jnp.cos(Y) * jnp.cos(Z)
    uy = -V0 * jnp.cos(X) * jnp.sin(Y) * jnp.cos(Z)
    uz = jnp.zeros_like(ux)

    p = 1.0 + (V0 ** 2 / 16.0) * (jnp.cos(2 * X) + jnp.cos(2 * Y)) * (jnp.cos(2 * Z) + 2)
    rho = p[..., None]
    u = jnp.stack([ux, uy, uz], axis=-1)
    return _make_state({"rho": rho, "u": u}, solver)


# ======================================================================
# SOD shock tube
# ======================================================================

def sod_subsonic(cfg: DictConfig, solver: BaseSolver) -> LBMState:
    shape = tuple(cfg.simulation.spatial_shape)
    Nx, Ny = shape[0], shape[1]
    mid = Nx // 2

    rho_L = cfg.experiment.get("rho_L", 0.5)
    rho_R = cfg.experiment.get("rho_R", 2.0)
    T_L = cfg.experiment.get("T_L", 0.2)
    T_R = cfg.experiment.get("T_R", 0.025)

    rho = jnp.ones(shape)
    rho = rho.at[:mid, :].set(rho_L)
    rho = rho.at[mid:, :].set(rho_R)
    rho = rho[..., None]

    T = jnp.ones(shape)
    T = T.at[:mid, :].set(T_L)
    T = T.at[mid:, :].set(T_R)
    T = T[..., None]

    u = jnp.zeros((*shape, 2))
    return _make_state({"rho": rho, "u": u, "T": T}, solver)


def sod_transonic(cfg: DictConfig, solver: BaseSolver) -> LBMState:
    shape = tuple(cfg.simulation.spatial_shape)
    Nx, Ny = shape[0], shape[1]
    mid = Nx // 2

    rho_L = cfg.experiment.get("rho_L", 1.0)
    rho_R = cfg.experiment.get("rho_R", 0.125)
    T_L = cfg.experiment.get("T_L", 0.2)
    T_R = cfg.experiment.get("T_R", 0.16)

    rho = jnp.ones(shape)
    rho = rho.at[:mid, :].set(rho_L)
    rho = rho.at[mid:, :].set(rho_R)
    rho = rho[..., None]

    T = jnp.ones(shape)
    T = T.at[:mid, :].set(T_L)
    T = T.at[mid:, :].set(T_R)
    T = T[..., None]

    u = jnp.zeros((*shape, 2))
    return _make_state({"rho": rho, "u": u, "T": T}, solver)


# ======================================================================
# Flow around cylinder (compressible)
# ======================================================================

def cylinder_compressible(cfg: DictConfig, solver: BaseSolver) -> LBMState:
    shape = tuple(cfg.simulation.spatial_shape)
    T0 = cfg.experiment.get("T0", 0.2)
    rho0 = cfg.experiment.get("rho0", 1.0)

    if solver.lattice.is_shifted:
        U0 = float(solver.lattice.shifts[0])
    else:
        Ma = cfg.experiment.get("Ma0", 0.15)
        cs = solver.lattice.c_s_sq ** 0.5
        U0 = Ma * cs

    rho = jnp.full((*shape, 1), rho0)
    T = jnp.full((*shape, 1), T0)
    u = jnp.zeros((*shape, 2))
    u = u.at[..., 0].set(U0)

    return _make_state({"rho": rho, "u": u, "T": T}, solver)


# ======================================================================
# Flow around cylinder (acoustic / isothermal)
# ======================================================================

def cylinder_acoustic(cfg: DictConfig, solver: BaseSolver) -> LBMState:
    shape = tuple(cfg.simulation.spatial_shape)
    Ma = cfg.solver.get("Ma", cfg.experiment.get("Ma", 0.173))
    rho0 = cfg.solver.get("rho_0", 1.0)
    cs = solver.lattice.c_s_sq ** 0.5
    V0 = Ma * cs

    rho = jnp.full((*shape, 1), rho0)
    u = jnp.zeros((*shape, 2))
    u = u.at[..., 0].set(V0)

    if solver.obstacles:
        mask = solver.obstacles[0].mask
        u = u * (~mask)[..., None]

    return _make_state({"rho": rho, "u": u}, solver)


# ======================================================================
# Acoustic cases (uniform initial state)
# ======================================================================

def acoustic_uniform(cfg: DictConfig, solver: BaseSolver) -> LBMState:
    shape = tuple(cfg.simulation.spatial_shape)
    rho0 = cfg.solver.get("rho_0", 1.0)
    rho = jnp.full((*shape, 1), rho0)
    u = jnp.zeros((*shape, 2))
    return _make_state({"rho": rho, "u": u}, solver)


# ======================================================================
# Default (uniform rest state)
# ======================================================================

def default(cfg: DictConfig, solver: BaseSolver) -> LBMState:
    shape = tuple(cfg.simulation.spatial_shape)
    D = solver.lattice.D
    rho = jnp.ones((*shape, 1))
    u = jnp.zeros((*shape, D))
    macro: MacroState = {"rho": rho, "u": u}
    if any(d.label == "G" for d in solver.distributions):
        macro["T"] = jnp.full((*shape, 1), 1.0 / 3.0)
    return _make_state(macro, solver)


# ======================================================================
# Registry
# ======================================================================

REGISTRY: dict[str, callable] = {
    "default": default,
    "double_shear_layer": double_shear_layer,
    "shallow_water_dam_break": shallow_water_dam_break,
    "taylor_green_3d": taylor_green_3d,
    "sod_subsonic": sod_subsonic,
    "sod_transonic": sod_transonic,
    "cylinder_compressible": cylinder_compressible,
    "cylinder_acoustic": cylinder_acoustic,
    "acoustic_uniform": acoustic_uniform,
}


def initialize(cfg: DictConfig, solver: BaseSolver) -> LBMState:
    """Dispatch to the correct initial condition function."""
    ic_name = cfg.get("initial_conditions", "default")
    fn = REGISTRY.get(ic_name)
    if fn is None:
        raise ValueError(
            f"Unknown initial_conditions: {ic_name!r}. "
            f"Available: {list(REGISTRY)}"
        )
    return fn(cfg, solver)
