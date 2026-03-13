"""Hydra entry point for LBM experiments.

Usage:
    python experiments/run.py experiment=double_shear_layer
    python experiments/run.py experiment=sod_subsonic
    python experiments/run.py experiment=cylinder_acoustic
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lbm.jax_config import configure_jax

configure_jax()

import hydra
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from lbm.config import (
    build_absorbing_spec,
    build_boundary_data,
    build_solver,
)
from lbm.obstacles import Cylinder, Obstacle, RigidWall

from initial_conditions import initialize

log = logging.getLogger(__name__)


def _get(cfg, key, default=None):
    """Retrieve a value from a dict or OmegaConf node."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _build_obstacles(
    obs_cfgs: list, spatial_shape: tuple[int, ...],
) -> tuple[Obstacle, ...]:
    obstacles: list[Obstacle] = []
    if not obs_cfgs:
        return ()
    for obs_cfg in obs_cfgs:
        obs_type = _get(obs_cfg, "type")
        if obs_type == "cylinder":
            obstacles.append(Cylinder(
                center=tuple(_get(obs_cfg, "center")),
                radius=int(_get(obs_cfg, "radius")),
                spatial_shape=spatial_shape,
            ))
        elif obs_type == "rigid_wall":
            obstacles.append(RigidWall(
                axis=int(_get(obs_cfg, "axis")),
                position=int(_get(obs_cfg, "position")),
                start=int(_get(obs_cfg, "start")),
                end=int(_get(obs_cfg, "end")),
                spatial_shape=spatial_shape,
            ))
        else:
            raise ValueError(f"Unknown obstacle type: {obs_type!r}")
    return tuple(obstacles)


def _build_source_cfgs(source_list: list) -> list:
    """Convert OmegaConf source dicts into SourceConfig-like objects."""
    from lbm.config import SourceConfig
    sources = []
    for s in source_list:
        sc = SourceConfig(
            kind=s.kind,
            position=list(s.position) if s.get("position") else None,
            amplitude=float(s.get("amplitude", 0.01)),
            lateral_amplitude=float(s.lateral_amplitude) if s.get("lateral_amplitude") is not None else None,
            frequency=float(s.frequency) if s.get("frequency") is not None else None,
            direction=list(s.direction) if s.get("direction") is not None else None,
            velocity=list(s.velocity) if s.get("velocity") is not None else None,
            rotation_speed=float(s.rotation_speed) if s.get("rotation_speed") is not None else None,
            chirp_rate=float(s.chirp_rate) if s.get("chirp_rate") is not None else None,
            start_time=float(s.start_time) if s.get("start_time") is not None else None,
        )
        sources.append(sc)
    return sources


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info("Experiment: %s", cfg.name)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    spatial_shape = tuple(cfg.simulation.spatial_shape)
    exp = cfg.experiment

    # Build obstacles
    obs_list = OmegaConf.to_container(exp.obstacles, resolve=True) if exp.get("obstacles") else []
    obstacles = _build_obstacles(obs_list, spatial_shape)

    # Build solver
    solver = build_solver(cfg)
    if obstacles:
        import equinox as eqx
        solver = eqx.tree_at(lambda s: s.obstacles, solver, obstacles)

    # Build absorbing layer
    rho_0 = cfg.solver.get("rho_0", 1.0) or 1.0
    absorbing_spec = build_absorbing_spec(cfg.boundary, spatial_shape, rho_0=rho_0)

    # Build boundary velocity / pressure data
    boundary_velocity, boundary_pressure = build_boundary_data(
        cfg.boundary, spatial_shape, solver.lattice,
    )

    # Initialize state
    state = initialize(cfg, solver)
    log.info(
        "Initial state: rho_sum=%.4f, u_max=%.6f, shape=%s",
        float(jnp.sum(state.rho)),
        float(jnp.max(jnp.linalg.norm(state.u, axis=-1))),
        state.rho.shape,
    )

    # Data writer
    data_writer = None
    if cfg.simulation.save_interval > 0 and cfg.simulation.save_path:
        from lbm.data import SimulationWriter
        save_dir = Path(cfg.simulation.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        data_writer = SimulationWriter(
            save_dir / "simulation.h5",
            lattice=solver.lattice,
            dt=cfg.simulation.dt,
            spatial_shape=spatial_shape,
            field_keys=list(cfg.output_variables),
            distribution_keys=["F"] + (["G"] if "G" in state.dists else []),
        )

    run_id = f"{cfg.name}_{np.random.randint(0, 10**9):09d}"

    try:
        state = solver.run(
            state,
            run_id,
            log_interval=cfg.simulation.log_interval,
            save_interval=cfg.simulation.save_interval,
            data_writer=data_writer,
            boundary_velocity=boundary_velocity or None,
            boundary_pressure=boundary_pressure or None,
        )
    finally:
        if data_writer is not None:
            data_writer.close()

    log.info(
        "Final state: rho_sum=%.4f, u_max=%.6f",
        float(jnp.sum(state.rho)),
        float(jnp.max(jnp.linalg.norm(state.u, axis=-1))),
    )

    if solver.plot_enabled:
        try:
            video_path = solver.write_video(run_id, output_name=f"{cfg.name}.mp4", fps=10)
            log.info("Video saved: %s", video_path)
        except (ImportError, RuntimeError, FileNotFoundError) as e:
            log.warning("Video skipped: %s", e)


if __name__ == "__main__":
    main()
