"""Hydra structured configs and builder functions for LBM experiments.

Provides dataclass-based config schema that maps cleanly to YAML files and
a ``build_solver`` function that constructs the appropriate solver from config.

Future NeurDE / stability fields are present as stubs so the schema is stable
when those features land.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
from jax import Array
from omegaconf import MISSING, DictConfig, OmegaConf

from .boundaries import AbsorbingLayerSpec, BoundarySpec
from .collision import BGKCollision
from .definitions import LBMState, MacroState
from .equilibrium import (
    EntropicEquilibrium,
    EquilibriumModel,
    ExtendedEquilibrium,
    LevermoreEquilibrium,
    PolynomialEquilibrium,
)
from .lattice import D2Q9, D3Q19, Lattice


# ======================================================================
# Config dataclasses (Hydra structured configs)
# ======================================================================


@dataclass
class LatticeConfig:
    type: str = "D2Q9"
    shifts: list[float] | None = None


@dataclass
class EquilibriumConfig:
    F: str = "polynomial"
    G: str | None = None


@dataclass
class CollisionConfig:
    type: str = "BGK"


@dataclass
class SolverConfig:
    type: str = "isothermal"
    # Isothermal
    Re: float | None = None
    Ma: float | None = None
    char_length: float | None = None
    # Compressible
    Pr: float | None = None
    gamma: float | None = None
    Cv: float | None = None
    viscosity: float | None = None
    # ShallowWater
    gravity: float | None = None
    height_0: float | None = None
    Fr: float | None = None
    gravity_axis: int | None = None
    # Acoustic
    rho_0: float | None = None
    c_0: float | None = None
    tau: float | None = None
    # Multiphase
    interaction_strength: float | None = None


@dataclass
class SimulationConfig:
    spatial_shape: list[int] = field(default_factory=lambda: [100, 100])
    dt: float = 1.0
    t_max: float = 1000.0
    log_interval: int = 100
    save_interval: int = 0
    save_path: str | None = None
    plot_enabled: bool = False
    plot_interval: int = 10
    plot_fields: list[str] = field(default_factory=lambda: ["rho", "u"])


@dataclass
class BoundaryFaceConfig:
    low: str = "periodic"
    high: str = "periodic"


@dataclass
class AbsorbingFaceConfig:
    width: int = 30
    sigma_max: float = 0.3


@dataclass
class BoundaryConfig:
    x: BoundaryFaceConfig = field(default_factory=BoundaryFaceConfig)
    y: BoundaryFaceConfig = field(default_factory=BoundaryFaceConfig)
    z: BoundaryFaceConfig | None = None
    absorbing: dict[str, AbsorbingFaceConfig] | None = None
    velocity: dict[str, Any] | None = None
    pressure: dict[str, Any] | None = None


@dataclass
class ObstacleConfig:
    type: str = "cylinder"
    center: list[int] | None = None
    radius: int | None = None
    # RigidWall
    axis: int | None = None
    position: int | None = None
    start: int | None = None
    end: int | None = None


@dataclass
class SourceConfig:
    kind: str = "monopole"
    position: list[int] | None = None
    amplitude: float = 0.01
    lateral_amplitude: float | None = None
    frequency: float | None = None
    direction: list[float] | None = None
    velocity: list[float] | None = None
    rotation_speed: float | None = None
    chirp_rate: float | None = None
    start_time: float | None = None


@dataclass
class NeurDEConfig:
    """Stub for future NeurDE integration."""
    enabled: bool = False
    F_type: str = "polynomial"
    G_type: str | None = None


@dataclass
class StabilityConfig:
    """Stub for future stability model integration."""
    type: str = "none"


@dataclass
class ExperimentConfig:
    name: str = "default"
    lattice: LatticeConfig = field(default_factory=LatticeConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    equilibrium: EquilibriumConfig = field(default_factory=EquilibriumConfig)
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    obstacles: list[ObstacleConfig] = field(default_factory=list)
    sources: list[SourceConfig] = field(default_factory=list)
    neurde: NeurDEConfig = field(default_factory=NeurDEConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    initial_conditions: str = "default"
    output_variables: list[str] = field(default_factory=lambda: ["rho", "u"])


# ======================================================================
# Builders
# ======================================================================


def build_lattice(cfg: LatticeConfig) -> Lattice:
    """Construct a Lattice from config."""
    from .lattice import D3Q27

    registry: dict[str, type[Lattice]] = {
        "D2Q9": D2Q9,
        "D3Q19": D3Q19,
        "D3Q27": D3Q27,
    }

    lattice_cls = registry.get(cfg.type)
    if lattice_cls is None:
        raise ValueError(f"Unknown lattice type: {cfg.type!r}. Available: {list(registry)}")

    shifts = getattr(cfg, "shifts", None)
    if shifts is not None:
        import jax.numpy as jnp
        shifts = jnp.array(list(shifts), dtype=jnp.float32)
    return lattice_cls(shifts=shifts)


def build_equilibrium(name: str) -> EquilibriumModel:
    """Construct an equilibrium model by name."""
    registry: dict[str, EquilibriumModel] = {
        "polynomial": PolynomialEquilibrium(),
        "entropic": EntropicEquilibrium(),
        "extended": ExtendedEquilibrium(),
        "levermore": LevermoreEquilibrium(),
    }
    model = registry.get(name)
    if model is None:
        raise ValueError(f"Unknown equilibrium: {name!r}. Available: {list(registry)}")
    return model


def _val(cfg, key: str, default):
    """Extract a config value with a fallback default."""
    v = getattr(cfg, key, None)
    return v if v is not None else default


def build_solver(cfg: ExperimentConfig):
    """Construct the appropriate solver from a full ExperimentConfig."""
    from .solvers import (
        AcousticSolver,
        CompressibleSolver,
        IsothermalSolver,
        MultiphaseSolver,
        ShallowWaterSolver,
    )

    lattice = build_lattice(cfg.lattice)
    eq_model = build_equilibrium(cfg.equilibrium.F)
    boundary_spec = build_boundary_spec(cfg.boundary)
    sim = cfg.simulation
    solver_cfg = cfg.solver

    common = dict(
        lattice=lattice,
        dt=sim.dt,
        t_max=sim.t_max,
        collision=BGKCollision(),
        boundary_spec=boundary_spec,
        plot_enabled=sim.plot_enabled,
        plot_interval=sim.plot_interval,
        plot_fields=tuple(sim.plot_fields),
    )

    solver_type = solver_cfg.type

    if solver_type == "isothermal":
        return IsothermalSolver(
            **common,
            Re=_val(solver_cfg, "Re", 100.0),
            Ma=_val(solver_cfg, "Ma", 0.1),
            char_length=_val(solver_cfg, "char_length", float(sim.spatial_shape[0])),
            equilibrium_model=eq_model if cfg.equilibrium.F != "polynomial" else None,
        )

    elif solver_type == "compressible":
        return CompressibleSolver(
            **common,
            Pr=_val(solver_cfg, "Pr", 0.71),
            gamma=_val(solver_cfg, "gamma", 1.4),
            viscosity=_val(solver_cfg, "viscosity", 0.01),
        )

    elif solver_type == "shallow_water":
        return ShallowWaterSolver(
            **common,
            gravity=_val(solver_cfg, "gravity", 0.01),
            height_0=_val(solver_cfg, "height_0", 1.0),
            Fr=solver_cfg.Fr,
            Re=_val(solver_cfg, "Re", 100.0),
            char_length=_val(solver_cfg, "char_length", float(sim.spatial_shape[0])),
            gravity_axis=_val(solver_cfg, "gravity_axis", 1),
        )

    elif solver_type == "acoustic":
        from .solvers.acoustic import AcousticSource
        raw_sources = cfg.get("sources", []) if hasattr(cfg, "get") else getattr(cfg, "sources", [])
        source_cfg_list = _convert_source_cfgs(raw_sources)
        sources = _build_acoustic_sources(source_cfg_list, tuple(sim.spatial_shape), lattice)
        return AcousticSolver(
            **common,
            rho_0=_val(solver_cfg, "rho_0", 1.0),
            c_0=solver_cfg.c_0,
            tau=_val(solver_cfg, "tau", 0.501),
            sources=tuple(sources),
        )

    elif solver_type == "multiphase":
        return MultiphaseSolver(
            **common,
            interaction_strength=_val(solver_cfg, "interaction_strength", -1.0),
            tau=_val(solver_cfg, "tau", 0.8),
        )

    else:
        raise ValueError(f"Unknown solver type: {solver_type!r}")


def build_boundary_spec(cfg) -> BoundarySpec | None:
    """Construct BoundarySpec from config."""
    x_pair = (cfg.x.low, cfg.x.high)
    y_pair = (cfg.y.low, cfg.y.high)
    z_pair = None
    z_cfg = cfg.get("z", None) if hasattr(cfg, "get") else getattr(cfg, "z", None)
    if z_cfg is not None:
        z_pair = (z_cfg.low, z_cfg.high)

    all_periodic = (
        x_pair == ("periodic", "periodic")
        and y_pair == ("periodic", "periodic")
        and (z_pair is None or z_pair == ("periodic", "periodic"))
    )
    if all_periodic:
        return None

    return BoundarySpec(x=x_pair, y=y_pair, z=z_pair)


def build_absorbing_spec(
    cfg, spatial_shape: tuple[int, ...], rho_0: float = 1.0,
) -> AbsorbingLayerSpec | None:
    """Construct AbsorbingLayerSpec if absorbing faces are configured."""
    absorbing = cfg.get("absorbing", None) if hasattr(cfg, "get") else getattr(cfg, "absorbing", None)
    if absorbing is None:
        return None
    cfg_absorbing = absorbing

    face_key_map = {
        "x_low": (0, 0), "x_high": (0, 1),
        "y_low": (1, 0), "y_high": (1, 1),
        "z_low": (2, 0), "z_high": (2, 1),
    }

    faces: dict[tuple[int, int], dict] = {}
    for key, face_cfg in cfg_absorbing.items():
        if key not in face_key_map:
            raise ValueError(f"Unknown absorbing face key: {key!r}")
        if isinstance(face_cfg, DictConfig):
            face_cfg = OmegaConf.to_container(face_cfg, resolve=True)
        if isinstance(face_cfg, AbsorbingFaceConfig):
            faces[face_key_map[key]] = {"width": face_cfg.width, "sigma_max": face_cfg.sigma_max}
        else:
            faces[face_key_map[key]] = dict(face_cfg)

    if not faces:
        return None

    return AbsorbingLayerSpec.build(spatial_shape, rho_0=rho_0, faces=faces)


def build_boundary_data(
    cfg, spatial_shape: tuple[int, ...], lattice: Lattice,
) -> tuple[
    dict[tuple[int, int], Array],
    dict[tuple[int, int], Array],
]:
    """Build boundary_velocity and boundary_pressure dicts from config."""
    D = lattice.D
    boundary_velocity: dict[tuple[int, int], Array] = {}
    boundary_pressure: dict[tuple[int, int], Array] = {}

    face_key_map = {
        "x_low": (0, 0), "x_high": (0, 1),
        "y_low": (1, 0), "y_high": (1, 1),
        "z_low": (2, 0), "z_high": (2, 1),
    }

    velocity_cfg = cfg.get("velocity", None) if hasattr(cfg, "get") else getattr(cfg, "velocity", None)
    if velocity_cfg is not None:
        for key, vel_data in velocity_cfg.items():
            if key not in face_key_map:
                continue
            ax, sd = face_key_map[key]
            face_shape = list(spatial_shape)
            face_shape.pop(ax)
            if isinstance(vel_data, (list, tuple)):
                u_val = jnp.array(vel_data, dtype=jnp.float32)
                u_face = jnp.broadcast_to(u_val, (*face_shape, D))
            else:
                u_face = jnp.zeros((*face_shape, D))
            boundary_velocity[(ax, sd)] = u_face

    pressure_cfg = cfg.get("pressure", None) if hasattr(cfg, "get") else getattr(cfg, "pressure", None)
    if pressure_cfg is not None:
        for key, rho_val in pressure_cfg.items():
            if key not in face_key_map:
                continue
            ax, sd = face_key_map[key]
            face_shape = list(spatial_shape)
            face_shape.pop(ax)
            rho_face = jnp.full((*face_shape, 1), float(rho_val), dtype=jnp.float32)
            boundary_pressure[(ax, sd)] = rho_face

    return boundary_velocity, boundary_pressure


def _convert_source_cfgs(raw_sources) -> list[SourceConfig]:
    """Convert OmegaConf source list to SourceConfig objects."""
    result = []
    if not raw_sources:
        return result
    for s in raw_sources:
        if isinstance(s, SourceConfig):
            result.append(s)
            continue
        result.append(SourceConfig(
            kind=s.get("kind", "monopole") if hasattr(s, "get") else s.kind,
            position=list(s.get("position", [])) if s.get("position") else None,
            amplitude=float(s.get("amplitude", 0.01)),
            lateral_amplitude=float(s.get("lateral_amplitude")) if s.get("lateral_amplitude") is not None else None,
            frequency=float(s.get("frequency")) if s.get("frequency") is not None else None,
            direction=list(s.get("direction")) if s.get("direction") is not None else None,
            velocity=list(s.get("velocity")) if s.get("velocity") is not None else None,
            rotation_speed=float(s.get("rotation_speed")) if s.get("rotation_speed") is not None else None,
            chirp_rate=float(s.get("chirp_rate")) if s.get("chirp_rate") is not None else None,
            start_time=float(s.get("start_time")) if s.get("start_time") is not None else None,
        ))
    return result


def _build_acoustic_sources(
    source_cfgs: list[SourceConfig],
    spatial_shape: tuple[int, ...],
    lattice: Lattice,
) -> list:
    """Build AcousticSource objects from config."""
    import math
    from .solvers.acoustic import AcousticSource

    D = lattice.D
    sources = []

    for src_cfg in source_cfgs:
        pos = tuple(src_cfg.position) if src_cfg.position else (0,) * D
        freq = src_cfg.frequency

        # Build time function
        if src_cfg.chirp_rate is not None:
            chirp_rate = src_cfg.chirp_rate
            if freq is not None:
                def make_chirp_fn(f0, cr):
                    def time_fn(t):
                        return jnp.sin(2.0 * math.pi * (f0 + cr * t) * t)
                    return time_fn
                time_fn = make_chirp_fn(freq, chirp_rate)
            else:
                time_fn = lambda t: 1.0
        elif freq is not None:
            def make_sin_fn(f):
                def time_fn(t):
                    return jnp.sin(2.0 * math.pi * f * t)
                return time_fn
            time_fn = make_sin_fn(freq)
        else:
            time_fn = lambda t: 1.0

        # Build direction function for rotating dipoles
        direction_fn = None
        if src_cfg.rotation_speed is not None and src_cfg.direction is not None:
            base_dir = src_cfg.direction[:D]
            rot_speed = src_cfg.rotation_speed
            def make_rot_fn(bd, rs, d):
                def dir_fn(t):
                    angle = rs * t
                    cos_a = jnp.cos(angle)
                    sin_a = jnp.sin(angle)
                    if d == 2:
                        return jnp.array([
                            bd[0] * cos_a - bd[1] * sin_a,
                            bd[0] * sin_a + bd[1] * cos_a,
                        ])
                    return jnp.array(bd)
                return dir_fn
            direction_fn = make_rot_fn(base_dir, rot_speed, D)

        amp_val = src_cfg.amplitude

        if src_cfg.kind == "monopole":
            amp_field = jnp.zeros(spatial_shape)
            amp_field = amp_field.at[pos].set(amp_val)
            sources.append(AcousticSource(
                kind="monopole", amplitude=amp_field, time_fn=time_fn,
                base_amplitude=amp_val, spatial_shape=spatial_shape,
            ))

        elif src_cfg.kind == "dipole":
            direction = src_cfg.direction or [1.0] + [0.0] * (D - 1)
            dir_arr = jnp.array(direction[:D], dtype=jnp.float32)
            amp_field = jnp.zeros((*spatial_shape, D))
            amp_field = amp_field.at[pos].set(amp_val * dir_arr)
            sources.append(AcousticSource(
                kind="dipole", amplitude=amp_field, time_fn=time_fn,
                direction_fn=direction_fn,
                base_amplitude=amp_val, spatial_shape=spatial_shape,
            ))

        elif src_cfg.kind == "quadrupole":
            direction = src_cfg.direction or [1.0] + [0.0] * (D - 1)
            dir_arr = jnp.array(direction[:D], dtype=jnp.float32)
            amp_field = jnp.zeros((*spatial_shape, D))
            amp_field = amp_field.at[pos].set(amp_val * dir_arr)

            lat_amp = None
            if src_cfg.lateral_amplitude is not None:
                lat_dir = [0.0] * D
                lat_dir[1 if direction[0] != 0 else 0] = 1.0
                lat_arr = jnp.array(lat_dir, dtype=jnp.float32)
                lat_field = jnp.zeros((*spatial_shape, D))
                lat_field = lat_field.at[pos].set(src_cfg.lateral_amplitude * lat_arr)
                lat_amp = lat_field

            sources.append(AcousticSource(
                kind="quadrupole",
                amplitude=amp_field,
                lateral_amplitude=lat_amp,
                time_fn=time_fn,
                base_amplitude=amp_val, spatial_shape=spatial_shape,
            ))

    return sources
