"""Abstract base solver for all LBM solver variants.

Provides the generic collide-stream-BC-lift step loop, streaming, run modes
(Python loop with logging/plotting and pure-JAX ``fori_loop``), and plotting
infrastructure. Concrete solvers override :pyattr:`distributions`,
:pymeth:`system_variables`, and optionally :pymeth:`_collide_and_stream` to
inject solver-specific physics (source terms, variable tau, thermal coupling).
"""
from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.animation as manim
import matplotlib.image as mimg
import matplotlib.pyplot as plt
import numpy as np
from jax import Array

from ..boundaries import BoundarySpec, apply_boundaries
from ..collision import BGKCollision, CollisionScheme
from ..definitions import DTYPE, DTYPE_LOW, PLOT_DPI, TAU_MIN, Level, LBMState, MacroState
from ..distributions import Distribution
from ..fields import Field
from ..lattice import Lattice
from ..obstacles import Obstacle, apply_obstacles
from ..plotting import plot_fields_grid
from ..utils.profiling import GPUProfiler, get_profiler

if TYPE_CHECKING:
    from ..data import SimulationWriter

matplotlib.use("Agg")


class BaseSolver(eqx.Module):
    """Abstract LBM solver.

    Subclasses must define:
        - ``distributions`` (property) — which population(s) this solver uses.
        - ``system_variables`` — physical-to-LBM parameter mapping.

    And may override:
        - ``_collide_and_stream`` — to inject source terms or variable-tau collision.
        - ``_lift_all`` — to customize macro extraction.
    """

    lattice: Lattice
    dt: float
    t_max: float | None
    collision: CollisionScheme = eqx.field(default_factory=BGKCollision)
    boundary_spec: BoundarySpec | None = None
    obstacles: tuple[Obstacle, ...] = ()

    use_mixed_precision: bool = False

    plot_enabled: bool = False
    plot_interval: int = 1
    plot_output_dir: str = "."
    plot_frame_dir: str = "/tmp/lbm_frames"
    plot_fields: tuple[str, ...] = ("rho", "u")

    logger: logging.Logger = eqx.field(
        default_factory=lambda: logging.getLogger("lbm"), static=True
    )
    gpu_profiler: GPUProfiler = eqx.field(
        default_factory=lambda: get_profiler(), static=True
    )

    steps: int | None = eqx.field(init=False)

    def __post_init__(self) -> None:
        if self.t_max is not None and self.dt > 0:
            object.__setattr__(self, "steps", int(self.t_max / self.dt))
        else:
            object.__setattr__(self, "steps", None)
        for dist in self.distributions:
            tau = dist.relaxation_time()
            if tau < TAU_MIN:
                raise ValueError(
                    f"BGK stability requires τ ≥ 0.5. "
                    f"{dist.__class__.__name__} has τ={tau}."
                )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def distributions(self) -> tuple[Distribution, ...]:
        """Population tuple this solver operates on (e.g. (F,) or (F, G))."""
        ...

    @abstractmethod
    def system_variables(self) -> dict[str, float]:
        """Physical-to-LBM parameter mapping (Re, Ma, tau, etc.)."""
        ...

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def _collide_and_stream(
        self,
        state: LBMState,
        macro: MacroState,
        boundary_velocity: dict[tuple[int, int], Array] | None = None,
        boundary_pressure: dict[tuple[int, int], Array] | None = None,
    ) -> dict[str, Array]:
        """Collide, stream, and apply BCs for every distribution.

        Override in subclasses to inject source terms, variable tau, or
        custom collision logic.  Returns new distribution dict.
        """
        new_dists: dict[str, Array] = {}
        for dist in self.distributions:
            f = state.dists[dist.label]
            f_star = self.collision.collide(f, macro, dist, self.lattice)
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
                f_streamed = apply_obstacles(f_streamed, self.lattice, self.obstacles)
            new_dists[dist.label] = f_streamed
        return new_dists

    def _lift_all(
        self, new_dists: dict[str, Array], macro: MacroState,
    ) -> MacroState:
        """Extract macro fields from post-stream distributions.

        Runs lifts in distribution order so later distributions see earlier
        updates (e.g. thermal lift uses rho from momentum lift).
        """
        for dist in self.distributions:
            lifted = dist.lift(new_dists[dist.label], self.lattice, macro=macro)
            macro = {**macro, **lifted}
        return macro

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def _step_inner(
        self,
        state: LBMState,
        boundary_velocity: dict[tuple[int, int], Array] | None = None,
        boundary_pressure: dict[tuple[int, int], Array] | None = None,
    ) -> LBMState:
        state_f32 = jax.tree.map(lambda v: jnp.asarray(v, dtype=DTYPE), state)
        macro: MacroState = state_f32.macro

        new_dists = self._collide_and_stream(
            state_f32, macro, boundary_velocity, boundary_pressure,
        )
        macro = self._lift_all(new_dists, macro)

        new_state = LBMState(
            rho=macro["rho"],
            u=macro["u"],
            T=macro.get("T"),
            dists=new_dists,
        )
        if self.use_mixed_precision:
            new_state = jax.tree.map(
                lambda v: jnp.asarray(v, dtype=DTYPE_LOW), new_state,
            )
        return new_state

    # ------------------------------------------------------------------
    # Public step / run
    # ------------------------------------------------------------------

    @eqx.filter_jit
    def step(
        self,
        state: LBMState,
        *,
        boundary_velocity: dict[tuple[int, int], Array] | None = None,
        boundary_pressure: dict[tuple[int, int], Array] | None = None,
    ) -> LBMState:
        """One JIT-compiled time step."""
        return self._step_inner(state, boundary_velocity, boundary_pressure)

    @eqx.filter_jit
    def run_jit(
        self,
        state: LBMState,
        *,
        boundary_velocity: dict[tuple[int, int], Array] | None = None,
        boundary_pressure: dict[tuple[int, int], Array] | None = None,
    ) -> LBMState:
        """Run ``self.steps`` entirely on device via ``jax.lax.fori_loop``."""
        if self.steps is None:
            raise ValueError("run_jit() requires t_max and dt")

        def body(_, s):
            return self._step_inner(s, boundary_velocity, boundary_pressure)

        return jax.lax.fori_loop(0, self.steps, body, state)

    def run(
        self,
        state: LBMState,
        run_id: str,
        *,
        log_interval: int = 1,
        save_interval: int = 0,
        data_writer: SimulationWriter | None = None,
        boundary_velocity: dict[tuple[int, int], Array] | None = None,
        boundary_pressure: dict[tuple[int, int], Array] | None = None,
    ) -> LBMState:
        """Python loop with logging, plotting, and optional HDF5 saving."""
        if self.steps is None:
            raise ValueError("run() requires t_max and dt")
        for step_idx in range(self.steps):
            state = self.step(
                state,
                boundary_velocity=boundary_velocity,
                boundary_pressure=boundary_pressure,
            )
            if log_interval > 0 and step_idx % log_interval == 0:
                self.log(step_idx, state, run_id=run_id)
            if self.plot_enabled and (
                step_idx % self.plot_interval == 0 or step_idx == self.steps - 1
            ):
                self.plot(state, step_idx, run_id)
            if data_writer is not None and save_interval > 0 and step_idx % save_interval == 0:
                data_writer.write(state, step=step_idx, time=step_idx * self.dt)
        return state

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def _interpolate_shift(self, f: Array) -> Array:
        """Pre-streaming interpolation for Galilean lattice shifts.

        For each spatial dimension with nonzero shift s_d, blends
        distributions with the upstream neighbor:
            f <- f * (1 - |s|) + roll(f, sign(s), dim=d) * |s|
        Positive shifts roll +1, negative shifts roll -1.
        """
        if not self.lattice.is_shifted:
            return f
        for dim in range(self.lattice.D):
            s = self.lattice.shifts[dim]
            abs_s = jnp.abs(s)
            f_pos = jnp.roll(f, 1, axis=dim)
            f_neg = jnp.roll(f, -1, axis=dim)
            f_neighbor = jnp.where(s >= 0, f_pos, f_neg)
            f = f * (1.0 - abs_s) + f_neighbor * abs_s
        return f

    @eqx.filter_jit
    def _stream(self, f: Array) -> Array:
        D = self.lattice.D
        velocities = self.lattice.velocities

        def stream_direction(f_slice: Array, e_i: Array) -> Array:
            rolled = f_slice
            for dim in range(D):
                rolled = jnp.roll(rolled, e_i[dim], axis=dim)
            return rolled

        f_streamed = jax.vmap(stream_direction, in_axes=(-1, 0))(f, velocities)
        perm = tuple(range(1, D + 1)) + (0,)
        return jnp.transpose(f_streamed, perm)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(
        self,
        step: int,
        state: LBMState,
        *,
        run_id: str | None = None,
        gpu_device: int | None = None,
    ) -> None:
        t = step * self.dt
        rho_sum = float(jnp.sum(state.rho))
        u_mag = jnp.linalg.norm(state.u, axis=-1)
        u_max = float(jnp.max(u_mag))
        shape = state.rho.shape
        grid_str = "x".join(str(s) for s in shape)

        progress_parts: list[str] = [f"step={step}", f"t={t:.4f}"]
        if self.steps is not None:
            progress_parts.append(f"total_steps={self.steps}")
        if run_id is not None:
            progress_parts.append(f"run_id={run_id}")
        groups: list[str] = [" ".join(progress_parts), f"grid={grid_str}"]
        groups.append(f"rho_sum={rho_sum:.6f} u_max={u_max:.6f}")
        if state.T is not None:
            groups.append(
                f"T_min={float(jnp.min(state.T)):.4f} T_max={float(jnp.max(state.T)):.4f}"
            )
        self.logger.info(" | ".join(groups))
        if self.gpu_profiler is not None:
            self.logger.info(self.gpu_profiler.log(device=gpu_device))

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def _state_to_plot_fields(self, state: LBMState) -> list[Field]:
        fields: list[Field] = []
        names_components: dict[str, tuple[str, list[str]]] = {
            "rho": ("density", ["density"]),
            "u": ("velocity", ["u_x", "u_y"]),
            "T": ("temperature", ["T"]),
        }
        for key in self.plot_fields:
            if key not in state:
                continue
            arr = state[key]
            name, comp = names_components.get(key, (key, [key]))
            if key == "u" and arr.ndim >= 2:
                comp = [f"u_{i}" for i in range(arr.shape[-1])]
            field = Field.from_array(
                name, np.asarray(arr), Level.MACROSCOPIC, component_names=comp,
            )
            fields.append(field)
        return fields

    def plot(self, state: LBMState, step: int, run_id: str) -> Path | None:
        if not self.plot_enabled:
            return None
        plot_fields_list = self._state_to_plot_fields(state)
        if not plot_fields_list:
            return None
        frame_dir = Path(self.plot_frame_dir) / run_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        n = len(plot_fields_list)
        fig, _ = plot_fields_grid(plot_fields_list, ncols=n, figsize=(4 * n, 4))
        fig.suptitle(f"Step {step}")
        out_path = frame_dir / f"frame_{step:06d}.png"
        fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def write_video(
        self, run_id: str, output_name: str = "lbm_video.mp4", fps: int = 10,
    ) -> Path:
        frame_dir = Path(self.plot_frame_dir) / run_id
        out_path = Path(self.plot_output_dir) / output_name
        Path(self.plot_output_dir).mkdir(parents=True, exist_ok=True)
        frames_sorted = sorted(frame_dir.glob("frame_*.png"))
        if not frames_sorted:
            raise FileNotFoundError(f"No frames found in {frame_dir}")
        try:
            writer = manim.FFMpegWriter(fps=fps)
            fig, ax = plt.subplots()
            ax.axis("off")
            with writer.saving(fig, str(out_path), dpi=PLOT_DPI):
                for p in frames_sorted:
                    ax.imshow(mimg.imread(p), origin="upper")
                    writer.grab_frame()
            plt.close(fig)
        except (FileNotFoundError, OSError) as e:
            raise RuntimeError("Video encoding requires ffmpeg on PATH.") from e
        return out_path
