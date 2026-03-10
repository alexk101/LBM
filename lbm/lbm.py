"""Generic Lattice Boltzmann Method (LBM) solver using Equinox.

Supports single- and multi-distribution (coupled) systems. For thermal LBM,
use distributions=[ParticleDistribution(...), ThermalDistribution(...)] so that
momentum (f) and energy (g) share the same macroscopic state (ρ, u, T). See
Wikipedia: Lattice Boltzmann methods § Mathematical equations for simulations.
"""

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manim
import matplotlib.image as mimg
import equinox as eqx
import jax
import jax.numpy as jnp
from .boundaries import BoundarySpec, apply_boundaries
from .collision import BGKCollision, CollisionScheme
from .definitions import DTYPE, DTYPE_LOW, Level, LBMState, PLOT_DPI, TAU_MIN
from .distributions import (
    Distribution,
    MacroState,
)
from .fields import Field
from jax import Array
from .lattice import Lattice
from .plotting import plot_fields_grid
import logging

from .utils.profiling import GPUProfiler, get_profiler


class LBMSolver(eqx.Module):
    """LBM solver for one or more coupled distributions.

    State dict contains:
        - One key per distribution (e.g. "F", "G") -> array shape (..., N)
        - Macro keys "rho", "u", and optionally "T" (or others) -> updated each step

    Step order (per standard LBM): use current macro for all equilibria;
    collide and stream each distribution; then merge all lifts into new macro.

    Plotting (to file, no interactive display):
        - Set plot_enabled=True and plot_interval to save frames to /tmp.
        - Call solver.plot(state, step, run_id) at each interval.
        - Call solver.write_video(run_id) after the run to stitch frames into a video
          in plot_output_dir.

    Logging and GPU profiling:
        - logger and gpu_profiler are static fields: they are not traced by JAX, so
          Equinox and jit/vmap work unchanged. Use them in your Python run loop
          (e.g. after each solver.step()), not inside step(). Pass gpu_profiler=get_profiler()
          from lbm.utils.profiling to enable GPU metrics (optional; default None).
          Use run() to execute the simulation with logging and plotting in one call:
            from lbm.utils.profiling import get_profiler
            solver = LBMSolver(..., gpu_profiler=get_profiler())
            state = solver.run(state, run_id, log_interval=10)
          Or loop manually and call solver.log(step, state, run_id=run_id) when needed.
    """

    lattice: Lattice
    dt: float
    t_max: float | None
    distributions: tuple[Distribution, ...]  # ordered: e.g. (momentum, thermal)
    collision: CollisionScheme = eqx.field(default_factory=BGKCollision)
    boundary_spec: BoundarySpec | None = None  # None = all periodic (default)

    # Mixed precision: when True, state (f, rho, u, T) is stored in bf16 between steps;
    # compute remains float32 for stability. Saves memory/bandwidth.
    use_mixed_precision: bool = False

    # Plotting: managed by solver; frames to /tmp, video to plot_output_dir
    plot_enabled: bool = False
    plot_interval: int = 1
    plot_output_dir: str = "."
    plot_frame_dir: str = "/tmp/lbm_frames"
    plot_fields: tuple[str, ...] = ("rho", "u")  # macro keys to plot (add "T" for thermal)

    # Logging and GPU profiling: static (not traced). Use in the run loop, not inside step().
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
        for dist in self.distributions:
            tau = dist.relaxation_time()
            if tau < TAU_MIN:
                raise ValueError(
                    f"BGK stability requires τ ≥ 0.5 (ν = c_s²(τ - 0.5)). "
                    f"{dist.__class__.__name__} has τ={tau}; use τ ≥ 0.5 to avoid NaNs."
                )

    def _state_to_plot_fields(self, state: LBMState) -> list[Field]:
        """Build Field objects from state for keys in plot_fields that exist."""
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
                name, np.asarray(arr), Level.MACROSCOPIC, component_names=comp
            )
            fields.append(field)
        return fields

    def plot(self, state: LBMState, step: int, run_id: str) -> Path | None:
        """Generate a single frame from current state and save to file.

        Uses Field objects (built from state) and plotting utilities. Saves to
        plot_frame_dir / run_id / frame_{step:06d}.png. Call at each step when
        plot_enabled and step % plot_interval == 0.

        Returns:
            Path to saved frame, or None if plot_enabled is False or no fields to plot.
        """
        if not self.plot_enabled:
            return None
        plot_fields_list = self._state_to_plot_fields(state)
        if not plot_fields_list:
            return None
        frame_dir = Path(self.plot_frame_dir) / run_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        n = len(plot_fields_list)
        fig, _ = plot_fields_grid(
            plot_fields_list,
            ncols=n,
            figsize=(4 * n, 4),
        )
        fig.suptitle(f"Step {step}")
        out_path = frame_dir / f"frame_{step:06d}.png"
        fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def write_video(
        self,
        run_id: str,
        output_name: str = "lbm_video.mp4",
        fps: int = 10,
    ) -> Path:
        """Stitch saved frames in plot_frame_dir / run_id into a video using matplotlib.

        Requires ffmpeg on PATH for MP4. No extra Python deps beyond matplotlib.
        """

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
            raise RuntimeError(
                "Video encoding requires ffmpeg on your PATH. Install ffmpeg (e.g. "
                "apt install ffmpeg / pacman -S ffmpeg) and try again."
            ) from e
        return out_path

    @eqx.filter_jit
    def step(
        self,
        state: LBMState,
        *,
        boundary_velocity: dict[tuple[int, int], Array] | None = None,
        boundary_pressure: dict[tuple[int, int], Array] | None = None,
    ) -> LBMState:
        """One time step: collide and stream each distribution; update macro from lifts.

        For \"velocity\" or \"pressure\" BC faces, pass boundary_velocity or
        boundary_pressure keyed by (axis, side), e.g. boundary_velocity={(0, 0): u_inlet}.

        When use_mixed_precision is True, state is promoted to float32 for all compute,
        then only the returned state is cast to bf16 for storage (memory/bandwidth).
        """
        # Promote to float32 for compute (no-op if already float32)
        state_f32: LBMState = {
            k: jnp.asarray(v, dtype=DTYPE) for k, v in state.items()
        }
        macro: MacroState = {k: state_f32[k] for k in ["rho", "u"] if k in state_f32}
        if "T" in state_f32:
            macro["T"] = state_f32["T"]

        new_state: LBMState = {}
        for dist in self.distributions:
            f = state_f32[dist.label]
            f_star = self.collision.collide(f, macro, dist, self.lattice)
            f_streamed = self._stream(f_star)
            if self.boundary_spec is not None:
                f_streamed = apply_boundaries(
                    f_streamed,
                    self.lattice,
                    self.boundary_spec,
                    macro=macro,
                    boundary_velocity=boundary_velocity,
                    boundary_pressure=boundary_pressure,
                )
            new_state[dist.label] = f_streamed

        for dist in self.distributions:
            lifted = dist.lift(new_state[dist.label], self.lattice, macro=macro)
            new_state.update(lifted)
            macro = {**macro, **lifted}  # so next dist's lift sees updated fields (e.g. T = Σg/ρ)

        if self.use_mixed_precision:
            new_state = {k: jnp.asarray(v, dtype=DTYPE_LOW) for k, v in new_state.items()}
        return new_state

    def log(
        self,
        step: int,
        state: LBMState,
        *,
        run_id: str | None = None,
        gpu_device: int | None = None,
    ) -> None:
        """Log simulation state and optional GPU metrics.

        Simulation info is logged on one line, grouped as progress | grid | flow [| thermal].
        GPU info is logged on a separate line when gpu_profiler is set.

        Call from the run loop after solver.step(), e.g. when step % log_interval == 0.

        Args:
            step: Current step index.
            state: Current LBM state (dict with 'rho', 'u', optionally 'T', and distribution keys).
            run_id: Optional run identifier to include in the log line.
            gpu_device: GPU index to log, or None to log all GPUs (if gpu_profiler set).
        """
        t = step * self.dt
        rho = np.asarray(state["rho"])
        rho_sum = float(np.sum(rho))
        u = np.asarray(state["u"])
        u_mag = np.sqrt((u**2).sum(axis=-1))
        u_max = float(np.max(u_mag))
        shape = rho.shape
        grid_str = "x".join(str(s) for s in shape)

        # Simulation line: progress | grid | flow [| thermal]
        progress_parts: list[str] = [f"step={step}", f"t={t:.4f}"]
        if self.steps is not None:
            progress_parts.append(f"total_steps={self.steps}")
        if run_id is not None:
            progress_parts.append(f"run_id={run_id}")
        groups: list[str] = [" ".join(progress_parts), f"grid={grid_str}"]
        groups.append(f"rho_sum={rho_sum:.6f} u_max={u_max:.6f}")
        if "T" in state:
            T_arr = np.asarray(state["T"])
            groups.append(
                f"T_min={float(np.min(T_arr)):.4f} T_max={float(np.max(T_arr)):.4f}"
            )
        self.logger.info(" | ".join(groups))

        if self.gpu_profiler is not None:
            self.logger.info(self.gpu_profiler.log(device=gpu_device))

    def run(
        self,
        state: LBMState,
        run_id: str,
        *,
        log_interval: int = 1,
        boundary_velocity: dict[tuple[int, int], Array] | None = None,
        boundary_pressure: dict[tuple[int, int], Array] | None = None,
    ) -> LBMState:
        """Run the simulation from initial state, logging and plotting at configured intervals.

        Integrates step(), log(), and plot() in one loop. Use this for standard runs
        instead of manually looping over solver.step().

        Args:
            state: Initial LBM state.
            run_id: Run identifier for logging and for plot frame directory.
            log_interval: Log every this many steps (simulation + GPU lines).
            boundary_velocity: Optional dict passed to step() each time (e.g. inlet BC).
            boundary_pressure: Optional dict passed to step() each time (e.g. outlet BC).

        Returns:
            Final state after self.steps steps.

        Raises:
            ValueError: If self.steps is None (t_max or dt not set).
        """
        if self.steps is None:
            raise ValueError("run() requires t_max and dt to be set so steps is defined")
        for step in range(self.steps):
            state = self.step(
                state,
                boundary_velocity=boundary_velocity,
                boundary_pressure=boundary_pressure,
            )
            if log_interval > 0 and step % log_interval == 0:
                self.log(step, state, run_id=run_id)
            if self.plot_enabled and (
                step % self.plot_interval == 0 or step == self.steps - 1
            ):
                self.plot(state, step, run_id)
        return state

    @eqx.filter_jit
    def _stream(self, f: Array) -> Array:
        """Stream distribution along lattice velocities.

        This is a generic streaming operator that works for any dimension D
        by rolling the array in each direction according to the lattice velocity vectors.

        Args:
            f: Distribution function with shape (..., N) where ... are spatial dimensions

        Returns:
            Streamed distribution with same shape, shifted along characteristic directions
        """
        D = self.lattice.D
        velocities = self.lattice.velocities  # Shape (N, D)

        def stream_direction(f_slice: Array, e_i: Array) -> Array:
            """Stream a single direction's population."""
            rolled = f_slice
            for dim in range(D):
                # Roll by -e_i[dim] to propagate along velocity direction (JAX scalar, no int())
                rolled = jnp.roll(rolled, -e_i[dim], axis=dim)
            return rolled

        # Apply streaming to all directions using vmap
        # Input: f with shape (..., N), velocities with shape (N, D)
        # Map over last axis of f and axis 0 of velocities (direction index)
        f_streamed = jax.vmap(stream_direction, in_axes=(-1, 0))(f, velocities)

        # Permute axes to restore original spatial dimension ordering
        # Before: (spatial_dims..., N) -> After: (N, spatial_dims...)
        perm = tuple(range(1, D + 1)) + (0,)
        return jnp.transpose(f_streamed, perm)

    @eqx.filter_jit
    def initialize_fields(
        self, rho_init: Array, u_init: Array, T_init: Array | None = None
    ) -> dict[str, Field]:
        """Build Field metadata for state (spatial shape inferred from rho_init)."""
        if rho_init.ndim == 2:
            X, Y = rho_init.shape
            D = 2
        elif rho_init.ndim == 3 and u_init.ndim == 4:
            X, Y, Z = rho_init.shape
            D = 3
        else:
            raise ValueError(
                f"Unsupported field dimensions: "
                f"rho_init.shape={rho_init.shape}, u_init.shape={u_init.shape}"
            )
        shape_2d = (X, Y) if D == 2 else (X, Y, Z)
        fields_dict: dict[str, Field] = {}
        fields_dict["rho"] = Field(
            name="rho", level=Level.MACROSCOPIC, units="lattice_units",
            component_names=["density"], shape=shape_2d,
        )
        fields_dict["u"] = Field(
            name="velocity", level=Level.MACROSCOPIC, units="lattice_units",
            component_names=[f"u_{i}" for i in range(D)], shape=shape_2d,
        )
        N = self.lattice.N
        for dist in self.distributions:
            fields_dict[dist.label] = Field(
                name=dist.name, level=Level.MICROSCOPIC, units="lattice_units",
                component_names=[f"{dist.label}_{i}" for i in range(N)], shape=shape_2d,
            )
        if T_init is not None:
            fields_dict["T"] = Field(
                name="T", level=Level.MACROSCOPIC, units="lattice_units",
                component_names=["T"], shape=shape_2d,
            )
        return fields_dict


def compute_equilibrium_init(
    macro: MacroState, distribution: Distribution, lattice: Lattice
) -> Array:
    """Initial equilibrium for one distribution from macroscopic state."""
    return distribution.equilibrium(macro, lattice)