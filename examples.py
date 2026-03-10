from lbm.jax_config import configure_jax

# Configure JAX
configure_jax()

from lbm.lbm import LBMSolver
from lbm.lattice import D2Q9
from lbm.distributions import (
    ParticleDistribution,
    ThermalDistribution,
)
from lbm.boundaries import BoundarySpec
from lbm.definitions import MacroState, LBMState

import jax.numpy as jnp
import numpy as np
import logging
from pathlib import Path


def main():
    """Run single-distribution (isothermal); plotting to file when enabled."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    X, Y = 100, 100
    distance = X // 10
    lattice = D2Q9()
    dist_f = ParticleDistribution(tau=0.6)
    steps = 100
    run_id = f"run_{np.random.randint(0, 10**9):09d}"

    solver = LBMSolver(
        lattice=lattice,
        dt=0.1,
        t_max=float(steps) * 0.1,
        distributions=(dist_f,),
        plot_enabled=True,
        plot_interval=5,
        plot_output_dir=Path.cwd(),
        plot_frame_dir="/tmp/lbm_frames",
        plot_fields=("rho", "u"),
    )
    rho_init = jnp.ones((X, Y))
    rho_init = (
        rho_init.at[(X // 2) - (distance // 2), Y // 2]
        .set(2.0)
        .at[(X // 2) + (distance // 2), Y // 2]
        .set(2.0)
    )
    u_init = jnp.zeros((X, Y, 2))
    macro_init: MacroState = {
        "rho": rho_init[..., None],
        "u": u_init,
    }
    state: LBMState = {
        dist_f.label: dist_f.equilibrium(macro_init, lattice),
        "rho": macro_init["rho"],
        "u": macro_init["u"],
    }

    solver.logger.info(
        "Initial state (single-distribution): rho_sum=%.4f",
        float(np.sum(state["rho"])),
    )
    solver.logger.info(
        "Plotting: enabled, interval=%d, run_id=%s", solver.plot_interval, run_id
    )
    state = solver.run(state, run_id, log_interval=10)
    solver.logger.info("Final state: rho_sum=%.4f", float(np.sum(state["rho"])))
    if solver.plot_enabled:
        try:
            video_path = solver.write_video(run_id, output_name="lbm_video.mp4", fps=10)
            solver.logger.info("Video saved: %s", video_path)
        except (ImportError, RuntimeError) as e:
            solver.logger.warning(
                "Frames in %s; video not written: %s",
                Path(solver.plot_frame_dir) / run_id,
                e,
            )


def main_thermal():
    """Coupled thermal LBM: momentum (F) + energy (G); plotting to file with rho, u, T."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    X, Y = 64, 64
    lattice = D2Q9()
    dist_f = ParticleDistribution(tau=0.6)
    dist_g = ThermalDistribution(tau_T=0.5)
    steps = 50
    run_id = f"thermal_{np.random.randint(0, 10**9):09d}"
    solver = LBMSolver(
        lattice=lattice,
        dt=0.1,
        t_max=float(steps) * 0.1,
        distributions=(dist_f, dist_g),
        plot_enabled=True,
        plot_interval=max(1, steps // 15),
        plot_output_dir=Path.cwd(),
        plot_frame_dir="/tmp/lbm_frames",
        plot_fields=("rho", "u", "T"),
    )
    rho_init = jnp.ones((X, Y))[..., None]
    u_init = jnp.zeros((X, Y, 2))
    T_init = jnp.ones((X, Y))[..., None]
    T_init = T_init.at[X // 2, Y // 2].set(1.2)
    macro_init: MacroState = {"rho": rho_init, "u": u_init, "T": T_init}
    state: LBMState = {
        dist_f.label: dist_f.equilibrium(macro_init, lattice),
        dist_g.label: dist_g.equilibrium(macro_init, lattice),
        "rho": rho_init,
        "u": u_init,
        "T": T_init,
    }
    solver.logger.info("Coupled thermal (F + G): run_id=%s, plot_interval=%d", run_id, solver.plot_interval)
    state = solver.run(state, run_id, log_interval=10)
    if solver.plot_enabled:
        try:
            video_path = solver.write_video(run_id, output_name="lbm_thermal_video.mp4", fps=8)
            solver.logger.info("Video saved: %s", video_path)
        except (ImportError, RuntimeError) as e:
            solver.logger.warning("Frames in %s; video skipped: %s", Path(solver.plot_frame_dir) / run_id, e)


def main_boundaries():
    """Demonstrate boundary conditions: no-slip channel, then inlet/outlet (velocity + pressure)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    X, Y = 80, 40
    lattice = D2Q9()
    dist_f = ParticleDistribution(tau=0.6)
    steps = 1000
    plot_interval = 10
    run_id = f"bc_{np.random.randint(0, 10**9):09d}"

    # --- Example 1: Channel with no-slip walls (x), periodic (y) ---
    spec_channel = BoundarySpec(
        x=("no_slip", "no_slip"),
        y=("periodic", "periodic"),
    )
    solver = LBMSolver(
        lattice=lattice,
        dt=0.1,
        t_max=float(steps) * 0.1,
        distributions=(dist_f,),
        boundary_spec=spec_channel,
        plot_enabled=True,
        plot_interval=plot_interval,
        plot_output_dir=Path.cwd(),
        plot_frame_dir="/tmp/lbm_frames",
        plot_fields=("rho", "u"),
    )
    rho_init = jnp.ones((X, Y))[..., None]
    u_init = jnp.zeros((X, Y, 2))
    u_init = u_init.at[:, :, 0].set(0.02)
    macro_init: MacroState = {"rho": rho_init, "u": u_init}
    state: LBMState = {
        dist_f.label: dist_f.equilibrium(macro_init, lattice),
        "rho": macro_init["rho"],
        "u": macro_init["u"],
    }
    solver.logger.info("BC example 1: Channel (no-slip x, periodic y), initial u_x=0.02")
    state = solver.run(state, run_id, log_interval=20)
    if solver.plot_enabled:
        try:
            solver.write_video(run_id, output_name="lbm_bc_channel.mp4", fps=10)
            solver.logger.info("Video: lbm_bc_channel.mp4")
        except (ImportError, RuntimeError):
            pass

    # --- Example 2: Inlet (velocity) + outlet (outflow or pressure) ---
    run_id2 = f"bc_inout_{np.random.randint(0, 10**9):09d}"
    spec_inout = BoundarySpec(
        x=("velocity", "outflow"),
        y=("periodic", "periodic"),
    )
    solver2 = LBMSolver(
        lattice=lattice,
        dt=0.1,
        t_max=float(steps) * 0.1,
        distributions=(dist_f,),
        boundary_spec=spec_inout,
        plot_enabled=True,
        plot_interval=plot_interval,
        plot_output_dir=Path.cwd(),
        plot_frame_dir="/tmp/lbm_frames",
        plot_fields=("rho", "u"),
    )
    rho_init2 = jnp.ones((X, Y))[..., None]
    u_init2 = jnp.zeros((X, Y, 2))
    macro_init2: MacroState = {"rho": rho_init2, "u": u_init2}
    state2: LBMState = {
        dist_f.label: dist_f.equilibrium(macro_init2, lattice),
        "rho": macro_init2["rho"],
        "u": macro_init2["u"],
    }
    u_inlet = jnp.zeros((Y, 2))
    u_inlet = u_inlet.at[:, 0].set(0.05)
    solver2.logger.info("BC example 2: Inlet (velocity u_x=0.05) + outlet (outflow)")
    state2 = solver2.run(state2, run_id2, log_interval=20, boundary_velocity={(0, 0): u_inlet})
    if solver2.plot_enabled:
        try:
            solver2.write_video(run_id2, output_name="lbm_bc_inlet_outlet.mp4", fps=10)
            solver2.logger.info("Video: lbm_bc_inlet_outlet.mp4")
        except (ImportError, RuntimeError):
            pass

    # --- Example 3: Pressure outlet (prescribed rho at right) ---
    run_id3 = f"bc_pressure_{np.random.randint(0, 10**9):09d}"
    spec_pressure = BoundarySpec(
        x=("velocity", "pressure"),
        y=("periodic", "periodic"),
    )
    solver3 = LBMSolver(
        lattice=lattice,
        dt=0.1,
        t_max=float(steps) * 0.1,
        distributions=(dist_f,),
        boundary_spec=spec_pressure,
        plot_enabled=True,
        plot_interval=plot_interval,
        plot_output_dir=Path.cwd(),
        plot_frame_dir="/tmp/lbm_frames",
        plot_fields=("rho", "u"),
    )
    state3: LBMState = {
        dist_f.label: dist_f.equilibrium(macro_init2, lattice),
        "rho": macro_init2["rho"],
        "u": macro_init2["u"],
    }
    rho_outlet = jnp.ones((Y, 1)) * 0.98
    solver3.logger.info("BC example 3: Inlet (velocity) + outlet (pressure rho=0.98)")
    state3 = solver3.run(
        state3,
        run_id3,
        log_interval=20,
        boundary_velocity={(0, 0): u_inlet},
        boundary_pressure={(0, 1): rho_outlet},
    )
    if solver3.plot_enabled:
        try:
            solver3.write_video(run_id3, output_name="lbm_bc_pressure_outlet.mp4", fps=10)
            solver3.logger.info("Video: lbm_bc_pressure_outlet.mp4")
        except (ImportError, RuntimeError):
            pass
    solver3.logger.info("Boundary-condition examples done.")


if __name__ == "__main__":
    import sys

    # --thermal: coupled F + G; --bc: boundary-condition demos (no-slip, velocity, pressure)
    if "--thermal" in sys.argv:
        main_thermal()
    elif "--bc" in sys.argv:
        main_boundaries()
    else:
        main()
