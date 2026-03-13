from lbm.jax_config import configure_jax

configure_jax()

from lbm.solvers import IsothermalSolver, CompressibleSolver
from lbm.lattice import D2Q9
from lbm.boundaries import BoundarySpec
from lbm.definitions import MacroState, LBMState

import jax.numpy as jnp
import numpy as np
import logging
from pathlib import Path


def _make_state(macro: MacroState, solver) -> LBMState:
    """Build an LBMState from macro fields and a solver's distributions."""
    dists = {d.label: d.equilibrium(macro, solver.lattice) for d in solver.distributions}
    return LBMState(
        rho=macro["rho"],
        u=macro["u"],
        T=macro.get("T"),
        dists=dists,
    )


def main():
    """Isothermal solver demo (single-distribution F)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    X, Y = 100, 100
    distance = X // 10
    lattice = D2Q9()
    steps = 100
    run_id = f"run_{np.random.randint(0, 10**9):09d}"

    solver = IsothermalSolver(
        lattice=lattice,
        dt=0.1,
        t_max=float(steps) * 0.1,
        Re=100.0,
        Ma=0.1,
        char_length=float(X),
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
    macro_init: MacroState = {"rho": rho_init[..., None], "u": u_init}
    state = _make_state(macro_init, solver)

    solver.logger.info(
        "IsothermalSolver: rho_sum=%.4f, sys_vars=%s",
        float(jnp.sum(state.rho)),
        solver.system_variables(),
    )
    state = solver.run(state, run_id, log_interval=10)
    solver.logger.info("Final: rho_sum=%.4f", float(jnp.sum(state.rho)))
    if solver.plot_enabled:
        try:
            solver.write_video(run_id, output_name="lbm_video.mp4", fps=10)
        except (ImportError, RuntimeError) as e:
            solver.logger.warning("Video skipped: %s", e)


def main_thermal():
    """Compressible solver demo (F + G with thermal coupling)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    X, Y = 64, 64
    lattice = D2Q9()
    steps = 50
    run_id = f"thermal_{np.random.randint(0, 10**9):09d}"

    solver = CompressibleSolver(
        lattice=lattice,
        dt=0.1,
        t_max=float(steps) * 0.1,
        Pr=0.71,
        gamma=1.4,
        viscosity=0.01,
        plot_enabled=True,
        plot_interval=max(1, steps // 15),
        plot_output_dir=Path.cwd(),
        plot_frame_dir="/tmp/lbm_frames",
        plot_fields=("rho", "u", "T"),
    )
    rho_init = jnp.ones((X, Y))[..., None]
    u_init = jnp.zeros((X, Y, 2))
    T_init = jnp.ones((X, Y))[..., None] * (1.0 / 3.0)
    T_init = T_init.at[X // 2, Y // 2].set(0.4)
    macro_init: MacroState = {"rho": rho_init, "u": u_init, "T": T_init}
    state = _make_state(macro_init, solver)

    solver.logger.info(
        "CompressibleSolver: run_id=%s, sys_vars=%s", run_id, solver.system_variables(),
    )
    state = solver.run(state, run_id, log_interval=10)
    if solver.plot_enabled:
        try:
            solver.write_video(run_id, output_name="lbm_thermal_video.mp4", fps=8)
        except (ImportError, RuntimeError) as e:
            solver.logger.warning("Video skipped: %s", e)


def main_boundaries():
    """Boundary condition demos using IsothermalSolver."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    X, Y = 80, 40
    lattice = D2Q9()
    steps = 1000
    plot_interval = 10

    # --- Example 1: Channel with no-slip walls (x), periodic (y) ---
    run_id = f"bc_{np.random.randint(0, 10**9):09d}"
    solver = IsothermalSolver(
        lattice=lattice,
        dt=0.1,
        t_max=float(steps) * 0.1,
        Re=100.0,
        Ma=0.02,
        char_length=float(X),
        boundary_spec=BoundarySpec(x=("no_slip", "no_slip"), y=("periodic", "periodic")),
        plot_enabled=True,
        plot_interval=plot_interval,
        plot_output_dir=Path.cwd(),
        plot_frame_dir="/tmp/lbm_frames",
    )
    rho_init = jnp.ones((X, Y))[..., None]
    u_init = jnp.zeros((X, Y, 2)).at[:, :, 0].set(0.02)
    state = _make_state({"rho": rho_init, "u": u_init}, solver)
    solver.logger.info("BC example 1: Channel (no-slip x, periodic y)")
    state = solver.run(state, run_id, log_interval=20)
    if solver.plot_enabled:
        try:
            solver.write_video(run_id, output_name="lbm_bc_channel.mp4", fps=10)
        except (ImportError, RuntimeError):
            pass

    # --- Example 2: Inlet (velocity) + outlet (outflow) ---
    run_id2 = f"bc_inout_{np.random.randint(0, 10**9):09d}"
    solver2 = IsothermalSolver(
        lattice=lattice,
        dt=0.1,
        t_max=float(steps) * 0.1,
        Re=100.0,
        Ma=0.05,
        char_length=float(X),
        boundary_spec=BoundarySpec(x=("velocity", "outflow"), y=("periodic", "periodic")),
        plot_enabled=True,
        plot_interval=plot_interval,
        plot_output_dir=Path.cwd(),
        plot_frame_dir="/tmp/lbm_frames",
    )
    state2 = _make_state({"rho": jnp.ones((X, Y))[..., None], "u": jnp.zeros((X, Y, 2))}, solver2)
    u_inlet = jnp.zeros((Y, 2)).at[:, 0].set(0.05)
    solver2.logger.info("BC example 2: Inlet (velocity) + outlet (outflow)")
    state2 = solver2.run(state2, run_id2, log_interval=20, boundary_velocity={(0, 0): u_inlet})
    if solver2.plot_enabled:
        try:
            solver2.write_video(run_id2, output_name="lbm_bc_inlet_outlet.mp4", fps=10)
        except (ImportError, RuntimeError):
            pass

    # --- Example 3: Pressure outlet ---
    run_id3 = f"bc_pressure_{np.random.randint(0, 10**9):09d}"
    solver3 = IsothermalSolver(
        lattice=lattice,
        dt=0.1,
        t_max=float(steps) * 0.1,
        Re=100.0,
        Ma=0.05,
        char_length=float(X),
        boundary_spec=BoundarySpec(x=("velocity", "pressure"), y=("periodic", "periodic")),
        plot_enabled=True,
        plot_interval=plot_interval,
        plot_output_dir=Path.cwd(),
        plot_frame_dir="/tmp/lbm_frames",
    )
    state3 = _make_state({"rho": jnp.ones((X, Y))[..., None], "u": jnp.zeros((X, Y, 2))}, solver3)
    rho_outlet = jnp.ones((Y, 1)) * 0.98
    solver3.logger.info("BC example 3: Inlet (velocity) + outlet (pressure rho=0.98)")
    state3 = solver3.run(
        state3, run_id3, log_interval=20,
        boundary_velocity={(0, 0): u_inlet},
        boundary_pressure={(0, 1): rho_outlet},
    )
    if solver3.plot_enabled:
        try:
            solver3.write_video(run_id3, output_name="lbm_bc_pressure_outlet.mp4", fps=10)
        except (ImportError, RuntimeError):
            pass
    solver3.logger.info("Boundary-condition examples done.")


if __name__ == "__main__":
    import sys

    if "--thermal" in sys.argv:
        main_thermal()
    elif "--bc" in sys.argv:
        main_boundaries()
    else:
        main()
