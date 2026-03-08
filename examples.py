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
from pathlib import Path


def main():
    """Run single-distribution (isothermal); plotting to file when enabled."""
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
        plot_interval=max(1, steps // 25),
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

    print("Initial state (single-distribution):")
    print(f"  rho sum: {float(np.sum(state['rho'])):.4f}")
    print(f"  Plotting: enabled, interval={solver.plot_interval}, run_id={run_id}")

    for i in range(steps):
        state = solver.step(state)
        if i % 10 == 0:
            u_mag = (state["u"] ** 2).sum(axis=-1) ** 0.5
            print(
                f"Step {i}: rho_sum={float(np.sum(state['rho'])):.4f}, "
                f"u_max={float(np.max(np.asarray(u_mag))):.6f}"
            )
        if solver.plot_enabled and (i % solver.plot_interval == 0 or i == steps - 1):
            solver.plot(state, i, run_id)

    print("\nFinal state:")
    print(f"  rho sum: {float(np.sum(state['rho'])):.4f}")
    if solver.plot_enabled:
        try:
            video_path = solver.write_video(run_id, output_name="lbm_video.mp4", fps=10)
            print(f"  Video saved: {video_path}")
        except (ImportError, RuntimeError) as e:
            print(f"  Frames in {Path(solver.plot_frame_dir) / run_id}; video not written: {e}")


def main_thermal():
    """Coupled thermal LBM: momentum (F) + energy (G); plotting to file with rho, u, T."""
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
    print("Coupled thermal (F + G): initial state, plotting to file")
    print(f"  run_id={run_id}, plot_interval={solver.plot_interval}")
    for i in range(steps):
        state = solver.step(state)
        if i % 10 == 0:
            T = state["T"]
            print(f"Step {i}: rho_sum={float(np.sum(state['rho'])):.4f}, T_max={float(np.max(np.asarray(T))):.4f}")
        if solver.plot_enabled and (i % solver.plot_interval == 0 or i == steps - 1):
            solver.plot(state, i, run_id)
    if solver.plot_enabled:
        try:
            video_path = solver.write_video(run_id, output_name="lbm_thermal_video.mp4", fps=8)
            print(f"  Video saved: {video_path}")
        except (ImportError, RuntimeError) as e:
            print(f"  Frames in {Path(solver.plot_frame_dir) / run_id}; video skipped: {e}")


def main_boundaries():
    """Demonstrate boundary conditions: no-slip channel, then inlet/outlet (velocity + pressure)."""
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
    # Slight horizontal flow to see wall effect
    u_init = u_init.at[:, :, 0].set(0.02)
    macro_init: MacroState = {"rho": rho_init, "u": u_init}
    state: LBMState = {
        dist_f.label: dist_f.equilibrium(macro_init, lattice),
        "rho": macro_init["rho"],
        "u": macro_init["u"],
    }
    print("BC example 1: Channel (no-slip x, periodic y), initial u_x=0.02")
    for i in range(steps):
        state = solver.step(state)
        if i % 20 == 0:
            u_mag = (state["u"] ** 2).sum(axis=-1) ** 0.5
            print(f"  Step {i}: rho_sum={float(np.sum(state['rho'])):.4f}, u_max={float(np.max(np.asarray(u_mag))):.6f}")
        if solver.plot_enabled and (i % solver.plot_interval == 0 or i == steps - 1):
            solver.plot(state, i, run_id)
    if solver.plot_enabled:
        try:
            solver.write_video(run_id, output_name="lbm_bc_channel.mp4", fps=10)
            print("  Video: lbm_bc_channel.mp4")
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
    # Prescribed inlet velocity at left face: (axis=0, side=0), shape (Y, 2)
    u_inlet = jnp.zeros((Y, 2))
    u_inlet = u_inlet.at[:, 0].set(0.05)
    print("\nBC example 2: Inlet (velocity u_x=0.05) + outlet (outflow)")
    for i in range(steps):
        state2 = solver2.step(state2, boundary_velocity={(0, 0): u_inlet})
        if i % 20 == 0:
            print(f"  Step {i}: rho_sum={float(np.sum(state2['rho'])):.4f}")
        if solver2.plot_enabled and (i % solver2.plot_interval == 0 or i == steps - 1):
            solver2.plot(state2, i, run_id2)
    if solver2.plot_enabled:
        try:
            solver2.write_video(run_id2, output_name="lbm_bc_inlet_outlet.mp4", fps=10)
            print("  Video: lbm_bc_inlet_outlet.mp4")
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
    print("\nBC example 3: Inlet (velocity) + outlet (pressure rho=0.98)")
    for i in range(steps):
        state3 = solver3.step(
            state3,
            boundary_velocity={(0, 0): u_inlet},
            boundary_pressure={(0, 1): rho_outlet},
        )
        if i % 20 == 0:
            print(f"  Step {i}: rho_sum={float(np.sum(state3['rho'])):.4f}")
        if solver3.plot_enabled and (i % solver3.plot_interval == 0 or i == steps - 1):
            solver3.plot(state3, i, run_id3)
    if solver3.plot_enabled:
        try:
            solver3.write_video(run_id3, output_name="lbm_bc_pressure_outlet.mp4", fps=10)
            print("  Video: lbm_bc_pressure_outlet.mp4")
        except (ImportError, RuntimeError):
            pass
    print("Boundary-condition examples done.")


if __name__ == "__main__":
    import sys

    # --thermal: coupled F + G; --bc: boundary-condition demos (no-slip, velocity, pressure)
    if "--thermal" in sys.argv:
        main_thermal()
    elif "--bc" in sys.argv:
        main_boundaries()
    else:
        main()
