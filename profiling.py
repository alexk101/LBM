from lbm.jax_config import configure_jax
import jax

configure_jax()

from lbm.solvers import IsothermalSolver
from lbm.lattice import D2Q9
from lbm.definitions import MacroState, LBMState

import jax.numpy as jnp
import numpy as np
import logging
from pathlib import Path


def main():
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
    state = LBMState(
        rho=macro_init["rho"],
        u=macro_init["u"],
        dists={d.label: d.equilibrium(macro_init, lattice) for d in solver.distributions},
    )

    solver.logger.info(
        "Initial state: rho_sum=%.4f", float(jnp.sum(state.rho)),
    )

    state = solver.step(state)
    jax.block_until_ready(state)

    with jax.profiler.trace("./jax-trace", create_perfetto_link=False):
        state = solver.run(state, run_id, log_interval=5)
        jax.block_until_ready(state)

    try:
        from lbm.utils.trace_report import report_trace
        report_trace("./jax-trace", print_report=True)
    except Exception as e:
        solver.logger.warning("Trace report skipped: %s", e)


if __name__ == "__main__":
    main()