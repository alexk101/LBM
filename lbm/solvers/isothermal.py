"""Weakly compressible isothermal LBM solver.

Single distribution F with polynomial equilibrium and constant relaxation
time.  Suitable for low-Mach-number incompressible flows (Stokes, channel
flow, lid-driven cavity, etc.).

Physical parameters Re and Ma are converted to the LBM relaxation time via:
    V0  = Ma * c_s                      (reference velocity)
    nu  = V0 * L / Re                   (kinematic viscosity)
    tau = nu / c_s^2 + 0.5              (BGK relaxation time)
where L is the characteristic length (taken as the first spatial dimension).
"""
from __future__ import annotations

import equinox as eqx

from ..distributions import ParticleDistribution, Distribution
from ..equilibrium import EquilibriumModel
from ..lattice import Lattice
from .base import BaseSolver


class IsothermalSolver(BaseSolver):
    """Weakly compressible isothermal LBM (single distribution F).

    Specify physical parameters ``Re`` and ``Ma``; the solver computes tau
    automatically.  Alternatively, pass ``tau`` directly and leave Re/Ma as
    defaults.

    Args:
        lattice: Lattice definition.
        dt: Time step.
        t_max: Total simulation time (steps = t_max / dt).
        Re: Reynolds number.
        Ma: Mach number.
        char_length: Characteristic length in lattice units (default: inferred
            at construction time — you can override if your domain size differs
            from the first spatial dimension).
        equilibrium_model: Override equilibrium (default: polynomial).
    """

    Re: float = 100.0
    Ma: float = 0.1
    char_length: float = 1.0
    equilibrium_model: EquilibriumModel | None = None

    _dist_f: ParticleDistribution = eqx.field(init=False)

    def __post_init__(self) -> None:
        cs2 = self.lattice.c_s_sq
        cs = cs2 ** 0.5
        V0 = self.Ma * cs
        nu = V0 * self.char_length / self.Re
        tau = nu / cs2 + 0.5
        object.__setattr__(
            self, "_dist_f",
            ParticleDistribution(tau=tau, equilibrium_model=self.equilibrium_model),
        )
        super().__post_init__()

    @property
    def distributions(self) -> tuple[Distribution, ...]:
        return (self._dist_f,)

    def system_variables(self) -> dict[str, float]:
        cs2 = self.lattice.c_s_sq
        cs = cs2 ** 0.5
        V0 = self.Ma * cs
        nu = V0 * self.char_length / self.Re
        tau = nu / cs2 + 0.5
        return {
            "Re": self.Re,
            "Ma": self.Ma,
            "V0": V0,
            "nu": nu,
            "tau": tau,
            "c_s": cs,
        }
