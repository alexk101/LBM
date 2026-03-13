"""LBM solver hierarchy.

All solvers inherit from :class:`BaseSolver` and share the same step/run/run_jit
interface.  Each solver specialises the physics (distributions, source terms,
collision logic) via the ``_collide_and_stream`` hook.
"""

from .acoustic import AcousticSolver, AcousticSource
from .base import BaseSolver
from .compressible import CompressibleSolver
from .isothermal import IsothermalSolver
from .multiphase import MultiphaseSolver
from .shallow_water import ShallowWaterSolver

__all__ = [
    "BaseSolver",
    "IsothermalSolver",
    "ShallowWaterSolver",
    "CompressibleSolver",
    "AcousticSolver",
    "AcousticSource",
    "MultiphaseSolver",
]
