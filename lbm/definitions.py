from enum import Enum

import jax.numpy as jnp
from jax import Array

# Type for shared macroscopic state: ρ, u, T, etc. (keys are field names)
MacroState = dict[str, Array]

# State is a flat dict: distribution arrays keyed by dist.label, plus macro keys
LBMState = dict[str, Array]

class Level(Enum):
    """Field level indicates the physical abstraction layer."""

    MACROSCOPIC = "macroscopic"  # Density, velocity, temperature
    MICROSCOPIC = "microscopic"  # Distribution functions f_i


# Plotting (resolution for saved frames and video)
PLOT_DPI: int = 150

# BGK stability: ν = c_s²(τ - 0.5) requires τ ≥ 0.5. τ < 0.5 gives negative ν and NaNs.
TAU_MIN: float = 0.5

# Precision: compute in float32; optional bf16 storage for state (see LBMSolver.use_mixed_precision)
DTYPE = jnp.float32
DTYPE_LOW = jnp.float32  # use only for storage of f, rho, u, T between steps


# Physical constants
R = 8.31446261  # J/(mol*K)

# The boltzmann BGK equation for conserved mass and energy:
# $D$ = dimension
# $Q$ = number of directions
# $R$ = gas constant (molar Boltzmann constant) = $8.31446261$ J/(mol*K)
# $c$ = speed of sound = $\sqrt{\gamma R_{*} T}$
# $c_s$ = speed of sound in given medium s
# $\gamma$ = heat capacity ratio / adiabatic index = 1.4 (room temperature, dry air)
# $R_{*}$ = gas constant for dry air = $R / M_{air}$
# $M_{air}$ = molar mass of dry air in = $0.0289644$ kg/mol
# $\tau$ = time relaxation constant
# $\nu$ = fluid viscosity = $\left(\tau - \frac{1}{2}\right) c_s^2$

# Macroscopic variables:
# $\rho$ = density
# $\vec{u}$ = velocity
# $T$ = absolute temperature

# Microscopic variables:
# $\vec{e}$ = velocity
