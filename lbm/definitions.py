from enum import Enum


class Level(Enum):
    """Field level indicates the physical abstraction layer."""

    MACROSCOPIC = "macroscopic"  # Density, velocity, temperature
    MICROSCOPIC = "microscopic"  # Distribution functions f_i


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
