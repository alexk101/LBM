# distributions.py
from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from lattice import Lattice


class Distribution(eqx.Module):
    """Base class for LBM distribution functions.

    Each distribution represents a population of particles moving in discrete
    velocity directions defined by the lattice. The equilibrium and lift
    methods implement the physics-specific transformations.

    Design Principles:
        1. Pure functions - no side effects, easy to test independently
        2. Required parameters (rho, u) for standard LBM variants
        3. Optional parameters (T, etc.) for extended physics models

    Parameters:
        name: Identifier string for this distribution type.
        label: Symbolic label used in output fields and diagnostics.
    """

    name: str = eqx.field(static=True)
    label: str = eqx.field(static=True)

    @abstractmethod
    def equilibrium(self, rho: Array, u: Array, lattice: Lattice) -> Array:
        """Computes the equilibrium distribution f_eq from macroscopic fields.

        Args:
            rho: Density field with shape (...,). Represents mass per cell.
                 Must be positive everywhere for numerical stability.
            u: Velocity field with shape (..., D). Vector velocity at each point.
               Should satisfy |u| << c_s for low-Mach-number assumptions.
            lattice: Lattice configuration providing velocities and weights.

        Returns:
            Equilibrium distribution f_eq with shape (..., N) where N is the
            number of discrete velocity directions on the lattice.

        Raises:
            NotImplementedError: If subclass does not implement this method.
        """
        raise NotImplementedError()

    @abstractmethod
    def lift(self, f: Array, lattice: Lattice) -> dict[str, Array]:
        """Extracts macroscopic fields from microscopic variables.

        Computes the moments of the distribution function to recover
        density, velocity (and potentially temperature or other quantities).

        Args:
            f: Microscopic distribution with shape (..., N). The result
               of streaming/collision steps that may deviate from equilibrium.
            lattice: Lattice configuration providing velocities for moment calculation.

        Returns:
            Dictionary mapping field names to their computed values:
                - "rho": Density field (zeroth moment), shape (..., 1)
                - "u": Velocity field (first moment / rho), shape (..., D)
                - Additional fields as needed by specific physics models.
        """
        raise NotImplementedError()


class ParticleDistribution(Distribution):
    """Standard single-distribution LBM for incompressible flow.

    The most common LBM model using the BGK collision operator with a
    single distribution function F. Suitable for low-Mach-number,
    isothermal incompressible flows.

    Physics:
        - BGK collision: f_i(x + e_i*dt, t+dt) = f_i(x,t) - ω(f_i - f_i^eq)
        - Equilibrium: Maxwell-Boltzmann expansion to second order
        - Viscosity: ν = c_s²(τ - 0.5), where τ is relaxation time

    Parameters:
        name: Identifier for this distribution (default: "Particle")
        label: Symbolic label used in output fields (default: "F")
        tau: Relaxation time parameter controlling viscosity (default: 0.5)

    Stability Notes:
        - τ must be > 0.5 for numerical stability (positive viscosity)
        - Typical range: 0.6 ≤ τ ≤ 1.0 balances stability and accuracy
        - For high Reynolds numbers, may need adaptive τ or MRT collision
    """

    name: str = "Particle"
    label: str = "F"
    tau: float = 0.5

    def equilibrium(self, rho: Array, u: Array, lattice: Lattice) -> Array:
        """Computes the Maxwell-Boltzmann equilibrium distribution.

        Mathematical Formula:
            f_eq,i = w_i * ρ * [1 + (e_i·u)/c_s² + ((e_i·u)²)/(2*c_s⁴) - (u·u)/(2*c_s²)]

        Where:
            e_dot_u = Σ_c e_i,c * u_c  (dot product of lattice and fluid velocities)
            w_i     = weight for direction i
            c_s²    = speed of sound squared in lattice units
            u·u     = |u|² = velocity magnitude squared

        This is the second-order Taylor expansion of the Maxwell-Boltzmann
        distribution, valid for low Mach numbers (Ma = |u|/c_s << 1).

        Args:
            rho: Density field with shape (...,). Positive density required.
            u: Velocity field with shape (..., D). Vector velocity at each point.
            lattice: Lattice configuration providing velocities and weights.

        Returns:
            Equilibrium distribution f_eq with shape (..., N) where N is the
            number of discrete velocity directions on the lattice.

        Mathematical Details:
            Let e_dot_u = Σ_c e_i,c * u_c  (dot product of lattice and fluid velocities)
            Then: f_eq,i = w_i * ρ * [1 + e_dot_u/c_s² + e_dot_u²/(2c_s⁴) - |u|²/(2c_s²)]

        Numerical Notes:
            - The Taylor expansion is truncated at second order for efficiency
            - Higher-order terms become significant when Ma > 0.15
            - For very low velocities, the equilibrium approaches isotropic state
        """
        # Compute e_i · u for all directions and grid points
        # Shape: (..., N) where last dim is over discrete velocity directions
        e_dot_u = jnp.einsum("...c,dc->...d", u, lattice.velocities)

        # Compute |u|² (velocity magnitude squared) at each point
        # Shape: (..., 1) for proper broadcasting
        u_sq = jnp.sum(u**2, axis=-1, keepdims=True)

        # Taylor expansion of Maxwell-Boltzmann equilibrium
        # Weights are broadcast to match the output shape via [None, ...]
        f_eq = (
            rho[..., None]
            * lattice.weights[lattice.indices][None, :]
            * (
                1.0
                + e_dot_u / lattice.c_s_sq
                + e_dot_u**2 / (2.0 * lattice.c_s_sq**2)
                - u_sq / (2.0 * lattice.c_s_sq)
            )
        )

        return f_eq

    def lift(self, f: Array, lattice: Lattice) -> dict[str, Array]:
        """Extracts macroscopic fields from distribution function via moment calculation.

        Mathematical Formula:
            ρ = Σ_i f_i                           (zeroth moment - conservation of mass)
            ρu = Σ_i f_i * e_i                    (first moment - conservation of momentum)
            u = (ρu) / ρ                          (velocity from momentum density)

        The moments of the distribution function give macroscopic quantities:
            - Zeroth moment gives total density (mass per cell)
            - First moment gives momentum density (mass-weighted velocity sum)
            - Velocity is obtained by dividing momentum by density

        Args:
            f: Microscopic distribution with shape (..., N). May be out
               of equilibrium after collision/streaming steps.
            lattice: Lattice configuration providing velocities for moment calculation.

        Returns:
            Dictionary containing:
                "rho": Density field from zeroth moment, shape (..., 1)
                "u": Velocity field from first moment divided by rho,
                     shape (..., D). Computed with small offset to avoid div by zero.

        Numerical Notes:
            - A small offset (1e-10) is added to denominator for numerical stability
            - In regions where f approaches zero, velocity may become noisy
            - Consider adding floor on rho if simulating near-vacuum conditions
        """
        # Zeroth moment = density
        rho = jnp.sum(f, axis=-1, keepdims=True)

        # First moment = momentum (mass-weighted velocity sum)
        momentum = jnp.einsum("...i,ic->...c", f, lattice.velocities)

        # Velocity = momentum / density with numerical stability term
        u = momentum / (rho + 1e-10)

        return {"rho": rho, "u": u}


class ThermalDistribution(Distribution):
    """Energy distribution for thermal LBM using double-distribution approach.

    This model adds temperature transport to the standard incompressible flow,
    enabling simulation of natural convection and heat transfer phenomena.

    Physics:
        - Two separate distributions: F (momentum) + G (energy/temperature)
        - Temperature is advected by velocity field from F distribution
        - Thermal diffusivity controlled by τ_T parameter

    Parameters:
        name: Identifier for this distribution (default: "Thermal")
        label: Symbolic label used in output fields (default: "G")
        tau_T: Relaxation time for temperature field (default: 0.5)
        gamma: Heat capacity ratio (adiabatic index, default: 1.4 for air)

    Coupled Simulation Notes:
        - Use ParticleDistribution + ThermalDistribution together
        - Prandtl number = ν/α ≈ τ/τ_T (ratio of momentum to thermal diffusivity)
        - Typical values: Pr ~ 0.7 for gases, Pr ~ 7 for water
    """

    name: str = "Thermal"
    label: str = "G"
    tau_T: float = 0.5
    gamma: float = 1.4

    def equilibrium(self, rho: Array, u: Array, T: Array, lattice: Lattice) -> Array:
        """Computes the thermal equilibrium distribution for temperature transport.

        Mathematical Formula:
            g_eq,i = ρ * T / (2πc_s²)^D/2 * exp(-|e_i - u|²/(2c_s²))

        Where:
            e_minus_u = e_i - u  (peculiar velocity)
            |e_minus_u|² = Σ_c (e_i,c - u_c)²  (squared peculiar velocity magnitude)
            Z = (2πc_s²)^D/2  (partition function for normalization)

        This represents the high-speed expansion of the Maxwell-Boltzmann
        distribution adapted for scalar (temperature) transport.

        Args:
            rho: Density field with shape (...,). Used for normalization.
            u: Velocity field with shape (..., D). Creates peculiar velocity.
            T: Temperature field with shape (...,). Absolute temperature at each point.
            lattice: Lattice configuration providing velocities and weights.

        Returns:
            Thermal equilibrium distribution g_eq with shape (..., N).

        Mathematical Details:
            Let e_minus_u = e_i - u  (peculiar velocity)
            Then: g_eq,i = ρ * T / Z * exp(-|e_minus_u|²/(2c_s²))
            where Z is the partition function ensuring normalization.

        Physical Notes:
            - The exponential term penalizes large deviations from fluid velocity
            - Temperature diffuses via lattice propagation, advected by u
            - At equilibrium (u=0), distribution becomes isotropic Gaussian
        """
        # Peculiar velocity: e_i - u (velocity relative to local fluid)
        # Shape: (..., N, D) where N is over directions, D is spatial dimensions
        peculiar_velocity = lattice.velocities[None, ...] - u[..., None, :]

        # Squared magnitude of peculiar velocity |e_i - u|²
        e_minus_u_sq = jnp.sum(peculiar_velocity**2, axis=-1)

        # Exponential term from Maxwell-Boltzmann distribution
        exp_term = jnp.exp(-e_minus_u_sq / (2.0 * lattice.c_s_sq))

        # Pre-factor including temperature and density normalization
        g_eq = (
            rho[..., None] * T[..., None] / (2.0 * jnp.pi * lattice.c_s_sq) ** 1.0
        ) * exp_term

        return g_eq

    def lift(self, g: Array, lattice: Lattice) -> dict[str, Array]:
        """Extracts temperature from thermal distribution via moment calculation.

        Mathematical Formula:
            T = Σ_i g_i / ρ  (zeroth moment normalized by density)

        The zeroth moment of the energy distribution gives the temperature field.
        This is analogous to how density is extracted from F distribution.

        Args:
            g: Thermal distribution with shape (..., N). May be out of equilibrium.
            lattice: Lattice configuration providing velocities for moment calculation.

        Returns:
            Dictionary containing:
                "T": Temperature field from zeroth moment, shape (..., 1)
        """
        # Zeroth moment = temperature (with density normalization)
        T = jnp.sum(g, axis=-1, keepdims=True)

        return {"T": T}


class MultiphaseDistribution(Distribution):
    """Density-dependent distribution for multiphase flow using Shan-Chen model.

    This implements the pseudo-potential (Shan-Chen) method for simulating
    immiscible fluid interfaces and surface tension effects without explicit
    interface tracking.

    Physics:
        - Fluid particles interact via short-range repulsive/attractive force
        - The interaction strength controls interfacial surface tension
        - Density variations naturally lead to phase separation

    Parameters:
        name: Identifier for this distribution (default: "Multiphase")
        label: Symbolic label used in output fields (default: "F")
        tau: Relaxation time parameter (default: 0.5)
        interaction_strength: Force coupling parameter σ (default: -1.0).
                              Negative values lead to phase separation.

    Phase Separation Notes:
        - |σ| < 0.25 for stable simulations without spurious currents
        - More negative = stronger surface tension and sharper interfaces
        - Density ratio between phases limited (~10:1 typical)
    """

    name: str = "Multiphase"
    label: str = "F"
    tau: float = 0.5
    interaction_strength: float = -1.0

    def equilibrium(self, rho: Array, u: Array, lattice: Lattice) -> Array:
        """Computes multiphase equilibrium with density-dependent weights.

        Mathematical Formula:
            f_eq,i = w_i * ψ(ρ) * [1 + (e_i·u)/c_s² + ((e_i·u)²)/(2*c_s⁴) - (u·u)/(2*c_s²)]

        Where:
            ψ(ρ) = ρ + σ*ρ²  (effective density/potential function including interaction)
            e_dot_u = Σ_c e_i,c * u_c  (dot product of lattice and fluid velocities)
            w_i     = weight for direction i
            c_s²    = speed of sound squared in lattice units

        The effective density ψ(ρ) creates a non-linear coupling between
        neighboring cells through the interaction term, leading to phase separation.

        Args:
            rho: Density field with shape (...,). Varies between phases.
            u: Velocity field with shape (..., D). Same for both phases.
            lattice: Lattice configuration providing velocities and weights.

        Returns:
            Equilibrium distribution f_eq with shape (..., N).

        Mathematical Details:
            Effective density: ψ(ρ) = ρ + σ*ρ²
            This creates a non-linear coupling between neighboring cells
            through the interaction term, leading to phase separation.

        Stability Notes:
            - Choose |σ| < 0.25 for numerical stability
            - Stronger interactions require smaller time steps
            - Spurious currents may appear near interfaces
        """
        # Compute effective density including interaction term (Shan-Chen potential)
        # ψ = ρ + σ*ρ² creates the non-linear coupling between phases
        psi = rho[..., None] + self.interaction_strength * rho**2

        # Standard equilibrium computation but using effective density instead of raw rho
        e_dot_u = jnp.einsum("...c,dc->...d", u, lattice.velocities)
        u_sq = jnp.sum(u**2, axis=-1, keepdims=True)

        f_eq = (
            psi
            * lattice.weights[lattice.indices][None, :]
            * (
                1.0
                + e_dot_u / lattice.c_s_sq
                + e_dot_u**2 / (2.0 * lattice.c_s_sq**2)
                - u_sq / (2.0 * lattice.c_s_sq)
            )
        )

        return f_eq

    def lift(self, f: Array, lattice: Lattice) -> dict[str, Array]:
        """Extracts density and velocity from multiphase distribution.

        Mathematical Formula:
            ρ = Σ_i f_i                           (zeroth moment - conservation of mass)
            ρu = Σ_i f_i * e_i                    (first moment - conservation of momentum)
            u = (ρu) / ρ                          (velocity from momentum density)

        Same moment extraction as standard LBM, but the resulting fields
        show phase separation due to the interaction term in equilibrium.

        Args:
            f: Distribution with shape (..., N). Contains information about
               both phases after collision/streaming steps.
            lattice: Lattice configuration providing velocities for moment calculation.

        Returns:
            Dictionary containing:
                "rho": Density field showing phase distribution, shape (..., 1)
                "u": Velocity field with shape (..., D)
        """
        rho = jnp.sum(f, axis=-1, keepdims=True)
        momentum = jnp.einsum("...i,ic->...c", f, lattice.velocities)
        u = momentum / (rho + 1e-10)

        return {"rho": rho, "u": u}


class SinglePhaseDistribution(Distribution):
    """Simplified single-phase distribution for educational purposes.

    This is a minimal implementation focusing on the core LBM mechanics
    without additional physics complexities. Good for learning and debugging.

    Simplifications:
        - Uses first-order Taylor expansion (less accurate than ParticleDistribution)
        - No explicit viscosity control via tau in equilibrium formula
        - Minimal error for low Mach numbers but less robust

    Parameters:
        name: Identifier (default: "SinglePhase")
        label: Symbolic label (default: "F")
        tau: Relaxation time parameter (default: 0.5)
    """

    name: str = "SinglePhase"
    label: str = "F"
    tau: float = 0.5

    def equilibrium(self, rho: Array, u: Array, lattice: Lattice) -> Array:
        """Computes simplified BGK equilibrium with first-order expansion.

        Mathematical Formula (Simplified):
            f_eq,i ≈ w_i * ρ * [1 + (e_i·u)/c_s² - 0.5*|u|²]

        This is a first-order Taylor approximation of the full Maxwell-Boltzmann
        equilibrium, missing the ((e_i·u)²)/(2*c_s⁴) term for simplicity.

        Args:
            rho: Density field with shape (...,). Positive density required.
            u: Velocity field with shape (..., D). Vector velocity at each point.
            lattice: Lattice configuration providing velocities and weights.

        Returns:
            Equilibrium distribution f_eq with shape (..., N).

        Notes:
            - Less accurate than ParticleDistribution for higher Mach numbers
            - Simpler form useful for educational purposes or quick prototyping
            - Recommended to use ParticleDistribution for production simulations
        """
        e_dot_u = jnp.einsum("...c,dc->...d", u, lattice.velocities)

        # Simplified form (less accurate than full second-order expansion)
        f_eq = (
            rho[..., None]
            * lattice.weights[lattice.indices][None, :]
            * (1.0 + e_dot_u / lattice.c_s_sq - 0.5 * u**2)
        )

        return f_eq

    def lift(self, f: Array, lattice: Lattice) -> dict[str, Array]:
        """Extracts macroscopic fields via moment calculation."""
        rho = jnp.sum(f, axis=-1, keepdims=True)
        momentum = jnp.einsum("...i,ic->...c", f, lattice.velocities)
        u = momentum / (rho + 1e-10)

        return {"rho": rho, "u": u}
