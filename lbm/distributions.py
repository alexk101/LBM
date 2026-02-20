from dataclasses import dataclass
from abc import ABC, abstractmethod
from fields import Field, Field2D
from lattice import Lattice
from definitions import *
import jax.numpy as jnp

@dataclass
class Distribution(ABC):
    name: str
    label: str
    i: Field | None
    eq: Field | None

    @abstractmethod
    def init_microscopic(self, lattice: Lattice):
        pass

    @abstractmethod
    def equilibrium(self, lattice: Lattice):
        pass

    @abstractmethod
    def lift(self, lattice: Lattice):
        pass


@dataclass
class Particle(Distribution):
    u: Field
    rho: Field
    tau: float
    name = "Particle"
    label = "F"

    def init_microscopic(self, lattice: Lattice):
        assert lattice.D in [2], f"Lattice dimension of {lattice.D} not supported"
        if lattice.D == 2:
            eq = Field2D(
                f"{self.label}_eq",
                self.u.shape[0],
                self.u.shape[1],
                Level.Microscopic,
                "lattice units",
                list(range(lattice.N))
            )
            i = Field2D(
                f"{self.label}",
                self.u.shape[0],
                self.u.shape[1],
                Level.Microscopic,
                "lattice units",
                list(range(lattice.N))
            )
        self.i = i
        self.eq = eq
        self.eq.field = self.equilibrium(lattice)
        self.i.field = self.eq.field

    def equilibrium(self, lattice: Lattice):
        # $e_dot_u = \vec{e} \cdot \vec{u}$ | shape: (X, Y, lattice.N)
        # $eu_i$ = the i-th component of $e_dot_u$ | The ith item in final dimension of $e_dot_u$
        # $f^{eq}$ = equilibrium distribution function (Maxwell-Boltzmann distribution) | shape: (X, Y, lattice.N)
        # $f^{eq}_i$ = equilibrium distribution function for the i-th direction | The ith item in final dimension of $f^{eq}$
        # $f^{eq}_i = \omega_i \rho ( 1 + \frac{\vec{e_dot_u}_i}{c_s^2} + \frac{(\vec{e_dot_u}_i)^2}{2 (c_s^2)^2} - \frac{(\vec{u})^2}{2 c_s^2})$
        e_dot_u = jnp.einsum("...c,dc->...d", self.u.field, lattice.microscopic_velocities)
        taylor = (
            1
            + (e_dot_u / lattice.c_squared)
            + ((e_dot_u**2) / (2 * lattice.c_squared**2))
            - (jnp.sum(self.u.field ** 2, axis=-1, keepdims=True) / (2 * lattice.c_squared))
        )
        return self.rho.field * lattice.weights[lattice.indices] * taylor

    def lift(self, lattice: Lattice):
        # Density = zeroth moment of f
        # \rho = \sum_i f_{xyi}
        self.rho.field = jnp.sum(self.i.field, axis=-1, keepdims=True)
        # Momentum =  first moment -> transitively u
        # \rho u = \sum_i f_{xyi}e_i
        self.u.field = jnp.einsum("...i,ic->...c", self.i.field, lattice.microscopic_velocities) / self.rho.field


@dataclass
class Energy(Distribution):
    u: Field
    rho: Field
    tau: float
    name = "Energy"
    label = "G"

    def init_microscopic(self, lattice: Lattice):
        assert lattice.D in [2], f"Lattice dimension of {lattice.D} not supported"
        if lattice.D == 2:
            eq = Field2D(
                f"{self.label}_eq",
                self.u.shape[0],
                self.u.shape[1],
                Level.Microscopic,
                "lattice units",
                list(range(lattice.N))
            )
            i = Field2D(
                f"{self.label}",
                self.u.shape[0],
                self.u.shape[1],
                Level.Microscopic,
                "lattice units",
                list(range(lattice.N))
            )
        self.i = i
        self.eq = eq
        self.eq.field = self.equilibrium(lattice)
        self.i.field = self.eq.field

    def equilibrium(self, lattice: Lattice):
        # $e_minus_u = \vec{e} - \vec{u}$ | peculiar velocity | shape: (X, Y, lattice.N)
        # $e_minus_u_i$ = the i-th component of $e_minus_u$ | The ith item in final dimension of $e_minus_u$
        # $g^{eq}$ = equilibrium distribution function (Maxwell-Boltzmann distribution) | shape: (X, Y, lattice.N)
        # $g^{eq}_i$ = equilibrium distribution function for the i-th direction | The ith item in final dimension of $g^{eq}$
        # $g^{eq}_i = \frac{\rho e_minus_u^2}{2(2\pi c_s^2)^{D/2} \exp(-\frac{e_minus_u^2}{2 c_s^2})$
        pass

    def lift(self, lattice: Lattice):
        # Density = zeroth moment of f
        # \rho = \sum_i f_{xyi}
        self.rho.field = jnp.sum(self.i.field, axis=-1, keepdims=True)
        # Momentum =  first moment -> transitively u
        # \rho u = \sum_i f_{xyi}e_i
        self.u.field = jnp.einsum("...i,ic->...c", self.i.field, lattice.microscopic_velocities) / self.rho.field