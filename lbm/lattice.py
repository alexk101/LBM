from abc import abstractmethod
from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class Lattice(eqx.Module):
    N: int

    @property
    @abstractmethod
    def weights(self) -> Array: ...

    @property
    @abstractmethod
    def indices(self) -> Array: ...

    @property
    @abstractmethod
    def velocities(self) -> Array: ...

    @eqx.field(static=True)
    @property
    def D(self) -> int:
        """Lattice spatial dimensions"""
        return self.velocities.shape[1]

    @eqx.field(static=True)
    @property
    def c_s_sq(self) -> float:
        """Speed of sound squared in lattice units (should always be 1/3)"""
        squared_speeds = jnp.sum(self.velocities**2, axis=1)  # (N,)
        return (jnp.sum(self.weights[self.indices] * squared_speeds)) / self.D


class D2Q9(Lattice):
    """D2Q9 lattice: 2 dimensions, 9 velocity directions.

    The most common LBM lattice for 2D simulations. Provides good isotropy
    properties and supports Mach numbers up to ~0.15 with acceptable errors.
    """

    N: int = 9

    @property
    def weights(self) -> Array:
        """Weights for D2Q9.

        Three unique weights corresponding to direction classes:
            - Class 0 (index): Rest particle w_0 = 4/9
            - Class 1 (indices): Cardinal directions w_{1-4} = 1/9 each
            - Class 2 (indices): Diagonal directions w_{5-8} = 1/36 each

        Returns:
            Array of shape (3,) containing the unique weight values.
        """
        return jnp.array([4.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0])

    @property
    def indices(self) -> Array:
        """Indices to map weights to lattice points"""
        return jnp.array([0, 1, 1, 1, 1, 2, 2, 2, 2])

    @property
    def velocities(self) -> Array:
        """Lattice velocity vectors for all directions."""
        return jnp.array(
            [
                # Rest particle (center)
                (0.0, 0.0),
                # Cardinal directions (primary, weight = 1/9)
                (1.0, 0.0),
                (0.0, 1.0),
                (-1.0, 0.0),
                (0.0, -1.0),
                # Diagonal directions (secondary, weight = 1/36)
                (1.0, 1.0),
                (-1.0, 1.0),
                (-1.0, -1.0),
                (1.0, -1.0),
            ]
        )


class D3Q19(Lattice):
    """D3Q29 lattice: 3 dimensions, 19 velocity directions.

    The most common LBM lattice for 3D simulations. Provides good isotropy
    properties and supports Mach numbers up to ~0.15 with acceptable errors.
    """

    N: int = 19

    @property
    def weights(self) -> Array:
        """Weights for D3Q19.

        Three unique weights corresponding to direction classes:
            - Class 0 (index): Rest particle w_0 = 20/72
            - Class 1 (indices): Primary directions w_{1-6} = 8/72 each
            - Class 2 (indices): Secondary directions w_{7-18} = 1/72 each

        Returns:
            Array of shape (3,) containing the unique weight values.
        """
        return jnp.array([20.0 / 72.0, 8.0 / 72.0, 1.0 / 72.0])

    @property
    def indices(self) -> Array:
        """Indices to map weights to lattice points"""
        return jnp.array([0, *([1] * 6), *[2] * 12])

    @property
    def velocities(self) -> Array:
        """Lattice velocity vectors for all directions."""
        return jnp.array(
            [
                # Rest particle (center) - weight = 20/72
                (0.0, 0.0, 0.0),
                # Primary directions (cardinal axes) - weight = 8/72 each
                *(
                    e
                    for e in [(d, 0, 0) for d in (-1, 1)]
                    + [(0, d, 0) for d in (-1, 1)]
                    + [(0, 0, d) for d in (-1, 1)]
                ),
                # Secondary directions (face diagonals) - weight = 1/72 each
                *(
                    e
                    for e in [
                        (d1, d2, 0)
                        for d1 in (-1, 1)
                        for d2 in (-1, 1)
                        for _ in range(3)  # xy, yz, xz planes
                    ]
                ),
            ]
        )
