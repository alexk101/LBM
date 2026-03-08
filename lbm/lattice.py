from abc import ABC, abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jax import Array


class Lattice(eqx.Module, ABC):
    """Base lattice for LBM. Use D2Q9 or D3Q19; do not mix with @dataclass."""

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

    @property
    def D(self) -> int:
        """Lattice spatial dimensions"""
        return self.velocities.shape[1]

    @property
    def c_s_sq(self) -> Array:
        """Speed of sound squared in lattice units (should always be 1/3)"""
        squared_speeds = jnp.sum(self.velocities**2, axis=1)  # (N,)
        return jnp.sum(self.weights[self.indices] * squared_speeds) / self.D

    @property
    @abstractmethod
    def opposite_indices(self) -> Array:
        """Index j for each i such that velocities[j] == -velocities[i]. Used for bounce-back BCs."""
        ...

    @property
    @abstractmethod
    def mirror_indices_per_axis(self) -> Array:
        """For each axis dim, mirror[dim][i] = j with vel[j] = vel[i] but vel[j][dim] = -vel[i][dim]. Used for free-slip BCs. Shape (D, N)."""
        ...


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

    @property
    def opposite_indices(self) -> Array:
        """Index j for each i such that velocities[j] == -velocities[i]. D2Q9: 0↔0, 1↔3, 2↔4, 5↔7, 6↔8."""
        return jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

    @property
    def mirror_indices_per_axis(self) -> Array:
        """Reflect only the given axis component. D2Q9: axis 0 [0,3,2,1,4,6,5,8,7], axis 1 [0,1,4,3,2,8,7,6,5]."""
        return jnp.array(
            [[0, 3, 2, 1, 4, 6, 5, 8, 7], [0, 1, 4, 3, 2, 8, 7, 6, 5]]
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

    @property
    def opposite_indices(self) -> Array:
        """Index j for each i such that velocities[j] == -velocities[i]. D3Q19: 1↔2, 3↔4, 5↔6; 7-9↔16-18; 10-12↔13-15."""
        return jnp.array(
            [0, 2, 1, 4, 3, 6, 5, 16, 17, 18, 13, 14, 15, 10, 11, 12, 7, 8, 9]
        )

    @property
    def mirror_indices_per_axis(self) -> Array:
        """Reflect only the given axis. D3Q19: axis 0 flips 1↔2, 5↔6, 7↔8, 9↔10, 11↔12, 13↔14, 15↔16, 17↔18; axis 1 and 2 analogous."""
        # vel order: 0 rest; 1(1,0,0) 2(-1,0,0); 3(0,1,0) 4(0,-1,0); 5(0,0,1) 6(0,0,-1); 7..18 diagonals
        # axis 0: 1↔2, 7↔8, 9↔10, 11↔12, 13↔14, 15↔16, 17↔18 3↔3)
        # So axis 0: 0->0, 1->2, 2->1, 3->3, 4->4, 5->5, 6->6, then diagonals: (1,1,0)↔(-1,1,0) so 7↔8, (1,-1,0)↔(-1,-1,0) 9↔10, etc.
        # D3Q19 secondaries from code: (1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0) each x3. So 7,8,9,10 = (1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0) then repeat. So 7↔8, 9↔10, 11↔12, 13↔14, 15↔16, 17↔18.
        mirror_0 = jnp.array([0, 2, 1, 3, 4, 5, 6, 16, 17, 18, 13, 14, 15, 10, 11, 12, 7, 8, 9])
        mirror_1 = jnp.array([0, 1, 2, 4, 3, 5, 6, 10, 11, 12, 7, 8, 9, 16, 17, 18, 13, 14, 15])
        mirror_2 = jnp.array([0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        return jnp.stack([mirror_0, mirror_1, mirror_2])
