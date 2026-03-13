from abc import ABC, abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jax import Array


class Lattice(eqx.Module, ABC):
    """Base lattice for LBM. Use D2Q9 or D3Q19; do not mix with @dataclass."""

    N: int
    shifts: Array | None = None

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
        """Lattice spatial dimensions."""
        return self.velocities.shape[1]

    @property
    def is_shifted(self) -> bool:
        """True when a nonzero Galilean shift is active."""
        return self.shifts is not None

    @property
    def e(self) -> Array:
        """Effective (shifted) velocity vectors used in all physics.

        When Galilean shifts are active, ``e = velocities + shifts``.
        Otherwise falls back to the integer ``velocities``.
        """
        if self.shifts is not None:
            return self.velocities + self.shifts[None, :]
        return self.velocities

    @property
    def c_s_sq(self) -> float:
        """Speed of sound squared (always 1/3 for standard lattices)."""
        return 1.0 / 3.0

    @property
    def expanded_weights(self) -> Array:
        """Per-direction weights, shape (N,). Convenience for weights[indices]."""
        return self.weights[self.indices]

    def levermore_weights(self, T: Array) -> Array:
        """Temperature-dependent Levermore factorized weights.

        For a direction with k nonzero velocity components in D dimensions:
            w_i(T) = (1 - T)^(D - k) * (T / 2)^k

        At the reference temperature T = 1/3 this recovers the standard weights
        (for lattices where the standard weights are the Levermore weights, i.e.
        D2Q9, D3Q27; D3Q19 sums to < 1 because corner directions are absent).

        Args:
            T: Temperature field, any shape (...). Broadcast over spatial dims.

        Returns:
            Weights with shape (..., N).
        """
        nonzero_per_dir = jnp.sum(jnp.abs(self.velocities) > 0.5, axis=-1)  # (N,)
        D = self.D
        return (1.0 - T[..., None]) ** (D - nonzero_per_dir) * (T[..., None] / 2.0) ** nonzero_per_dir

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
    """D3Q19 lattice: 3 dimensions, 19 velocity directions.

    The most common LBM lattice for 3D simulations. Provides good isotropy
    properties and supports Mach numbers up to ~0.15 with acceptable errors.

    Velocity ordering:
        0: rest (0,0,0)
        1-6: face-connected cardinal directions (weight 1/18)
        7-18: edge-connected face diagonals in xy, xz, yz planes (weight 1/36)
    """

    N: int = 19

    @property
    def weights(self) -> Array:
        """Weights for D3Q19.

        Three unique weights corresponding to direction classes:
            - Class 0: Rest particle w_0 = 1/3
            - Class 1: Face-connected (cardinal) w_{1-6} = 1/18 each
            - Class 2: Edge-connected (diagonals) w_{7-18} = 1/36 each

        Returns:
            Array of shape (3,) containing the unique weight values.
        """
        return jnp.array([1.0 / 3.0, 1.0 / 18.0, 1.0 / 36.0])

    @property
    def indices(self) -> Array:
        """Indices to map weights to lattice points"""
        return jnp.array([0, *([1] * 6), *([2] * 12)])

    @property
    def velocities(self) -> Array:
        """Lattice velocity vectors for all 19 directions."""
        return jnp.array(
            [
                # Rest particle (center) — weight = 1/3
                (0.0, 0.0, 0.0),
                # Face-connected cardinal directions — weight = 1/18 each
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, 1.0),
                # Edge-connected: xy-plane diagonals — weight = 1/36 each
                (-1.0, -1.0, 0.0),
                (-1.0, 1.0, 0.0),
                (1.0, -1.0, 0.0),
                (1.0, 1.0, 0.0),
                # Edge-connected: xz-plane diagonals
                (-1.0, 0.0, -1.0),
                (-1.0, 0.0, 1.0),
                (1.0, 0.0, -1.0),
                (1.0, 0.0, 1.0),
                # Edge-connected: yz-plane diagonals
                (0.0, -1.0, -1.0),
                (0.0, -1.0, 1.0),
                (0.0, 1.0, -1.0),
                (0.0, 1.0, 1.0),
            ]
        )

    @property
    def opposite_indices(self) -> Array:
        """Index j such that velocities[j] == -velocities[i].

        Pairs: 0↔0, 1↔2, 3↔4, 5↔6, 7↔10, 8↔9, 11↔14, 12↔13, 15↔18, 16↔17.
        """
        return jnp.array(
            [0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15]
        )

    @property
    def mirror_indices_per_axis(self) -> Array:
        """Reflect only the given axis component. Shape (3, 19)."""
        # Axis 0 (flip x): 1↔2, 7↔9, 8↔10, 11↔13, 12↔14; y/z dirs unchanged
        mirror_0 = jnp.array(
            [0, 2, 1, 3, 4, 5, 6, 9, 10, 7, 8, 13, 14, 11, 12, 15, 16, 17, 18]
        )
        # Axis 1 (flip y): 3↔4, 7↔8, 9↔10, 15↔17, 16↔18; x/z dirs unchanged
        mirror_1 = jnp.array(
            [0, 1, 2, 4, 3, 5, 6, 8, 7, 10, 9, 11, 12, 13, 14, 17, 18, 15, 16]
        )
        # Axis 2 (flip z): 5↔6, 11↔12, 13↔14, 15↔16, 17↔18; x/y dirs unchanged
        mirror_2 = jnp.array(
            [0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 12, 11, 14, 13, 16, 15, 18, 17]
        )
        return jnp.stack([mirror_0, mirror_1, mirror_2])


class D3Q27(Lattice):
    """D3Q27 lattice: 3 dimensions, 27 velocity directions.

    The full 3D lattice with rest, face, edge, and corner neighbors.
    Provides exact isotropy up to fourth-order tensors, which is needed
    for accurate thermal and compressible LBM.

    Velocity ordering:
        0:     rest (0,0,0)
        1-6:   face-connected cardinal (weight 2/27)
        7-18:  edge-connected diagonals (weight 1/54)
        19-26: corner-connected diagonals (weight 1/216)
    """

    N: int = 27

    @property
    def weights(self) -> Array:
        return jnp.array([8.0 / 27.0, 2.0 / 27.0, 1.0 / 54.0, 1.0 / 216.0])

    @property
    def indices(self) -> Array:
        return jnp.array([0, *([1] * 6), *([2] * 12), *([3] * 8)])

    @property
    def velocities(self) -> Array:
        return jnp.array([
            # Rest
            (0, 0, 0),
            # Face neighbors (6)
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
            # Edge neighbors (12)
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
            (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
            # Corner neighbors (8)
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
            (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1),
        ], dtype=jnp.float32)

    @property
    def opposite_indices(self) -> Array:
        v = self.velocities
        N = self.N
        opp = jnp.zeros(N, dtype=jnp.int32)
        for i in range(N):
            for j in range(N):
                if jnp.allclose(v[i], -v[j]):
                    opp = opp.at[i].set(j)
                    break
        return opp

    @property
    def mirror_indices_per_axis(self) -> Array:
        v = self.velocities
        N = self.N
        mirrors = []
        for axis in range(3):
            mirror = jnp.zeros(N, dtype=jnp.int32)
            for i in range(N):
                target = v[i].at[axis].set(-v[i][axis])
                for j in range(N):
                    if jnp.allclose(v[j], target):
                        mirror = mirror.at[i].set(j)
                        break
            mirrors.append(mirror)
        return jnp.stack(mirrors)
