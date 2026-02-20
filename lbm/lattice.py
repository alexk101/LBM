import jax.numpy as jnp
from jax import Array
import matplotlib.pyplot as plt
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Lattice(ABC):
    N: int
    weights: Array
    indices: Array
    microscopic_velocities: Array

    def __post_init__(self):
        self.D = self.microscopic_velocities.shape[1]
        self.c_squared = self.speed_of_sound_squared()
    
    def speed_of_sound_squared(self):
        squared_speeds = jnp.sum(self.microscopic_velocities ** 2, axis=1)  # (N,)
        return (1 / self.D) * jnp.sum(self.weights[self.indices] * squared_speeds)

    @abstractmethod
    def plot(self, ax: plt.Axes):
        pass

class D2Q9(Lattice):
    def __init__(self):
        super().__init__(
            N=9,
            weights=jnp.array([4/9, 1/9, 1/36]),
            microscopic_velocities=jnp.array([
                [ 0,   0],
                [ 1,   0],
                [ 0,   1],
                [-1,   0],
                [ 0,  -1],
                [ 1,   1],
                [-1,   1],
                [-1,  -1],
                [ 1,  -1],
            ]),
            indices=jnp.array([0, 1, 1, 1, 1, 2, 2, 2, 2])
        )

    def plot(self, ax: plt.Axes):
        # Plot the D2Q9 lattice nodes and velocity directions
        origin = self.microscopic_velocities[0]  # center (0,0)
        colors = ['red', 'green', 'blue']
        # Plot grid points
        ax.scatter(self.microscopic_velocities[:, 0], self.microscopic_velocities[:, 1], color='black', s=40, zorder=2)
        # Draw the velocity vectors from origin

        for i, vec in enumerate(self.microscopic_velocities):
            if i == 0:
                # Origin node, skip vector
                continue
            if self.indices[i] == 1:
                color = colors[1]  # primary (axis-aligned)
                width = 0.035
            elif self.indices[i] == 2:
                color = colors[2]  # diagonals
                width = 0.022
            else:
                color = colors[0]  # center
                width = 0.045
            ax.arrow(
                origin[0], origin[1],
                vec[0], vec[1],
                head_width=0.15, length_includes_head=True, color=color,
                linewidth=2, zorder=3, alpha=0.8
            )
        ax.set_aspect('equal')
        ax.set_title("D2Q9 Lattice")
        ax.set_xlabel("$e_x$")
        ax.set_ylabel("$e_y$")
        ax.grid(True, alpha=0.2)
