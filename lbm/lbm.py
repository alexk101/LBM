import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from dataclasses import dataclass
from abc import ABC, abstractmethod

from distributions import Distribution, Particle
from lattice import Lattice, D2Q9
from fields import Field2D
from definitions import *

@dataclass
class LBM(ABC):
    lattice: Lattice
    dt: float
    t_max: int
    
    D: int
    distributions: list[Distribution]
    tau: float

    t: float = 0.0
    steps: int | None = None

    def __post_init__(self):
        if self.steps is None:
            self.steps = int(self.t_max // self.dt)
        for dist in self.distributions:
            dist.init_microscopic(self.lattice)

    @abstractmethod
    def collision(self):
        pass
    @abstractmethod
    def streaming(self):
        pass
    @abstractmethod
    def step(self):
        pass

    def run(self):
        for i in range(self.steps):
            self.step()
            self.t += self.dt
        return self.distributions

    def plot(self, ax: plt.Axes | None = None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(self.u.field, origin='lower')
        ax.set_title(self.u.name)
        return ax


class LBM2D(LBM):
    def __init__(self, u: Field2D, rho: Field2D, tau: float, dt: float, t_max: int):
        super().__init__(
            D=2,
            lattice=D2Q9(),
            tau=tau,
            dt=dt,
            t_max=t_max,
            distributions=[Particle(name="Particle", label="F", i=None, eq=None, u=u, rho=rho, tau=tau)]
        )


    def collision(self):
        dist = self.distributions[0]
        dist.eq.field = dist.equilibrium(self.lattice)
        return (dist.eq.field - dist.i.field) / dist.tau

    def streaming(self, f_star):
        def stream_one(f_slice, e_i):
            return jnp.roll(jnp.roll(f_slice, -e_i[0], axis=0), -e_i[1], axis=1)

        rolled = jax.vmap(stream_one, in_axes=(2, 0))(
            f_star, self.lattice.microscopic_velocities
        )
        return jnp.transpose(rolled, (1, 2, 0))

    def step(self):
        dist = self.distributions[0]
        f_star = dist.i.field + self.collision()
        dist.i.field = self.streaming(f_star)
        dist.lift(self.lattice)


def plot_fields(u: Field2D, rho: Field2D, t: float):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    u.plot(fig, axes[0])       # 3 axes: norm, u.x, u.y
    rho.plot(fig, axes[1, :2]) # 2 axes: norm, rho (skip 3rd subplot)
    axes[1][2].remove()
    fig.suptitle(f"Time: {t:.1f} seconds")
    fig.savefig(f"plots/fields_{t:.1f}.png")


def main():
    X = 100
    Y = 100

    distance = X // 10

    # Density field
    rho = Field2D(name="rho", X=X, Y=Y, level=Level.Macroscopic, units="kg/m^3", component_names=["rho"])
    rho.field = jnp.full_like(rho.field, 1)
    rho.field = rho.field.at[(X//2)-(distance//2), Y//2, 0].set(2).at[(X//2)+(distance//2), Y//2, 0].set(2)
    # Velocity field
    u = Field2D(name="u", X=X, Y=Y, level=Level.Macroscopic, units="m/s", component_names=["x", "y"])

    plot_fields(u, rho, 0)

    print(f"Before")
    print(f"rho: {rho.field.sum()}")
    print(f"u: {u.field.sum()}")
    tau = 0.5
    lbm = LBM2D(u, rho, tau, 0.1, 10)

    for i in range(10):
        lbm.step()
        lbm.t += lbm.dt
        plot_fields(u, rho, lbm.t)

    # dist: Particle
    # dist = lbm.run()[0]
    # print(f"After")
    # print(f"rho: {rho.field.sum()}")
    # print(f"u: {u.field.sum()}")


if __name__ == "__main__":
    main()