from dataclasses import dataclass
from abc import ABC, abstractmethod
from definitions import *
from jax import Array
import jax.numpy as jnp
import matplotlib.pyplot as plt

@dataclass
class Field(ABC):
    shape: tuple
    name: str
    level: Level
    units: str
    component_names: list[str]
    components: int | None = None
    field: Array | None = None

    def __post_init__(self):
        self.components = len(self.component_names)
        self.field = jnp.zeros(self.shape + (self.components,))

    @abstractmethod
    def plot(self, fig: plt.Figure, ax: plt.Axes):
        pass


class Field2D(Field):
    def __init__(self, name: str, X: int, Y: int, level: Level, units: str, component_names: list[str]):
        super().__init__(
            shape=(X, Y),
            name=name,
            units=units,
            level=level,
            component_names=component_names
        )
        X: int = X
        Y: int = Y


    def plot(self, fig: plt.Figure, axes: plt.Axes):
        assert len(axes) >= self.components+1, f"Number of axes must be greater than or equal to number of components+1 ({len(axes)} vs {self.components+1})"
        image = axes[0].imshow(jnp.linalg.norm(self.field, axis=-1))
        cbar = fig.colorbar(image, ax=axes[0], label=self.units)
        axes[0].set_title(f"Field: {self.name} {self.level.name} Norm")

        for i, (ax, component_name) in enumerate(zip(axes[1:], self.component_names)):
            image = ax.imshow(self.field[..., i])
            cbar = fig.colorbar(image, ax=ax, label=self.units)
            ax.set_title(f"{self.level.name} Field: {self.name}.{component_name} Component")