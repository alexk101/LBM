"""Definitions for Fields:
Mathematical fields are easily representable as arrays, but lack critical metadata
about what the field represents. The Field class provides a way to attach this metadata
to an array to allow for simpler visualization, interpretation, and debugging.
"""

import jax.numpy as jnp
from definitions import Level
from jax import Array


class Field:
    """Container for JAX arrays with metadata for visualization and diagnostics.

    A single class that works for both 2D and 3D fields (and even 1D).
    The dimensionality is inferred from the shape, not encoded in the class hierarchy.

    Attributes:
        name: Human-readable identifier (e.g., "rho", "u_x")
        level: Physical abstraction level (macroscopic/microscopic)
        units: SI or lattice units for the field
        component_names: Names of vector/tensor components if applicable
        shape: Shape of the underlying array (excluding component dimension)
        field: The actual JAX array backing this field
    """

    def __init__(
        self,
        name: str,
        level: Level,
        units: str,
        component_names: list[str],
        shape: tuple[int, ...],
    ):

        self.name = name
        self.level = level
        self.units = units
        self.component_names = component_names

        # Infer dimensions from shape (X, Y) or (X, Y, Z)
        if len(shape) == 2:
            self.X, self.Y = shape[0], shape[1]
            self.Z = None
        elif len(shape) == 3:
            self.X, self.Y, self.Z = shape[0], shape[1], shape[2]
        else:
            raise ValueError(f"Unsupported field dimensions: {len(shape)}D")

        # Number of components (vector/tensor vs scalar)
        self.components = len(component_names) if component_names else 1

        # Initialize the field array with zeros
        full_shape = shape + (self.components,) if self.components > 1 else shape
        self.field: Array = jnp.zeros(full_shape)

    @property
    def is_vector(self) -> bool:
        """Whether this field has multiple components."""
        return self.components > 1

    @property
    def dimensionality(self) -> int:
        """Return the spatial dimension (2 or 3)."""
        if self.Z is not None and self.Z > 0:
            return 3
        elif self.Y is not None and self.Y > 0:
            return 2
        else:
            raise ValueError("Cannot determine dimensionality")

    def __repr__(self):
        dim_str = f"{self.dimensionality}D" if self.Z else "2D"
        component_str = (
            f" ({', '.join(self.component_names)})" if self.is_vector else ""
        )
        return f"<Field {self.name}: Level={self.level.value}, Units={self.units}, {dim_str}{component_str}>"
