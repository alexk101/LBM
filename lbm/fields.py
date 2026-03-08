"""Definitions for Fields:
Mathematical fields are easily representable as arrays, but lack critical metadata
about what the field represents. The Field class provides a way to attach this metadata
to an array to allow for simpler visualization, interpretation, and debugging.
"""

import jax.numpy as jnp
from .definitions import Level
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

    @classmethod
    def from_array(
        cls,
        name: str,
        array: Array,
        level: Level,
        units: str = "lattice_units",
        component_names: list[str] | None = None,
    ) -> "Field":
        """Build a Field from an existing array for plotting/diagnostics.

        Infers spatial shape from array: scalar fields have shape (...,);
        vector fields have shape (..., D) and D is the last dimension.

        Args:
            name: Human-readable name (e.g. "rho", "velocity").
            array: JAX array; 2D (X, Y) or (X, Y, 1) for scalar, (X, Y, D) for vector.
            level: MACROSCOPIC or MICROSCOPIC.
            units: Units string for colorbar/labels.
            component_names: For vector fields, names per component; default ["u_0", ...].

        Returns:
            Field instance with .field set to the given array.
        """
        nd = array.ndim
        if nd == 2:
            shape = array.shape
            components = 1
            component_names = component_names or ["value"]
        elif nd >= 3 and array.shape[-1] > 1:
            shape = array.shape[:-1]
            components = array.shape[-1]
            component_names = component_names or [f"u_{i}" for i in range(components)]
        else:
            # (X, Y, 1) or (X, Y, Z, 1)
            shape = array.shape[:-1]
            components = 1
            component_names = component_names or ["value"]

        self = cls(
            name=name,
            level=level,
            units=units,
            component_names=component_names,
            shape=shape,
        )
        self.field = array
        return self

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
