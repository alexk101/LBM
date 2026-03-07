# ============================================================================
# Plotting Utilities (Separate from Field class)
# ============================================================================

import jax.numpy as jnp
import matplotlib.pyplot as plt
from fields import Field
from lattice import Lattice
from matplotlib.axes import Axes


def plot_field(field: Field, ax=None | Axes, cmap="viridis", **kwargs):
    """Plot a field on the given axes.

    Automatically handles 2D and 3D fields with appropriate visualization.
    For vector fields, plots magnitude by default but can show components.

    Args:
        field: Field object to plot (contains metadata for labels)
        ax: Matplotlib axes to plot on. If None, creates new figure/axes.
        cmap: Colormap name or colormap object.
        **kwargs: Additional kwargs passed to matplotlib plotting functions.

    Returns:
        The matplotlib Axes and Colorbar (if created).

    Example Usage:
        fig, ax = plt.subplots()
        plot_field(rho, ax)  # Automatically labels based on field metadata

        # Show individual components of vector field
        for i, name in enumerate(u.component_names):
            plot_field_component(u, component=i, ax=ax, title=f"{name} velocity")
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Access the raw array from field metadata
    data = field.field

    # Handle 2D scalar fields (most common case)
    if field.dimensionality == 2 and not field.is_vector:
        im = ax.imshow(data, origin="lower", cmap=cmap, **kwargs)

    # Handle 2D vector fields - plot magnitude by default
    elif field.dimensionality == 2 and field.is_vector:
        magnitude = jnp.linalg.norm(data, axis=-1)
        im = ax.imshow(magnitude, origin="lower", cmap=cmap, **kwargs)

    # Handle 3D scalar fields - show mid-plane slice
    elif field.dimensionality == 3 and not field.is_vector:
        z_mid = data.shape[0] // 2
        im = ax.imshow(data[z_mid], origin="lower", cmap=cmap, **kwargs)

    # Handle 3D vector fields - show magnitude of mid-plane slice
    elif field.dimensionality == 3 and field.is_vector:
        z_mid = data.shape[0] // 2
        magnitude = jnp.linalg.norm(data[z_mid], axis=-1)
        im = ax.imshow(magnitude, origin="lower", cmap=cmap, **kwargs)

    # Add title and colorbar based on field metadata
    ax.set_title(f"{field.name} ({field.level.value})")

    if not kwargs.get("cbar", False):  # Allow caller to disable colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(field.units)

    return ax, im


def plot_field_component(field: Field, component: int | str = 0, ax=None, **kwargs):
    """Plot a specific component of a vector field.

    Useful for visualizing individual components (u_x, u_y, etc.) separately.

    Args:
        field: Field object to plot from
        component: Component index or name to plot. Default is 0.
        ax: Matplotlib axes to plot on. If None, creates new figure/axes.
        **kwargs: Additional kwargs passed to matplotlib plotting functions.

    Returns:
        The matplotlib Axes and Image object.

    Example Usage:
        # Plot x-component of velocity with custom title
        plot_field_component(u, component="x", ax=ax, title="U_x")
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Get the component index by name or use directly
    if isinstance(component, str):
        try:
            idx = field.component_names.index(component)
        except ValueError:
            raise KeyError(f"Component '{component}' not found in {field.name}")
    else:
        idx = component

    # Access the specific component from the array
    data = field.field[..., idx] if field.is_vector else field.field

    im = ax.imshow(data, origin="lower", **kwargs)

    title = (
        f"{field.name}.{field.component_names[idx]}" if field.is_vector else field.name
    )
    ax.set_title(f"{title} ({field.level.value})")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(field.units)

    return ax, im


def plot_fields_grid(fields: list[Field], ncols: int = 3, figsize=(15, 5), **kwargs):
    """Plot multiple fields in a grid layout.

    Useful for comparing different macroscopic quantities or time steps.

    Args:
        fields: List of Field objects to plot (each gets one subplot)
        ncols: Number of columns in the grid
        figsize: Figure size tuple (width, height)
        **kwargs: Additional kwargs passed to each plot function

    Returns:
        Matplotlib figure and axes array.

    Example Usage:
        fig, axes = plot_fields_grid([rho, u_magnitude], ncols=2)
        fig.suptitle("LBM Simulation at t=10")
    """
    nfields = len(fields)
    nrows = (nfields + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if nfields == 1:
        # Single field case
        plot_field(fields[0], ax=axes, **kwargs)
    else:
        for i, (field, ax) in enumerate(zip(fields, axes.ravel())):
            plot_field(field, ax=ax, cbar=False, **kwargs)

        # Remove unused subplots if nfields < nrows * ncols
        for ax in axes.ravel()[nfields:]:
            fig.delaxes(ax)

    return fig, axes


# Visualization helper
def plot_lattice(lattice: Lattice, ax: Axes):
    """Plot the velocity vectors of a lattice on a 2D plane.

    Args:
        lattice: Lattice configuration object.
        ax: Matplotlib axes to plot on.

    Notes:
        For 3D lattices (D3Q19, D3Q27), only the xy-plane is shown.
        The colors indicate different weight classes (via indices).
    """

    # Get velocities and their corresponding weights
    velocities = lattice.velocities[:, :2]  # Only x,y components

    # Plot grid points
    ax.scatter(
        velocities[:, 0],
        velocities[:, 1],
        s=60,
        c="black",
        zorder=2,
        label="Directions",
    )

    # Draw velocity vectors from origin (colored by weight class)
    colors = {
        i: plt.cm.viridis(i / len(lattice.weights)) for i in range(len(lattice.weights))
    }

    for i, vec in enumerate(velocities):
        if i == 0 and jnp.all(vec == 0):  # Skip rest particle (origin)
            continue

        color = colors[lattice.indices[i]]

        ax.arrow(
            0,
            0,
            vec[0],
            vec[1],
            head_width=0.15,
            length_includes_head=True,
            color=color,
            linewidth=2,
            zorder=3,
            alpha=0.8,
        )

    ax.set_aspect("equal")
    ax.set_title(f"{lattice.__class__.__name__} (N={lattice.N})")
    ax.set_xlabel("$e_x$")
    ax.set_ylabel("$e_y$")
    ax.grid(True, alpha=0.2)
