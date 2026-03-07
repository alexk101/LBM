from definitions import Level
from fields import Field


def create_macroscopic_field(
    name: str,
    X: int,
    Y: int,
    units: str = "",
    component_names: list[str] | None = None,
) -> Field:
    """Create a macroscopic field with standard configuration."""
    if not all([X, Y]):
        raise ValueError("Must specify X and Y dimensions")

    return Field(
        name=name,
        shape=(X, Y),
        level=Level.MACROSCOPIC,
        units=units,
        component_names=component_names or [name],
    )


def create_microscopic_field(
    name: str,
    N: int,  # Number of distribution directions
    X: int,
    Y: int,
    units: str = "lattice_units",
) -> Field:
    """Create a microscopic (distribution function) field."""
    return Field(
        name=name,
        shape=(X, Y),
        level=Level.MICROSCOPIC,
        units=units,
        component_names=[f"f_{i}" for i in range(N)],
    )
