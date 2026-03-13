"""HDF5 data pipeline: writing simulation snapshots and loading them with grain.

Layout::

    simulation.h5
    ├── attrs: lattice_type, dt, D, N, spatial_shape, fields, num_snapshots
    ├── times          (S,)           float64
    ├── steps          (S,)           int32
    ├── rho            (S, X, Y, 1)   float32      [always present]
    ├── u              (S, X, Y, D)   float32      [always present]
    ├── T              (S, X, Y, 1)   float32      [if thermal]
    ├── F              (S, X, Y, N)   float32      [if save_distributions]
    └── G              (S, X, Y, N)   float32      [if thermal + save_distributions]

Writing::

    with SimulationWriter("out.h5", lattice=lat, dt=0.1,
                          spatial_shape=(100,100)) as w:
        for step in range(steps):
            state = solver.step(state)
            if step % save_every == 0:
                w.write(state, step=step, time=step*dt)

Reading (grain DataLoader)::

    source = HDF5DataSource("out.h5")
    loader = grain.load(source, batch_size=32, shuffle=True, seed=42)
    for batch in loader:
        rho = jnp.asarray(batch["rho"])  # (B, X, Y, 1)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import h5py
import jax.numpy as jnp
import numpy as np
from jax import Array

from .definitions import LBMState
from .lattice import Lattice


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class SimulationWriter:
    """Incrementally write LBM simulation snapshots to an HDF5 file.

    Use as a context manager or call :meth:`close` explicitly.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        lattice: Lattice,
        dt: float,
        spatial_shape: tuple[int, ...],
        field_keys: Sequence[str] = ("rho", "u"),
        distribution_keys: Sequence[str] = (),
        metadata: dict | None = None,
    ) -> None:
        self.path = Path(path)
        self.field_keys = list(field_keys)
        self.distribution_keys = list(distribution_keys)
        self._all_keys = self.field_keys + self.distribution_keys

        D = lattice.D
        N = lattice.N

        field_shapes: dict[str, tuple[int, ...]] = {
            "rho": (*spatial_shape, 1),
            "u": (*spatial_shape, D),
            "T": (*spatial_shape, 1),
        }
        for dk in distribution_keys:
            field_shapes[dk] = (*spatial_shape, N)

        self._f = h5py.File(self.path, "w")
        self._f.attrs["lattice_type"] = type(lattice).__name__
        self._f.attrs["dt"] = dt
        self._f.attrs["D"] = D
        self._f.attrs["N"] = N
        self._f.attrs["spatial_shape"] = list(spatial_shape)
        self._f.attrs["fields"] = self.field_keys
        self._f.attrs["distributions"] = self.distribution_keys
        if metadata:
            for k, v in metadata.items():
                self._f.attrs[k] = v

        self._f.create_dataset("times", shape=(0,), maxshape=(None,), dtype=np.float64)
        self._f.create_dataset("steps", shape=(0,), maxshape=(None,), dtype=np.int32)
        for key in self._all_keys:
            if key not in field_shapes:
                continue
            ds_shape = field_shapes[key]
            self._f.create_dataset(
                key,
                shape=(0, *ds_shape),
                maxshape=(None, *ds_shape),
                dtype=np.float32,
                chunks=(1, *ds_shape),
            )
        self._count = 0

    def write(self, state: LBMState, *, step: int, time: float) -> None:
        """Append a single snapshot."""
        idx = self._count
        self._f["times"].resize(idx + 1, axis=0)
        self._f["times"][idx] = time
        self._f["steps"].resize(idx + 1, axis=0)
        self._f["steps"][idx] = step

        for key in self._all_keys:
            if key not in state or key not in self._f:
                continue
            ds = self._f[key]
            ds.resize(idx + 1, axis=0)
            ds[idx] = np.asarray(state[key])

        self._count += 1

    def close(self) -> None:
        if self._f:
            self._f.attrs["num_snapshots"] = self._count
            self._f.close()
            self._f = None  # type: ignore[assignment]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        if hasattr(self, "_f") and self._f:
            self.close()


# ---------------------------------------------------------------------------
# Grain RandomAccessDataSource backed by HDF5
# ---------------------------------------------------------------------------


class HDF5DataSource:
    """Grain-compatible random-access data source backed by an HDF5 file.

    Satisfies the ``grain.python.RandomAccessDataSource`` protocol
    (``__len__``, ``__getitem__``, ``__repr__``). Each record is a dict of
    numpy arrays for one simulation snapshot.

    Works with ``grain.load()`` and ``grain.DataLoader`` for batching,
    shuffling, and multi-worker loading.

    Example::

        source = HDF5DataSource("simulation.h5")
        loader = grain.load(source, batch_size=32, shuffle=True, seed=0)
        for batch in loader:
            rho = jnp.asarray(batch["rho"])  # (B, X, Y, 1)
    """

    def __init__(
        self,
        path: str | Path,
        field_keys: Sequence[str] | None = None,
        distribution_keys: Sequence[str] | None = None,
    ) -> None:
        self._path = Path(path)
        self._f = h5py.File(self._path, "r")
        self._field_keys = (
            list(field_keys)
            if field_keys is not None
            else list(self._f.attrs.get("fields", []))
        )
        self._dist_keys = (
            list(distribution_keys)
            if distribution_keys is not None
            else list(self._f.attrs.get("distributions", []))
        )
        self._all_keys = self._field_keys + self._dist_keys
        self._len = int(self._f["times"].shape[0])

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if index < 0:
            index = self._len + index
        record: dict[str, np.ndarray] = {}
        for key in self._all_keys:
            if key in self._f:
                record[key] = self._f[key][index]
        record["time"] = self._f["times"][index]
        record["step"] = self._f["steps"][index]
        return record

    def __repr__(self) -> str:
        return f"HDF5DataSource(path={self._path!r}, len={self._len})"

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._f.attrs)

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None  # type: ignore[assignment]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        if hasattr(self, "_f") and self._f:
            self.close()


def load_snapshot(path: str | Path, index: int) -> dict[str, Array]:
    """One-shot helper: load a single snapshot as JAX arrays."""
    with HDF5DataSource(path) as src:
        record = src[index]
    return {k: jnp.asarray(v) for k, v in record.items()}
