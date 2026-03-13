"""
Experiment tracking backends implementing a common interface.

Use get_tracker(backend, cfg, logging_cfg, ...) to obtain a tracker instance.
The default backend is "none" (NullTracker).

Available backends:
- "none": NullTracker (no-op, used when tracking is disabled)
- "wandb": Weights & Biases tracking
- "mlflow": MLflow tracking (local or remote)
"""

from __future__ import annotations

import logging
from typing import Optional

from omegaconf import DictConfig

from .base import ExperimentTracker, NullTracker, FigureFormat

log = logging.getLogger(__name__)

__all__ = [
    "ExperimentTracker",
    "NullTracker",
    "FigureFormat",
    "get_tracker",
    "create_tracker",
]


def get_tracker(
    backend: str,
    cfg: DictConfig,
    logging_cfg: DictConfig,
    run_id: Optional[str] = None,
) -> ExperimentTracker:
    """Create and return a tracker instance for the specified backend.

    This is the low-level factory function. For most use cases, prefer
    create_tracker() which handles config extraction automatically.

    Parameters
    ----------
    backend : str
        Backend name: "wandb", "mlflow", or "none".
    cfg : DictConfig
        Full Hydra configuration object.
    logging_cfg : DictConfig
        Logging-specific configuration containing tracker settings.
    run_id : Optional[str]
        For MLflow: resume an existing run with this ID.

    Returns
    -------
    ExperimentTracker
        An initialized tracker instance.

    Raises
    ------
    ValueError
        If backend is not recognized.
    ImportError
        If the backend's library is not installed.
    """
    if backend in (None, "none", ""):
        tracker_cfg = getattr(logging_cfg, "tracker", {})
        interactive = (
            bool(tracker_cfg.get("interactive_plots", False)) if tracker_cfg else False
        )
        return NullTracker(interactive_plots=interactive)

    if backend == "wandb":
        from .wandb import WandbTracker

        return WandbTracker(cfg, logging_cfg)

    if backend == "mlflow":
        from .mlflow import MlflowTracker

        return MlflowTracker(cfg, logging_cfg, run_id=run_id)

    raise ValueError(
        f"Unknown experiment tracking backend: '{backend}'. "
        f"Available backends: 'wandb', 'mlflow', 'none'."
    )


def create_tracker(
    cfg: DictConfig,
    logger: logging.Logger,
    run_id: Optional[str] = None,
) -> ExperimentTracker:
    """Create an experiment tracker from the full configuration.

    This is the high-level factory function that extracts logging config
    from the main config and handles errors gracefully.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration object with a 'logging' section.
    logger : logging.Logger
        Logger for info/warning messages.
    run_id : Optional[str]
        For MLflow: resume an existing run with this ID.

    Returns
    -------
    ExperimentTracker
        An initialized tracker instance. Returns NullTracker if:
        - logging config is missing
        - tracker is disabled
        - backend initialization fails (with warning)
    """
    logging_cfg = getattr(cfg, "logging", None)
    if logging_cfg is None:
        logger.info("No logging configuration found. Experiment tracking disabled.")
        return NullTracker()

    backend = logging_cfg.get("backend", "none")
    tracker_cfg = getattr(logging_cfg, "tracker", None)
    enabled = (
        getattr(tracker_cfg, "enabled", False) if tracker_cfg is not None else False
    )

    if not enabled or backend in (None, "none", ""):
        logger.info(
            "Experiment tracking disabled (backend=none or tracker.enabled=false)."
        )
        interactive = (
            bool(tracker_cfg.get("interactive_plots", False)) if tracker_cfg else False
        )
        return NullTracker(interactive_plots=interactive)

    # Attempt to create the tracker
    try:
        tracker = get_tracker(backend, cfg, logging_cfg, run_id=run_id)
        return tracker
    except ImportError as e:
        logger.warning(
            f"Could not import {backend} library: {e}. "
            f"Install it with 'pip install {backend}' or disable tracking. "
            f"Using NullTracker instead."
        )
        return NullTracker()
    except ValueError as e:
        logger.warning(f"{e}. Using NullTracker instead.")
        return NullTracker()
    except Exception as e:
        # Unexpected error during initialization - log and continue
        logger.warning(
            f"Failed to initialize {backend} tracker: {e}. "
            f"Using NullTracker instead."
        )
        return NullTracker()
