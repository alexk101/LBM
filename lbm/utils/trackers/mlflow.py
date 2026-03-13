"""
MLflow experiment tracking backend.

Provides native support for logging metrics, figures (matplotlib/plotly),
and hyperparameters to MLflow. Supports both local and remote tracking servers.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from omegaconf import DictConfig, OmegaConf

from .base import ExperimentTracker

if TYPE_CHECKING:
    from PIL import Image
    import matplotlib.figure
    import plotly.graph_objects

log = logging.getLogger(__name__)


def _is_matplotlib_figure(obj: Any) -> bool:
    """Check if object is a matplotlib Figure without importing matplotlib."""
    return (
        type(obj).__module__.startswith("matplotlib") and type(obj).__name__ == "Figure"
    )


def _is_plotly_figure(obj: Any) -> bool:
    """Check if object is a plotly Figure without importing plotly."""
    return type(obj).__module__.startswith("plotly") and "Figure" in type(obj).__name__


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dictionary with dot-separated keys."""
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, key))
        else:
            flat[key] = v
    return flat


class MlflowTracker(ExperimentTracker):
    """MLflow implementation of the experiment tracker.

    Supports local (sqlite/file) and remote tracking via tracking_uri.
    When run_id is provided (e.g. from a checkpoint when resuming),
    continues logging to that existing run instead of creating a new one.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration object.
    logging_cfg : DictConfig
        Logging-specific configuration containing tracker settings.
    run_id : Optional[str]
        If provided, resume an existing MLflow run.

    Raises
    ------
    ImportError
        If mlflow is not installed.
    """

    def __init__(
        self,
        cfg: DictConfig,
        logging_cfg: DictConfig,
        run_id: Optional[str] = None,
    ):
        import mlflow  # Raises ImportError if not installed

        tracker_cfg = getattr(logging_cfg, "tracker", {})
        interactive_plots = bool(tracker_cfg.get("interactive_plots", False))
        super().__init__(interactive_plots=interactive_plots)

        mlflow.enable_system_metrics_logging()

        # --- Environment Configuration ---
        # TLS verification for self-signed certs
        if tracker_cfg.get("insecure_tls", False):
            os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"

        # Authentication: username from config, password from environment
        username = tracker_cfg.get("username", None)
        if username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username

        if username and "MLFLOW_TRACKING_PASSWORD" not in os.environ:
            log.warning(
                "MLflow username set in config, but MLFLOW_TRACKING_PASSWORD "
                "not found in environment. Authentication may fail."
            )

        # Tracking URI
        tracking_uri = tracker_cfg.get("tracking_uri", None)
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
            log.info(f"MLflow tracking URI set to: {tracking_uri}")

        # --- Experiment and Run Setup ---
        project = tracker_cfg.get("project", None)
        run_name = tracker_cfg.get("run_name", None)
        run_tag = tracker_cfg.get("run_tag", None)
        resume_run_id = run_id or tracker_cfg.get("run_id", None)

        experiment_name = project or "default"
        mlflow.set_experiment(experiment_name)

        self._mlflow = mlflow
        if resume_run_id is not None:
            self._run = mlflow.start_run(run_id=resume_run_id)
            log.info(f"Resumed MLflow run: run_id={resume_run_id}")
        else:
            self._run = mlflow.start_run(run_name=run_name)

        if run_tag is not None:
            mlflow.set_tag("run_tag", str(run_tag))

        # Log full config as parameters (flattened)
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(config_dict, dict):
            flat_params = _flatten_dict(config_dict)
            self._log_params_safe(flat_params)

        log.info(
            f"Initialized MLflow run: experiment={experiment_name}, name={run_name}, "
            f"tracking_uri={mlflow.get_tracking_uri()}"
        )

    def _log_params_safe(self, params: Dict[str, Any]) -> None:
        """Log parameters, handling MLflow's 500-char limit gracefully."""
        for k, v in params.items():
            str_v = str(v)
            # MLflow has a 500 character limit for param values
            if len(str_v) > 500:
                log.debug(f"Truncating param '{k}' (length {len(str_v)} > 500)")
                str_v = str_v[:497] + "..."
            self._mlflow.log_param(k, str_v)

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        step_metric: Optional[str] = None,
    ) -> None:
        """Log scalar metrics to MLflow.

        MLflow uses a single step timeline; step_metric is ignored.
        Caller should pass step=global_step for both batch- and epoch-level
        metrics so they align on the same timeline.
        """
        for k, v in metrics.items():
            # Skip non-numeric values
            if not isinstance(v, (int, float)):
                continue
            if step is not None:
                self._mlflow.log_metric(k, float(v), step=step)
            else:
                self._mlflow.log_metric(k, float(v))

    def log_figure(
        self,
        figure: Union[
            "matplotlib.figure.Figure",
            "plotly.graph_objects.Figure",
            "Image.Image",
        ],
        key: str,
        step: Optional[int] = None,
        step_metric: Optional[str] = None,
    ) -> None:
        """Log a figure to MLflow.

        - Matplotlib figures: Logged via mlflow.log_figure as PNG
        - Plotly figures: Logged via mlflow.log_figure as HTML (interactive) or PNG (static)
        - PIL Images: Logged via mlflow.log_image

        The artifact filename includes step and timestamp for uniqueness.
        """
        timestamp = int(time.time())
        step_str = str(step).zfill(6) if step is not None else "000000"

        if _is_plotly_figure(figure):
            if self._interactive_plots:
                # Interactive HTML
                artifact_file = f"{key}_{step_str}_{timestamp}.html"
            else:
                # Static PNG
                artifact_file = f"{key}_{step_str}_{timestamp}.png"
            self._mlflow.log_figure(figure=figure, artifact_file=artifact_file)

        elif _is_matplotlib_figure(figure):
            artifact_file = f"{key}_{step_str}_{timestamp}.png"
            self._mlflow.log_figure(figure=figure, artifact_file=artifact_file)

        else:
            # PIL Image or compatible
            artifact_file = f"{key}_{step_str}_{timestamp}.png"
            self._mlflow.log_image(image=figure, artifact_file=artifact_file)

    def finish(self) -> None:
        """End the MLflow run."""
        self._mlflow.end_run()

    def get_run_id(self) -> Optional[str]:
        """Return the current MLflow run ID for checkpoint resumption."""
        if self._run is not None:
            return self._run.info.run_id
        return None
