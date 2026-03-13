"""
Weights & Biases experiment tracking backend.

Provides native support for logging metrics, figures (matplotlib/plotly),
and hyperparameters to W&B.
"""

from __future__ import annotations

import logging
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


class WandbTracker(ExperimentTracker):
    """Weights & Biases implementation of the experiment tracker.

    Supports:
    - Scalar metric logging with custom x-axes (batch vs epoch)
    - Hyperparameter logging via wandb.config
    - Figure logging (matplotlib as images, plotly as interactive or static)

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration object.
    logging_cfg : DictConfig
        Logging-specific configuration containing tracker settings.

    Raises
    ------
    ImportError
        If wandb is not installed.
    """

    def __init__(self, cfg: DictConfig, logging_cfg: DictConfig):
        import wandb  # Raises ImportError if not installed

        tracker_cfg = getattr(logging_cfg, "tracker", {})
        interactive_plots = bool(tracker_cfg.get("interactive_plots", False))
        super().__init__(interactive_plots=interactive_plots)

        project = tracker_cfg.get("project", None)
        entity = tracker_cfg.get("entity", None)
        run_name = tracker_cfg.get("run_name", None)
        run_tag = tracker_cfg.get("run_tag", None)
        extra_kwargs = tracker_cfg.get("extra_init_kwargs", {}) or {}

        # Flatten the Hydra config into a plain dict for logging
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        tags = []
        if run_tag is not None:
            tags.append(str(run_tag))

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config_dict,
            tags=tags if tags else None,
            **extra_kwargs,
        )

        # Register custom step metrics so per-batch and per-epoch metrics can
        # use different x-axes (see docs.wandb.ai/guides/track/log/customize-logging-axes)
        self._run.define_metric("batch")
        self._run.define_metric("epoch")

        # Track which metrics have been associated with step metrics
        self._defined_for_step: set[tuple[str, str]] = set()

        log.info(
            f"Initialized Weights & Biases run: project={project}, "
            f"entity={entity}, name={run_name}, tags={tags}"
        )

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        step_metric: Optional[str] = None,
    ) -> None:
        """Log scalar metrics to W&B.

        When step_metric is "batch" or "epoch", metrics are associated with
        that custom x-axis so they don't mix on the same scale.
        """
        if step_metric in ("batch", "epoch"):
            to_log = dict(metrics)
            if step_metric == "batch" and step is not None:
                to_log["batch"] = step
            # "epoch" is already in metrics from build_epoch_log_metrics

            # Define step metric association for new keys
            for key in to_log:
                if key in ("batch", "epoch"):
                    continue
                tag = (key, step_metric)
                if tag not in self._defined_for_step:
                    self._run.define_metric(key, step_metric=step_metric)
                    self._defined_for_step.add(tag)

            self._wandb.log(to_log)
        else:
            if step is not None:
                self._wandb.log(metrics, step=step)
            else:
                self._wandb.log(metrics)

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
        """Log a figure to W&B.

        - Matplotlib figures: Logged via wandb.Image (automatic conversion)
        - Plotly figures: Logged via wandb.Plotly (interactive) or wandb.Image (static)
        - PIL Images: Logged via wandb.Image
        """
        to_log: Dict[str, Any] = {}

        if _is_plotly_figure(figure):
            if self._interactive_plots:
                # Interactive plotly chart
                to_log[key] = self._wandb.Plotly(figure)
            else:
                # Static image from plotly
                to_log[key] = self._wandb.Image(figure)
        elif _is_matplotlib_figure(figure):
            # W&B handles matplotlib figures automatically via wandb.Image
            to_log[key] = self._wandb.Image(figure)
        else:
            # Assume PIL Image or compatible
            to_log[key] = self._wandb.Image(figure)

        # Handle step metric association
        if step_metric in ("batch", "epoch") and step is not None:
            to_log[step_metric] = step
            tag = (key, step_metric)
            if tag not in self._defined_for_step:
                self._run.define_metric(key, step_metric=step_metric)
                self._defined_for_step.add(tag)
            self._wandb.log(to_log)
        else:
            self._wandb.log(to_log, step=step)

    def finish(self) -> None:
        """Finish the W&B run and upload remaining data."""
        if self._run is not None:
            self._run.finish()
