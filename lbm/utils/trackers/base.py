"""
Abstract base class for experiment tracking backends.

Provides the interface that all trackers (W&B, MLflow, etc.) must implement.
Also includes NullTracker for when tracking is disabled.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from PIL import Image
    import matplotlib.figure
    import plotly.graph_objects


class FigureFormat(Enum):
    """Format for logging figures."""

    STATIC = "static"  # Log as static image (PNG)
    INTERACTIVE = "interactive"  # Log as interactive HTML (plotly only)


class ExperimentTracker(ABC):
    """Abstract experiment-tracking interface.

    All tracker implementations must provide these methods. The interface
    is designed to be backend-agnostic while supporting common ML experiment
    tracking patterns.
    """

    def __init__(self, interactive_plots: bool = False):
        """Initialize tracker with common settings.

        Parameters
        ----------
        interactive_plots : bool
            If True, log plotly figures as interactive HTML when supported.
            If False (default), log as static images.
        """
        self._interactive_plots = interactive_plots

    @property
    def interactive_plots(self) -> bool:
        """Whether to log plotly figures as interactive HTML."""
        return self._interactive_plots

    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        step_metric: Optional[str] = None,
    ) -> None:
        """Log scalar metrics.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Dictionary of metric names to values.
        step : Optional[int]
            Step/iteration number for the metrics.
        step_metric : Optional[str]
            When provided ("batch" or "epoch"), backends can use it so
            metrics are plotted on the correct time scale. WandB uses define_metric
            so per-batch and per-epoch metrics get separate x-axes; MLflow uses step
            as a single timeline.
        """

    @abstractmethod
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
        """Log a figure using the backend's native support.

        Handles matplotlib figures, plotly figures, and PIL images.
        When interactive_plots is True and the figure is plotly, logs
        as interactive HTML. Otherwise logs as static image.

        Parameters
        ----------
        figure : matplotlib.figure.Figure | plotly.graph_objects.Figure | PIL.Image.Image
            The figure to log.
        key : str
            Name/key for the logged figure.
        step : Optional[int]
            Step/iteration number.
        step_metric : Optional[str]
            Step metric type ("batch" or "epoch").
        """

    @abstractmethod
    def finish(self) -> None:
        """Clean up any resources before exiting."""

    def get_run_id(self) -> Optional[str]:
        """Return the current run ID if the backend supports resuming.

        Returns
        -------
        Optional[str]
            Run ID string, or None if not supported/available.
        """
        return None


class NullTracker(ExperimentTracker):
    """No-op tracker used when experiment tracking is disabled.

    All methods are implemented as no-ops, making it safe to use
    without checking for None throughout the codebase.
    """

    def __init__(self, interactive_plots: bool = False):
        super().__init__(interactive_plots=interactive_plots)

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        step_metric: Optional[str] = None,
    ) -> None:
        pass

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
        pass

    def finish(self) -> None:
        pass
