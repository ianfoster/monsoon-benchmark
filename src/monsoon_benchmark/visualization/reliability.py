"""
Reliability diagram and ROC curve visualizations.

These plots assess the calibration and discrimination
of probabilistic forecasts.
"""

from typing import List, Optional, Tuple
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from monsoon_benchmark.metrics.probabilistic import ReliabilityBin

logger = logging.getLogger(__name__)


def plot_reliability_diagram(
    reliability_data: List[ReliabilityBin],
    model_name: str = "Model",
    show_histogram: bool = True,
    figsize: Tuple[int, int] = (6, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot reliability diagram.

    A reliability diagram shows forecast probability vs observed frequency.
    A perfectly calibrated forecast lies on the diagonal.

    Parameters
    ----------
    reliability_data : list of ReliabilityBin
        Reliability data from compute_reliability_diagram()
    model_name : str
        Model name for legend
    show_histogram : bool
        Whether to show forecast frequency histogram
    figsize : tuple
        Figure size
    ax : plt.Axes, optional
        Existing axes to plot on

    Returns
    -------
    plt.Figure
        Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Extract data
    forecast_probs = [b.forecast_prob_mean for b in reliability_data]
    observed_freqs = [b.observed_frequency for b in reliability_data]
    std_errors = [b.std_error for b in reliability_data]
    n_samples = [b.n_samples for b in reliability_data]

    # Remove NaN entries
    valid = ~np.isnan(observed_freqs)
    forecast_probs = np.array(forecast_probs)[valid]
    observed_freqs = np.array(observed_freqs)[valid]
    std_errors = np.array(std_errors)[valid]

    # Plot perfect reliability line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect reliability", alpha=0.7)

    # Plot no-skill line (climatology)
    clim_freq = np.mean([b.observed_frequency for b in reliability_data if not np.isnan(b.observed_frequency)])
    ax.axhline(clim_freq, color="gray", linestyle=":", alpha=0.5, label="Climatology")
    ax.axvline(clim_freq, color="gray", linestyle=":", alpha=0.5)

    # Plot reliability curve with error bars
    ax.errorbar(
        forecast_probs,
        observed_freqs,
        yerr=2 * std_errors,  # 2 standard errors
        fmt="o-",
        label=model_name,
        capsize=3,
        markersize=8,
    )

    # Shading for confidence interval
    ax.fill_between(
        forecast_probs,
        observed_freqs - 2 * std_errors,
        observed_freqs + 2 * std_errors,
        alpha=0.2,
    )

    ax.set_xlabel("Forecast Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.set_title(f"Reliability Diagram - {model_name}")
    ax.grid(True, alpha=0.3)

    # Add histogram inset if requested
    if show_histogram:
        ax_hist = ax.inset_axes([0.6, 0.1, 0.35, 0.25])
        n_samples_valid = np.array([b.n_samples for b in reliability_data])[valid]
        ax_hist.bar(
            forecast_probs,
            n_samples_valid,
            width=0.08,
            alpha=0.7,
            color="steelblue",
        )
        ax_hist.set_xlabel("Forecast Prob", fontsize=8)
        ax_hist.set_ylabel("Count", fontsize=8)
        ax_hist.tick_params(labelsize=7)
        ax_hist.set_yscale("log")

    return fig


def plot_multi_model_reliability(
    model_reliability: dict,
    figsize: Tuple[int, int] = (12, 4),
) -> plt.Figure:
    """
    Plot reliability diagrams for multiple models.

    Parameters
    ----------
    model_reliability : dict
        Dictionary of {model_name: list of ReliabilityBin}
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
        Figure with subplots for each model
    """
    n_models = len(model_reliability)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, rel_data) in zip(axes, model_reliability.items()):
        plot_reliability_diagram(rel_data, model_name=model_name, ax=ax)

    plt.tight_layout()
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (6, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot ROC (Receiver Operating Characteristic) curve.

    Parameters
    ----------
    fpr : np.ndarray
        False positive rates
    tpr : np.ndarray
        True positive rates
    auc : float
        Area under the ROC curve
    model_name : str
        Model name for legend
    figsize : tuple
        Figure size
    ax : plt.Axes, optional
        Existing axes

    Returns
    -------
    plt.Figure
        Figure object
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Plot diagonal (no skill)
    ax.plot([0, 1], [0, 1], "k--", label="No skill (AUC=0.5)")

    # Plot ROC curve
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", linewidth=2)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.set_title("ROC Curve")
    ax.grid(True, alpha=0.3)

    return fig


def plot_probabilistic_metrics_summary(
    metrics_df,
    forecast_window: str = "1-15 day",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot summary of probabilistic metrics (BSS, RPSS, AUC).

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame with columns: model, bss, rpss, auc
    forecast_window : str
        Forecast window for title
    figsize : tuple
        Figure size

    Returns
    -------
    plt.Figure
        Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    metrics = ["bss", "rpss", "auc"]
    labels = ["Brier Skill Score (%)", "Ranked Probability Skill Score (%)", "AUC"]
    colors = ["steelblue", "forestgreen", "coral"]

    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        models = metrics_df["model"].values
        values = metrics_df[metric].values

        # Convert skill scores to percentages
        if metric in ["bss", "rpss"]:
            values = values * 100

        bars = ax.barh(models, values, color=color, alpha=0.8)

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}",
                va="center",
                fontsize=9,
            )

        ax.axvline(0, color="k", linestyle="-", linewidth=0.5)
        ax.set_xlabel(label)
        ax.set_title(metric.upper())
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle(f"Probabilistic Metrics - {forecast_window} forecast", fontsize=12)
    plt.tight_layout()

    return fig
