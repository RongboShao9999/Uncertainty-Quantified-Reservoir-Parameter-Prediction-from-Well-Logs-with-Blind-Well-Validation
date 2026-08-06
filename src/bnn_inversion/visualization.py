from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bnn_inversion.uncertainty.metrics import calibration_bin_metrics


MAX_PLOT_ROWS = 5000


def _safe_name(value: object) -> str:
    return str(value).replace("/", "_").replace("\\", "_").replace(" ", "_")


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _target_frame(predictions: pd.DataFrame, target: object) -> pd.DataFrame:
    frame = predictions[predictions["target"] == target].copy()
    return frame.reset_index(drop=True)


def _plot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) <= MAX_PLOT_ROWS:
        return frame
    return frame.iloc[np.linspace(0, len(frame) - 1, MAX_PLOT_ROWS, dtype=int)].reset_index(drop=True)


def _prediction_scatter(
    frame: pd.DataFrame, target: object, output: Path, figure_format: str
) -> Path:
    frame = _plot_frame(frame)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(frame["y_true"], frame["point_prediction"], s=22, alpha=0.75, label="point")
    if "interval_mean" in frame:
        ax.scatter(frame["y_true"], frame["interval_mean"], s=18, alpha=0.55, label="interval mean")
    values = pd.concat([frame["y_true"], frame["point_prediction"]], ignore_index=True)
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if not finite.empty:
        low, high = float(finite.min()), float(finite.max())
        ax.plot([low, high], [low, high], color="black", linewidth=1, linestyle="--")
    ax.set_title(f"{target} prediction")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.legend(loc="upper left")
    return _save(
        fig, output / f"prediction_scatter_{_safe_name(target)}.{figure_format}"
    )


def _interval_coverage(
    frame: pd.DataFrame, target: object, output: Path, figure_format: str
) -> Path | None:
    if not {"interval_mean", "lower", "upper"}.issubset(frame.columns):
        return None
    ordered = _plot_frame(frame.sort_values("y_true", kind="stable").reset_index(drop=True))
    x = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(x, ordered["y_true"], label="true", color="black", linewidth=1.5)
    ax.plot(x, ordered["interval_mean"], label="interval mean", color="#1f77b4", linewidth=1.2)
    ax.fill_between(
        x,
        ordered["lower"].to_numpy(dtype=float),
        ordered["upper"].to_numpy(dtype=float),
        color="#1f77b4",
        alpha=0.2,
        label="interval",
    )
    ax.set_title(f"{target} interval coverage")
    ax.set_xlabel("Sorted sample")
    ax.set_ylabel("Value")
    ax.legend(loc="upper left")
    return _save(
        fig, output / f"interval_coverage_{_safe_name(target)}.{figure_format}"
    )


def _uncertainty_error(
    frame: pd.DataFrame, target: object, output: Path, figure_format: str
) -> Path | None:
    if "total_variance" not in frame:
        return None
    frame = _plot_frame(frame)
    prediction_column = "interval_mean" if "interval_mean" in frame else "point_prediction"
    absolute_error = (frame[prediction_column] - frame["y_true"]).abs()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(frame["total_variance"], absolute_error, s=22, alpha=0.75)
    ax.set_title(f"{target} uncertainty vs error")
    ax.set_xlabel("Total variance")
    ax.set_ylabel("Absolute error")
    return _save(
        fig, output / f"uncertainty_error_{_safe_name(target)}.{figure_format}"
    )


def _calibration(
    frame: pd.DataFrame, target: object, output: Path, figure_format: str
) -> Path | None:
    if not {"lower", "upper", "total_variance"}.issubset(frame.columns):
        return None
    bins = calibration_bin_metrics(
        target=frame["y_true"].to_numpy(),
        lower=frame["lower"].to_numpy(),
        upper=frame["upper"].to_numpy(),
        total=frame["total_variance"].to_numpy(),
        bins=10,
    )
    if not bins:
        return None
    bin_frame = pd.DataFrame(bins)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(bin_frame["bin"], bin_frame["coverage"], marker="o", label="coverage")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"{target} calibration bins")
    ax.set_xlabel("Uncertainty bin")
    ax.set_ylabel("Coverage")
    ax.grid(True, alpha=0.25)
    return _save(fig, output / f"calibration_{_safe_name(target)}.{figure_format}")


def _trust_status(
    predictions: pd.DataFrame, output: Path, figure_format: str
) -> Path | None:
    if "status" not in predictions:
        return None
    labels = predictions["status"].astype(str).replace({"可信": "trusted", "存疑": "suspect"})
    counts = labels.value_counts()
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    ax.bar(counts.index.tolist(), counts.values.tolist(), color=["#2ca02c", "#d62728"][: len(counts)])
    ax.set_title("Trust status")
    ax.set_xlabel("Status")
    ax.set_ylabel("Rows")
    return _save(fig, output / f"trust_status.{figure_format}")


def _active_learning(
    active_learning: pd.DataFrame | None, output: Path, figure_format: str
) -> Path | None:
    if active_learning is None or active_learning.empty:
        return None
    required = {"round", "score", "epistemic_component", "inconsistency_component"}
    if not required.issubset(active_learning.columns):
        return None
    grouped = active_learning.groupby("round", as_index=False)[
        ["score", "epistemic_component", "inconsistency_component"]
    ].mean()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(grouped["round"], grouped["score"], marker="o", label="score")
    ax.plot(
        grouped["round"],
        grouped["epistemic_component"],
        marker="o",
        label="epistemic",
    )
    ax.plot(
        grouped["round"],
        grouped["inconsistency_component"],
        marker="o",
        label="inconsistency",
    )
    ax.set_title("Active learning score components")
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean value")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)
    return _save(fig, output / f"active_learning_scores.{figure_format}")


def write_visualizations(
    predictions: pd.DataFrame,
    output_dir: Path | str,
    *,
    active_learning: pd.DataFrame | None = None,
    figure_format: str = "pdf",
) -> list[Path]:
    """Write paper-facing diagnostic figures and return generated paths."""

    if figure_format not in {"pdf", "svg"}:
        raise ValueError("figure_format must be 'pdf' or 'svg'")
    output = Path(output_dir)
    paths: list[Path] = []
    if predictions.empty:
        return paths
    for target in predictions["target"].dropna().unique():
        frame = _target_frame(predictions, target)
        if frame.empty:
            continue
        paths.append(_prediction_scatter(frame, target, output, figure_format))
        for maybe_path in (
            _interval_coverage(frame, target, output, figure_format),
            _uncertainty_error(frame, target, output, figure_format),
            _calibration(frame, target, output, figure_format),
        ):
            if maybe_path is not None:
                paths.append(maybe_path)
    trust_path = _trust_status(predictions, output, figure_format)
    if trust_path is not None:
        paths.append(trust_path)
    active_path = _active_learning(active_learning, output, figure_format)
    if active_path is not None:
        paths.append(active_path)
    return paths
