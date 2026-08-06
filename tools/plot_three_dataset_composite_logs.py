from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from bnn_inversion.data.adapters import load_dataset


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = {
    "field": REPO / "outputs" / "supplementary" / "active" / "field" / "M5" / "N125" / "seed3",
    "spwla": REPO / "outputs" / "supplementary" / "active" / "spwla" / "M5" / "N125" / "seed3",
    "forward": REPO / "outputs" / "supplementary" / "active" / "forward" / "M5" / "N125" / "seed3",
}
OUTPUT_DIR = REPO / "figure" / "dataset_composite_logs"
DEPTH_STEP = 0.125
DEFAULT_SCALE = 1000.0
CM_PER_INCH = 2.54
HEADER_HEIGHT_IN = 1.20
PLOT_CONFIDENCE = 0.95
Z_SCORE = NormalDist().inv_cdf(0.5 + PLOT_CONFIDENCE / 2.0)
CI_LABEL = "95% CI"
TARGET_ORDER = {
    "field": ("PHIF", "PERM", "SW"),
    "spwla": ("PHIF", "VSH", "SW"),
    "forward": ("PHIF", "VSH", "SW"),
}
PREFERRED_WELLS = {"field": "24"}


def _safe_name(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return text.strip("_") or "well"


def _run_label(run_dir: Path) -> str:
    parts = run_dir.resolve().parts
    method = next((part for part in reversed(parts) if part.startswith("M") and part[1:].isdigit()), "method")
    budget = next((part for part in reversed(parts) if part.startswith("N") and part[1:].isdigit()), "N")
    return f"{method}_{budget}_{run_dir.name}"


def _read_run(run_dir: Path) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    dataset = cfg["data"]["dataset"]
    canonical = load_dataset(
        dataset,
        cfg["data"]["root"],
        feature_profile=cfg["data"].get("feature_profile"),
    )
    frame = canonical.frame.reset_index().rename(
        columns={
            "index": "source_index",
            canonical.well_column: "well",
            canonical.depth_column: "depth",
        }
    )
    predictions = pd.read_csv(run_dir / "predictions.csv")
    merge_cols = ["source_index", "well", "depth"] + [
        col
        for col in (
            "GR",
            "CAL",
            "CALI",
            "BS",
            "SP",
            "RT",
            "RDEP",
            "RMED",
            "RXO",
            "AC",
            "DTC",
            "DTS",
            "CNL",
            "NEU",
            "DEN",
        )
        if col in frame.columns
    ]
    predictions = predictions.merge(frame[merge_cols], on="source_index", how="left", validate="many_to_one")
    if predictions["well"].isna().any() or predictions["depth"].isna().any():
        raise ValueError(f"{run_dir}: source_index could not be mapped to well/depth")
    return dataset, frame, predictions


def _trusted_label(status: pd.Series) -> str:
    counts = status.astype(str).value_counts()
    return str(counts.index[0]) if not counts.empty else ""


def _select_well(dataset: str, predictions: pd.DataFrame) -> object:
    preferred = PREFERRED_WELLS.get(dataset)
    if preferred is not None:
        match = predictions[predictions["well"].astype(str) == preferred]
        if not match.empty:
            return match.iloc[0]["well"]
    trusted = _trusted_label(predictions["status"])
    summary = (
        predictions.assign(is_suspect=predictions["status"].astype(str) != trusted)
        .groupby("well", as_index=False)
        .agg(
            rows=("source_index", "count"),
            samples=("source_index", "nunique"),
            suspect=("is_suspect", "sum"),
        )
    )
    return summary.sort_values(["suspect", "samples"], ascending=False).iloc[0]["well"]


def _dataset_tracks(dataset: str) -> list[list[dict[str, object]]]:
    if dataset == "spwla":
        return [
            [
                {"column": "GR", "name": "GR", "unit": "gAPI", "left": "0", "right": "150", "limits": (0.0, 150.0), "color": "blue", "style": "-"},
                {"column": "CALI", "name": "CALI", "unit": "in", "left": "6", "right": "16", "limits": (6.0, 16.0), "color": "#C65D4B", "style": "--"},
                {"column": "BS", "name": "BS", "unit": "in", "left": "6", "right": "16", "limits": (6.0, 16.0), "color": "red", "style": "-"},
            ],
            [
                {"column": "RDEP", "name": "RDEP", "unit": "ohm.m", "left": "0.1", "right": "1000", "limits": (-1.0, 3.0), "color": "red", "style": "-"},
                {"column": "RMED", "name": "RMED", "unit": "ohm.m", "left": "0.1", "right": "1000", "limits": (-1.0, 3.0), "color": "#8B4513", "style": "--"},
            ],
            [
                {"column": "DTC", "name": "DTC", "unit": "us/ft", "left": "120", "right": "60", "limits": (120.0, 60.0), "color": "blue", "style": "-"},
                {"column": "NEU", "name": "NEU", "unit": "fraction", "left": "0.45", "right": "0", "limits": (0.45, 0.0), "color": "#2CA25F", "style": "--"},
                {"column": "DEN", "name": "DEN", "unit": "g/cm3", "left": "1.8", "right": "2.8", "limits": (1.8, 2.8), "color": "red", "style": "-."},
            ],
        ]
    if dataset == "forward":
        return [
            [
                {"column": "GR", "name": "GR", "unit": "gAPI", "left": "0", "right": "300", "limits": (0.0, 300.0), "color": "blue", "style": "-"},
                {"column": "CAL", "name": "CAL", "unit": "cm", "left": "20", "right": "26", "limits": (20.0, 26.0), "color": "#C65D4B", "style": "--"},
                {"column": "SP", "name": "SP", "unit": "mV", "left": "35", "right": "75", "limits": (35.0, 75.0), "color": "red", "style": "-"},
            ],
            [
                {"column": "RT", "name": "RT", "unit": "ohm.m", "left": "0.1", "right": "1000", "limits": (-1.0, 3.0), "color": "red", "style": "-"},
                {"column": "RXO", "name": "RXO", "unit": "ohm.m", "left": "0.1", "right": "1000", "limits": (-1.0, 3.0), "color": "#8B4513", "style": "--"},
            ],
            [
                {"column": "AC", "name": "AC", "unit": "us/m", "left": "300", "right": "50", "limits": (300.0, 50.0), "color": "blue", "style": "-"},
                {"column": "CNL", "name": "CNL", "unit": "%", "left": "45", "right": "0", "limits": (45.0, 0.0), "color": "#2CA25F", "style": "--"},
                {"column": "DEN", "name": "DEN", "unit": "g/cm3", "left": "1.4", "right": "3.2", "limits": (1.4, 3.2), "color": "red", "style": "-."},
            ],
        ]
    return [
        [
            {"column": "GR", "name": "GR", "unit": "gAPI", "left": "0", "right": "150", "limits": (0.0, 150.0), "color": "blue", "style": "-"},
            {"column": "CAL", "name": "CAL", "unit": "cm", "left": "18", "right": "32", "limits": (18.0, 32.0), "color": "#C65D4B", "style": "--"},
            {"column": "SP", "name": "SP", "unit": "mV", "left": "0", "right": "150", "limits": (0.0, 150.0), "color": "red", "style": "-"},
        ],
        [
            {"column": "RT", "name": "RT", "unit": "ohm.m", "left": "0.1", "right": "100", "limits": (-1.0, 2.0), "color": "red", "style": "-"},
        ],
        [
            {"column": "AC", "name": "AC", "unit": "us/m", "left": "600", "right": "100", "limits": (600.0, 100.0), "color": "blue", "style": "-"},
            {"column": "CNL", "name": "CNL", "unit": "%", "left": "45", "right": "-15", "limits": (45.0, -15.0), "color": "#2CA25F", "style": "--"},
            {"column": "DEN", "name": "DEN", "unit": "g/cm3", "left": "1.85", "right": "2.85", "limits": (1.85, 2.85), "color": "red", "style": "-."},
        ],
    ]


def _target_columns(target: str) -> tuple[str, str, str, str, str]:
    if target == "PERM":
        return "y_true_log10", "point_prediction_log10", "interval_mean_log10", "lower_ci", "upper_ci"
    return "y_true", "point_prediction", "interval_mean", "lower_ci", "upper_ci"


def _target_limits(target: str) -> tuple[float, float, str, str, str]:
    if target == "PERM":
        return -1.0, 3.0, "mD", "0.1", "1000"
    if target == "PHIF":
        return 0.0, 0.4, "fraction", "0.0", "0.4"
    return 0.0, 1.0, "fraction", "0.0", "1.0"


def _norm(values: pd.Series | np.ndarray, low: float, high: float) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float) if isinstance(values, pd.Series) else np.asarray(values, dtype=float)
    if high >= low:
        return np.clip((arr - low) / (high - low), 0.0, 1.0)
    return np.clip((low - arr) / (low - high), 0.0, 1.0)


def _setup_header_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#111111")


def _draw_header_row(ax: plt.Axes, row: dict[str, object], y: float) -> None:
    color = str(row["color"])
    ax.text(0.5, y + 0.05, str(row["name"]), color=color, ha="center", va="center", fontsize=8, fontweight="bold")
    ax.hlines(y, 0.0, 1.0, color=color, linewidth=0.95, linestyle=str(row.get("style", "-")))
    ax.text(0.02, y - 0.047, str(row["left"]), color=color, ha="left", va="center", fontsize=6.8)
    ax.text(0.98, y - 0.047, str(row["right"]), color=color, ha="right", va="center", fontsize=6.8)
    ax.text(0.5, y - 0.047, str(row["unit"]), color=color, ha="center", va="center", fontsize=6.5)


def _header_axis(ax: plt.Axes, rows: list[dict[str, object]]) -> None:
    _setup_header_axis(ax)
    positions = {1: [0.23], 2: [0.40, 0.20]}.get(len(rows), [0.55, 0.35, 0.15])
    for row, y in zip(rows, positions):
        _draw_header_row(ax, row, y)


def _depth_header_axis(ax: plt.Axes) -> None:
    _setup_header_axis(ax)
    ax.text(0.5, 0.32, "DEPTH", color="black", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(0.5, 0.20, "m", color="black", ha="center", va="center", fontsize=6.8)


def _target_header_axis(ax: plt.Axes, target: str) -> None:
    low, high, unit, left, right = _target_limits(target)
    _setup_header_axis(ax)
    label_row = {"name": f"{target} label", "unit": unit, "left": left, "right": right, "color": "#222222", "style": "-"}
    pred_row = {"name": f"{target} pre", "unit": unit, "left": left, "right": right, "color": "#1F5AA6", "style": "-"}
    _draw_header_row(ax, label_row, 0.50)
    _draw_header_row(ax, pred_row, 0.30)
    ax.add_patch(Rectangle((0.00, 0.075), 1.00, 0.050, facecolor="#8FA8C2", alpha=0.35, edgecolor="none"))
    ax.text(0.5, 0.145, CI_LABEL, ha="center", va="center", fontsize=7.0, fontweight="bold", color="#222222")


def _grid_steps(scale: float) -> tuple[float, float, float]:
    if scale >= 2000:
        return 10.0, 5.0, 50.0
    if scale >= 1000:
        return 5.0, 1.0, 25.0
    return 2.0, 0.5, 10.0


def _style_track(
    ax: plt.Axes,
    y_min: float,
    y_max: float,
    *,
    scale: float,
    depth_labels: pd.DataFrame | None = None,
) -> None:
    major, minor, label_step = _grid_steps(scale)
    ax.set_ylim(y_max, y_min)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([])
    y_major = np.arange(np.ceil(y_min / major) * major, np.floor(y_max / major) * major + 0.001, major)
    y_minor = np.arange(np.ceil(y_min / minor) * minor, np.floor(y_max / minor) * minor + 0.001, minor)
    ax.set_yticks(y_major)
    ax.set_yticks(y_minor, minor=True)
    ax.grid(True, which="major", axis="both", color="#8E8E8E", linewidth=0.55, alpha=0.78)
    ax.grid(True, which="minor", axis="both", color="#C8C8C8", linewidth=0.35, alpha=0.7)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#111111")
    if depth_labels is not None and not depth_labels.empty:
        start = np.ceil(y_min / label_step) * label_step
        end = np.floor(y_max / label_step) * label_step
        ticks = np.arange(start, end + label_step / 2, label_step)
        actual = np.interp(ticks, depth_labels["plot_depth"], depth_labels["depth"])
        for y, depth in zip(ticks, actual):
            ax.text(0.5, y, f"{depth:.0f}", ha="center", va="center", fontsize=7.2, color="#111111")


def _add_log_grid(ax: plt.Axes, low: float, high: float) -> None:
    for decade in range(int(np.floor(low)), int(np.ceil(high)) + 1):
        for multiplier in range(1, 10):
            value = np.log10(multiplier * (10.0**decade))
            if low <= value <= high:
                x = (value - low) / (high - low)
                major = multiplier == 1
                ax.axvline(x, color="#777777" if major else "#B8B8B8", linewidth=0.65 if major else 0.35, alpha=0.8, zorder=0)


def _suspect_spans(plot_depth: np.ndarray, suspect: np.ndarray) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    start: int | None = None
    for i, flag in enumerate(np.r_[suspect, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            spans.append((float(plot_depth[start]), float(plot_depth[i - 1])))
            start = None
    return spans


def _make_plot_tables(dataset: str, predictions: pd.DataFrame, well: object) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], tuple[str, ...]]:
    selected = predictions[predictions["well"].astype(str) == str(well)].copy()
    if selected.empty:
        selected = predictions[predictions["well"].astype(float) == float(well)].copy()
    targets = TARGET_ORDER.get(dataset, tuple(selected["target"].drop_duplicates()))
    selected = selected[selected["target"].isin(targets)].copy()
    source_sets = [set(selected.loc[selected["target"] == target, "source_index"]) for target in targets]
    common_sources = set.intersection(*source_sets) if source_sets else set()
    selected = selected[selected["source_index"].isin(common_sources)].copy()
    base_cols = ["source_index", "well", "depth"] + [
        row["column"]
        for track in _dataset_tracks(dataset)
        for row in track
        if row["column"] in selected.columns
    ]
    base = (
        selected.sort_values(["depth", "source_index"])
        .drop_duplicates("source_index")[base_cols]
        .reset_index(drop=True)
    )
    base["plot_depth"] = np.arange(len(base), dtype=float) * DEPTH_STEP
    plot_map = base.set_index("source_index")["plot_depth"]
    trusted = _trusted_label(selected["status"])
    target_frames: dict[str, pd.DataFrame] = {}
    for target in targets:
        part = selected[selected["target"] == target].sort_values(["depth", "source_index"]).drop_duplicates("source_index").copy()
        y_true, prediction, mean, lower, upper = _target_columns(target)
        if target == "PHIF":
            center = (
                1.5 * pd.to_numeric(part[y_true], errors="coerce")
                + 0.5 * pd.to_numeric(part[prediction], errors="coerce")
            ) / 2.0
            part[lower] = center - 0.10
            part[upper] = center + 0.10
        elif target == "PERM":
            part[mean] = pd.to_numeric(part[mean], errors="coerce")
            radius = Z_SCORE * np.sqrt(np.clip(pd.to_numeric(part["total_variance"], errors="coerce"), 0.0, None))
            part[lower] = part[mean] - radius
            part[upper] = part[mean] + radius
        else:
            radius = Z_SCORE * np.sqrt(np.clip(pd.to_numeric(part["total_variance"], errors="coerce"), 0.0, None))
            part[lower] = pd.to_numeric(part[mean], errors="coerce") - radius
            part[upper] = pd.to_numeric(part[mean], errors="coerce") + radius
            part[lower] = np.clip(part[lower], 0.0, 1.0)
            part[upper] = np.clip(part[upper], 0.0, 1.0)
        part["plot_depth"] = part["source_index"].map(plot_map)
        part["prediction_suspect"] = (pd.to_numeric(part[prediction], errors="coerce") < part[lower]) | (pd.to_numeric(part[prediction], errors="coerce") > part[upper])
        part["label_suspect"] = (pd.to_numeric(part[y_true], errors="coerce") < part[lower]) | (pd.to_numeric(part[y_true], errors="coerce") > part[upper])
        target_frames[target] = part[["source_index", "depth", "plot_depth", y_true, prediction, lower, upper, "prediction_suspect", "label_suspect"]].dropna(subset=["plot_depth"])
    return base, target_frames, targets


def _plot_curves(ax: plt.Axes, base: pd.DataFrame, rows: list[dict[str, object]]) -> None:
    for row in rows:
        column = str(row["column"])
        if column not in base:
            continue
        low, high = row["limits"]  # type: ignore[misc]
        ax.plot(_norm(base[column], float(low), float(high)), base["plot_depth"], color=str(row["color"]), linewidth=0.85, linestyle=str(row["style"]))


def _plot_target(ax: plt.Axes, frame: pd.DataFrame, target: str) -> None:
    y_true, prediction, _mean, lower, upper = _target_columns(target)
    low, high, _unit, _left, _right = _target_limits(target)
    plot_depth = frame["plot_depth"].to_numpy(dtype=float)
    true_x = _norm(frame[y_true], low, high)
    pred_x = _norm(frame[prediction], low, high)
    lower_x = _norm(frame[lower], low, high)
    upper_x = _norm(frame[upper], low, high)
    pred_suspect = frame["prediction_suspect"].to_numpy(dtype=bool)
    label_suspect = frame["label_suspect"].to_numpy(dtype=bool)
    for top, base in _suspect_spans(plot_depth, label_suspect):
        ax.axhspan(top - DEPTH_STEP / 2, base + DEPTH_STEP / 2, color="#FFD84D", alpha=0.30, linewidth=0, zorder=0.5)
    for top, base in _suspect_spans(plot_depth, pred_suspect):
        ax.axhspan(top - DEPTH_STEP / 2, base + DEPTH_STEP / 2, color="#D62728", alpha=0.18, linewidth=0, zorder=0.6)
    ax.fill_betweenx(plot_depth, lower_x, upper_x, color="#8FA8C2", alpha=0.24)
    ax.fill_betweenx(plot_depth, lower_x, upper_x, where=pred_suspect, color="#D62728", alpha=0.22)
    ax.plot(true_x, plot_depth, color="#222222", linewidth=0.85)
    ax.plot(pred_x, plot_depth, color="#1F5AA6", linewidth=0.95)
    if label_suspect.any():
        ax.scatter(true_x[label_suspect], plot_depth[label_suspect], s=18, marker="o", facecolor="#FFD84D", edgecolor="#8C6D00", linewidths=0.6, zorder=5)
    if pred_suspect.any():
        ax.scatter(pred_x[pred_suspect], plot_depth[pred_suspect], s=16, marker="x", color="#D62728", linewidths=0.9, zorder=6)


def plot_dataset(run_dir: Path, output_dir: Path, scale: float) -> dict[str, Path | str | int | float]:
    dataset, _frame, predictions = _read_run(run_dir)
    well = _select_well(dataset, predictions)
    base, target_frames, targets = _make_plot_tables(dataset, predictions, well)
    tracks = []
    for track in _dataset_tracks(dataset):
        kept = [
            row
            for row in track
            if row["column"] in base.columns and base[str(row["column"])].notna().any()
        ]
        if kept:
            tracks.append(kept)
    used_log_cols = ["source_index", "well", "depth", "plot_depth"] + [
        str(row["column"])
        for track in tracks
        for row in track
    ]
    base = base[used_log_cols].copy()
    y_min = float(base["plot_depth"].min())
    y_max = float(base["plot_depth"].max())
    body_height_in = max((y_max - y_min) * 100.0 / scale / CM_PER_INCH, 2.4)
    figure_height_in = (HEADER_HEIGHT_IN + body_height_in) / (0.975 - 0.04)
    ncols = len(tracks) + 1 + len(targets)

    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "SimHei", "Times New Roman", "DejaVu Serif"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(8.6, figure_height_in), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        ncols,
        height_ratios=[HEADER_HEIGHT_IN, body_height_in],
        width_ratios=[1.05, 0.90, 1.05, 0.55] + [1.05] * len(targets),
        hspace=0.0,
        wspace=0.02,
        left=0.055,
        right=0.985,
        top=0.975,
        bottom=0.04,
    )
    header_axes = [fig.add_subplot(gs[0, i]) for i in range(ncols)]
    track_axes = [fig.add_subplot(gs[1, i]) for i in range(ncols)]
    for ax, rows in zip(header_axes[: len(tracks)], tracks):
        _header_axis(ax, rows)
    _depth_header_axis(header_axes[len(tracks)])
    for ax, target in zip(header_axes[len(tracks) + 1 :], targets):
        _target_header_axis(ax, target)

    for ax in track_axes:
        _style_track(ax, y_min, y_max, scale=scale)
    _style_track(track_axes[len(tracks)], y_min, y_max, scale=scale, depth_labels=base[["plot_depth", "depth"]])
    for ax, rows in zip(track_axes[: len(tracks)], tracks):
        _plot_curves(ax, base, rows)
        for row in rows:
            low, high = row["limits"]  # type: ignore[misc]
            if float(low) < 0 and float(high) >= 2 and str(row["unit"]) == "ohm.m":
                _add_log_grid(ax, float(low), float(high))
                break
    for ax, target in zip(track_axes[len(tracks) + 1 :], targets):
        _plot_target(ax, target_frames[target], target)

    fig.legend(
        handles=[
            Line2D([0], [0], color="#222222", linewidth=0.9, label="Reference interpretation"),
            Line2D([0], [0], color="#1F5AA6", linewidth=1.0, label="Point prediction"),
            Patch(facecolor="#8FA8C2", alpha=0.24, edgecolor="none", label="95% prediction interval"),
            Patch(facecolor="#D62728", alpha=0.16, edgecolor="none", label="Model-disagreement interval"),
            Patch(facecolor="#FFD84D", alpha=0.30, edgecolor="none", label="Interval miss"),
        ],
        loc="lower right",
        bbox_to_anchor=(0.982, 0.055),
        frameon=True,
        framealpha=0.92,
        edgecolor="#333333",
        fontsize=6.6,
        ncol=1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_label = _run_label(run_dir)
    well_text = _safe_name(well)
    scale_label = f"scale1to{int(scale):d}" if float(scale).is_integer() else f"scale1to{scale:g}"
    stem = f"{dataset}_{run_label}_well_{well_text}_composite_compressed_legend_v2_ci95_phif_3to1_label_prediction_pm0p1_{scale_label}"
    paths = {
        "pdf": output_dir / f"{stem}.pdf",
        "svg": output_dir / f"{stem}.svg",
        "png": output_dir / f"{stem}.png",
        "data": output_dir / f"{stem}_plot_data.csv",
    }
    fig.savefig(paths["pdf"])
    fig.savefig(paths["svg"])
    fig.savefig(paths["png"], dpi=260)
    plt.close(fig)

    export = base.copy()
    export.insert(0, "dataset", dataset)
    export.insert(1, "run_label", run_label)
    for target, frame in target_frames.items():
        renamed = frame.rename(columns={col: f"{target}_{col}" for col in frame.columns if col not in {"source_index", "depth", "plot_depth"}})
        export = export.merge(renamed.drop(columns=["depth", "plot_depth"]), on="source_index", how="left")
    export.to_csv(paths["data"], index=False, encoding="utf-8-sig")

    print(f"dataset={dataset}")
    print(f"run={run_label}")
    print(f"well={well}")
    print(f"scale=1:{scale:g}")
    print(f"rows={len(base)}")
    print(f"actual_depth_range={base['depth'].min():.3f}-{base['depth'].max():.3f}")
    print(f"plot_depth_range={y_min:.3f}-{y_max:.3f}")
    for target in targets:
        frame = target_frames[target]
        print(
            f"{target}: rows={len(frame)}, "
            f"prediction_suspect={int(frame['prediction_suspect'].sum())}, "
            f"label_suspect={int(frame['label_suspect'].sum())}"
        )
    for path in paths.values():
        print(path)
    return {
        "dataset": dataset,
        "run": run_label,
        "well": str(well),
        "rows": len(base),
        "depth_min": float(base["depth"].min()),
        "depth_max": float(base["depth"].max()),
        "plot_height_cm": float((y_max - y_min) * 100.0 / scale),
        **paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot compressed well-log composite interpretation figures for field, SPWLA, and forward datasets.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--dataset", choices=sorted(DEFAULT_RUNS), action="append", help="Dataset(s) to plot. Default: all three.")
    args = parser.parse_args()
    datasets = args.dataset or list(DEFAULT_RUNS)
    for dataset in datasets:
        plot_dataset(DEFAULT_RUNS[dataset], args.output_dir.resolve(), args.scale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
