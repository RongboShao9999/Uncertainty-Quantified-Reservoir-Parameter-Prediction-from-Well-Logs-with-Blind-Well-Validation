from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from statistics import NormalDist

from bnn_inversion.data.adapters import load_dataset


REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "outputs" / "field_m1_m9_multiseed" / "M9" / "N50" / "seed2"
OUTPUT_DIR = REPO / "figure" / "field_suspect_well"
TARGETS = ("PHIF", "PERM", "SW")
DEPTH_STEP = 0.125
DEPTH_SCALE_DENOMINATOR = 200.0
MAX_INTERP_GAP = DEPTH_STEP * 1.01
CM_PER_INCH = 2.54
HEADER_HEIGHT_IN = 1.20
PLOT_CONFIDENCE = 0.95
Z_SCORE = NormalDist().inv_cdf(0.5 + PLOT_CONFIDENCE / 2.0)
CI_LABEL = "95% CI"


def _read_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    canonical = load_dataset(
        cfg["data"]["dataset"],
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
    usecols = [
        "source_index",
        "target",
        "y_true",
        "point_prediction",
        "interval_mean",
        "lower",
        "upper",
        "total_variance",
        "status",
        "y_true_log10",
        "point_prediction_log10",
        "interval_mean_log10",
        "lower_log10",
        "upper_log10",
    ]
    predictions = pd.read_csv(run_dir / "predictions.csv", usecols=lambda c: c in usecols)
    predictions = predictions.merge(
        frame[
            [
                "source_index",
                "well",
                "depth",
                "GR",
                "CAL",
                "SP",
                "AC",
                "CNL",
                "DEN",
                "RT",
                "PHIF",
                "PERM",
                "SW",
            ]
        ],
        on="source_index",
        how="left",
        validate="many_to_one",
    )
    if predictions["well"].isna().any():
        raise ValueError("source_index could not be mapped to well/depth")
    return frame, predictions, canonical.well_column


def _select_well(predictions: pd.DataFrame) -> object:
    perm = predictions[predictions["target"] == "PERM"].copy()
    perm["absolute_error"] = (perm["point_prediction"] - perm["y_true"]).abs()
    summary = (
        perm.groupby("well", as_index=False)
        .agg(
            rows=("source_index", "count"),
            suspect_count=("status", lambda s: int(s.astype(str).eq("存疑").sum())),
            suspect_mae=(
                "absolute_error",
                lambda s: float(s.loc[perm.loc[s.index, "status"].astype(str).eq("存疑")].mean())
                if perm.loc[s.index, "status"].astype(str).eq("存疑").any()
                else np.nan,
            ),
        )
        .dropna(subset=["suspect_mae"])
    )
    if summary.empty:
        raise ValueError("no field well has suspect PERM intervals")
    return summary.sort_values(["suspect_count", "suspect_mae"], ascending=False).iloc[0]["well"]


def _representative_wells(predictions: pd.DataFrame, count: int = 3) -> list[object]:
    """Choose low-, median-, and high-error wells from PERM in log10 space."""
    perm = predictions[predictions["target"] == "PERM"].copy()
    error = perm["point_prediction_log10"] - perm["y_true_log10"]
    summary = perm.assign(squared_error=error**2).groupby("well", as_index=False).agg(
        rows=("source_index", "count"), rmse=("squared_error", lambda x: float(np.sqrt(np.mean(x))))
    )
    summary = summary[summary["rows"] >= 10].sort_values("rmse").reset_index(drop=True)
    if len(summary) < count:
        raise ValueError(f"need at least {count} field wells with predictions")
    return summary.iloc[np.linspace(0, len(summary) - 1, count).round().astype(int)]["well"].tolist()


def _norm(values: pd.Series, low: float, high: float) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if high >= low:
        return np.clip((arr - low) / (high - low), 0.0, 1.0)
    return np.clip((low - arr) / (low - high), 0.0, 1.0)


def _plot_y(frame: pd.DataFrame) -> pd.Series:
    return frame["plot_depth"] if "plot_depth" in frame.columns else frame["depth"]


def _format_well(well: object) -> str:
    try:
        value = float(well)
        return str(int(value)) if value.is_integer() else f"{value:g}"
    except Exception:
        return str(well)


def _setup_header_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("#111111")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def _draw_standard_header_row(
    ax: plt.Axes,
    *,
    y: float,
    name: str,
    unit_text: str,
    left: str,
    right: str,
    color: str,
    linestyle: str = "-",
    name_color: str | None = None,
) -> None:
    text_color = name_color or color
    ax.text(0.5, y + 0.050, name, color=text_color, ha="center", va="center", fontsize=8, fontweight="bold")
    ax.hlines(y, 0.0, 1.0, color=color, linewidth=0.95, linestyle=linestyle)
    ax.text(0.02, y - 0.047, left, color=color, ha="left", va="center", fontsize=6.8)
    ax.text(0.98, y - 0.047, right, color=color, ha="right", va="center", fontsize=6.8)
    ax.text(0.5, y - 0.047, unit_text, color=color, ha="center", va="center", fontsize=6.5)


def _header_axis(ax: plt.Axes, lines: list[tuple[str, str, str, str, str, str]]) -> None:
    _setup_header_axis(ax)
    n = len(lines)
    if n == 1:
        y_positions = [0.23]
    elif n == 2:
        y_positions = [0.40, 0.20]
    else:
        y_positions = [0.55, 0.35, 0.15]
    for idx, (name, unit_text, left, right, color, linestyle) in enumerate(lines):
        _draw_standard_header_row(
            ax,
            y=y_positions[idx],
            name=name,
            unit_text=unit_text,
            left=left,
            right=right,
            color=color,
            linestyle=linestyle,
        )


def _target_header_axis(ax: plt.Axes, target: str, unit_text: str, left: str, right: str) -> None:
    _setup_header_axis(ax)
    _draw_standard_header_row(
        ax,
        y=0.50,
        name=f"{target}_label",
        unit_text=unit_text,
        left=left,
        right=right,
        color="#222222",
        name_color="black",
    )
    _draw_standard_header_row(
        ax,
        y=0.30,
        name=f"{target}_pre",
        unit_text=unit_text,
        left=left,
        right=right,
        color="#1F5AA6",
        name_color="#1F5AA6",
    )
    ax.add_patch(Rectangle((0.00, 0.075), 1.00, 0.050, facecolor="#8FA8C2", alpha=0.35, edgecolor="none"))
    ax.text(0.5, 0.145, CI_LABEL, ha="center", va="center", fontsize=7.0, fontweight="bold", color="#222222")


def _depth_header_axis(ax: plt.Axes) -> None:
    _setup_header_axis(ax)
    ax.text(0.5, 0.32, "DEPTH", color="black", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(0.5, 0.20, "m", color="black", ha="center", va="center", fontsize=6.8)


def _depth_grid_steps(scale: float) -> tuple[float, float, float]:
    """Return major grid, minor grid, and labelled-depth spacing in metres."""
    if scale >= 2000:
        return 10.0, 5.0, 50.0
    if scale >= 500:
        return 5.0, 1.0, 20.0
    return 1.0, 0.5, 10.0


def _style_track(
    ax: plt.Axes,
    depth_min: float,
    depth_max: float,
    *,
    show_y: bool = False,
    scale: float = DEPTH_SCALE_DENOMINATOR,
    depth_label_frame: pd.DataFrame | None = None,
) -> None:
    ax.set_ylim(depth_max, depth_min)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([])
    major_step, minor_step, label_step = _depth_grid_steps(scale)
    y_major = np.arange(np.ceil(depth_min / major_step) * major_step, np.floor(depth_max / major_step) * major_step + 0.001, major_step)
    y_minor = np.arange(np.ceil(depth_min / minor_step) * minor_step, np.floor(depth_max / minor_step) * minor_step + 0.001, minor_step)
    ax.set_yticks(y_major)
    ax.set_yticks(y_minor, minor=True)
    ax.grid(True, which="major", axis="both", color="#8E8E8E", linewidth=0.55, alpha=0.78)
    ax.minorticks_on()
    ax.set_yticks(y_minor, minor=True)
    ax.grid(True, which="minor", axis="both", color="#C8C8C8", linewidth=0.35, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#111111")
    if not show_y:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        start = np.ceil(depth_min / label_step) * label_step
        end = np.floor(depth_max / label_step) * label_step
        ticks = np.arange(start, end + label_step / 2, label_step)
        if depth_label_frame is not None and not depth_label_frame.empty and "plot_depth" in depth_label_frame:
            labels = depth_label_frame[["plot_depth", "depth"]].dropna().sort_values("plot_depth")
            if not labels.empty:
                actual_depth = np.interp(
                    ticks,
                    labels["plot_depth"].to_numpy(dtype=float),
                    labels["depth"].to_numpy(dtype=float),
                )
                for tick, label in zip(ticks, actual_depth):
                    ax.text(0.5, tick, f"{label:.0f}", ha="center", va="center", fontsize=7.2, color="#111111")
        else:
            for tick in ticks:
                ax.text(0.5, tick, f"{tick:.0f}", ha="center", va="center", fontsize=7.2, color="#111111")


def _add_log_grid(ax: plt.Axes, low_log10: float, high_log10: float) -> None:
    for decade in range(int(np.floor(low_log10)), int(np.ceil(high_log10)) + 1):
        for multiplier in range(1, 10):
            value = np.log10(multiplier * (10.0**decade))
            if low_log10 <= value <= high_log10:
                x = (value - low_log10) / (high_log10 - low_log10)
                major = multiplier == 1
                ax.axvline(
                    x,
                    color="#777777" if major else "#B8B8B8",
                    linewidth=0.65 if major else 0.35,
                    alpha=0.85 if major else 0.70,
                    zorder=0,
                )


def _gap_mask(depth_grid: np.ndarray, source_depth: np.ndarray, max_gap: float = MAX_INTERP_GAP) -> np.ndarray:
    """Return grid samples that fall inside missing source-depth intervals."""
    depth = np.unique(source_depth[np.isfinite(source_depth)])
    missing = np.zeros(len(depth_grid), dtype=bool)
    if len(depth) < 2:
        return missing
    for start, end in zip(depth[:-1], depth[1:]):
        if end - start > max_gap:
            missing |= (depth_grid > start + 1e-9) & (depth_grid < end - 1e-9)
    return missing


def _interp_on_grid(frame: pd.DataFrame, depth_grid: np.ndarray, columns: list[str]) -> pd.DataFrame:
    ordered = frame.sort_values("depth").drop_duplicates("depth")
    result = pd.DataFrame({"depth": depth_grid})
    depth = ordered["depth"].to_numpy(dtype=float)
    for column in columns:
        values = pd.to_numeric(ordered[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(depth) & np.isfinite(values)
        if valid.sum() >= 2:
            interpolated = np.interp(depth_grid, depth[valid], values[valid], left=np.nan, right=np.nan)
            interpolated[_gap_mask(depth_grid, depth[valid])] = np.nan
            result[column] = interpolated
        else:
            result[column] = np.nan
    return result


def _plot_log_curves(ax: plt.Axes, well_frame: pd.DataFrame, curves: list[tuple[str, str, tuple[float, float], str]]) -> None:
    depth = _plot_y(well_frame)
    for column, color, limits, style in curves:
        ax.plot(_norm(well_frame[column], *limits), depth, color=color, linewidth=0.85, linestyle=style)


def _target_columns(target: str) -> tuple[str, str, str, str, str]:
    if target == "PERM":
        return "y_true_log10", "point_prediction_log10", "interval_mean_log10", "lower_ci", "upper_ci"
    return "y_true", "point_prediction", "interval_mean", "lower_ci", "upper_ci"


def _suspect_spans(depth: pd.Series, suspect: np.ndarray) -> list[tuple[float, float]]:
    values = depth.to_numpy(dtype=float)
    spans: list[tuple[float, float]] = []
    start: int | None = None
    for i, flag in enumerate(np.r_[suspect, False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            end = i - 1
            spans.append((float(values[start]), float(values[end])))
            start = None
    return spans


def _plot_target_track(
    ax: plt.Axes,
    target_frame: pd.DataFrame,
    *,
    target: str,
    limits: tuple[float, float],
) -> None:
    y_true, prediction, _mean, lower, upper = _target_columns(target)
    target_frame = target_frame.sort_values("depth")
    depth = _plot_y(target_frame)
    true_x = _norm(target_frame[y_true], *limits)
    prediction_x = _norm(target_frame[prediction], *limits)
    lower_x = _norm(target_frame[lower], *limits)
    upper_x = _norm(target_frame[upper], *limits)
    finite_target = (
        np.isfinite(target_frame[y_true].to_numpy(dtype=float))
        & np.isfinite(target_frame[prediction].to_numpy(dtype=float))
        & np.isfinite(target_frame[lower].to_numpy(dtype=float))
        & np.isfinite(target_frame[upper].to_numpy(dtype=float))
    )
    prediction_suspect = target_frame["prediction_suspect"].to_numpy(dtype=bool) & finite_target
    label_suspect = target_frame["label_suspect"].to_numpy(dtype=bool) & finite_target
    for top, base in _suspect_spans(depth, label_suspect):
        ax.axhspan(top - DEPTH_STEP / 2, base + DEPTH_STEP / 2, color="#FFD84D", alpha=0.30, linewidth=0, zorder=0.5)
    for top, base in _suspect_spans(depth, prediction_suspect):
        ax.axhspan(top - DEPTH_STEP / 2, base + DEPTH_STEP / 2, color="#D62728", alpha=0.18, linewidth=0, zorder=0.6)
    ax.fill_betweenx(depth, lower_x, upper_x, color="#8FA8C2", alpha=0.24)
    ax.fill_betweenx(depth, lower_x, upper_x, where=prediction_suspect, color="#D62728", alpha=0.22)
    ax.plot(true_x, depth, color="#222222", linewidth=0.85)
    ax.plot(prediction_x, depth, color="#1F5AA6", linewidth=0.95)
    if label_suspect.any():
        ax.scatter(
            true_x[label_suspect],
            depth.to_numpy()[label_suspect],
            s=18,
            marker="o",
            facecolor="#FFD84D",
            edgecolor="#8C6D00",
            linewidths=0.6,
            zorder=5,
        )
    if prediction_suspect.any():
        ax.scatter(
            prediction_x[prediction_suspect],
            depth.to_numpy()[prediction_suspect],
            s=16,
            marker="x",
            color="#D62728",
            linewidths=0.9,
            zorder=6,
        )


def _target_limits(target_frame: pd.DataFrame, target: str) -> tuple[float, float]:
    if target == "PHIF":
        return 0.0, 0.4
    if target == "SW":
        return 0.0, 1.0
    return -1.0, 3.0


def _select_depth_window(pred_well: pd.DataFrame) -> tuple[float, float]:
    perm = pred_well[pred_well["target"] == "PERM"].sort_values("depth").reset_index(drop=True)
    suspect = perm["status"].astype(str).eq("存疑").to_numpy()
    if not suspect.any():
        center = float(perm["depth"].median())
        return center - 8.0, center + 8.0
    segments = _suspect_spans(perm["depth"], suspect)
    margin = 3.0
    start, end = max(segments, key=lambda span: span[1] - span[0])
    changed = True
    while changed:
        changed = False
        window_start, window_end = start - margin, end + margin
        for segment_start, segment_end in segments:
            if segment_start <= window_end and segment_end >= window_start:
                new_start = min(start, segment_start)
                new_end = max(end, segment_end)
                if new_start != start or new_end != end:
                    start, end = new_start, new_end
                    changed = True
    return start - margin, end + margin


def _run_label(run_dir: Path) -> str:
    parts = run_dir.resolve().parts
    method = next((part for part in reversed(parts) if part.startswith("M") and part[1:].isdigit()), "method")
    budget = next((part for part in reversed(parts) if part.startswith("N") and part[1:].isdigit()), "N")
    seed = run_dir.name
    return f"{method}_{budget}_{seed}"


def plot_composite(
    run_dir: Path = RUN_DIR,
    output_dir: Path = OUTPUT_DIR,
    *,
    full_well: bool = False,
    scale: float = DEPTH_SCALE_DENOMINATOR,
    compress_missing: bool = False,
    well: object | None = None,
) -> dict[str, Path]:
    run_dir = run_dir.resolve()
    output_dir = output_dir.resolve()
    full_frame, predictions, _ = _read_run(run_dir)
    well = _select_well(predictions) if well is None else well
    well_text = _format_well(well)
    well_frame = full_frame[full_frame["well"].astype(str) == str(well)].sort_values("depth").copy()
    if well_frame.empty:
        well_frame = full_frame[full_frame["well"].astype(float) == float(well)].sort_values("depth").copy()
    pred_well = predictions[predictions["well"].astype(float) == float(well)].copy()
    if full_well:
        depth_min = float(pred_well["depth"].min())
        depth_max = float(pred_well["depth"].max())
    else:
        depth_min, depth_max = _select_depth_window(pred_well)
    depth_min = np.floor(depth_min / DEPTH_STEP) * DEPTH_STEP
    depth_max = np.ceil(depth_max / DEPTH_STEP) * DEPTH_STEP
    well_frame = well_frame[(well_frame["depth"] >= depth_min) & (well_frame["depth"] <= depth_max)].copy()
    pred_well = pred_well[(pred_well["depth"] >= depth_min) & (pred_well["depth"] <= depth_max)].copy()
    depth_grid = np.arange(depth_min, depth_max + DEPTH_STEP / 2, DEPTH_STEP)

    ranges = {
        "GR": (0.0, 150.0),
        "CAL": (6.0, 16.0),
        "SP": (0.0, 100.0),
        "RT": (-1.0, 2.0),  # log10(0.1)-log10(100), header remains RT / ohm.m
        "AC": (600.0, 100.0),
        "CNL": (45.0, -15.0),
        "DEN": (1.85, 2.85),
    }
    well_frame = _interp_on_grid(well_frame, depth_grid, ["GR", "CAL", "SP", "RT", "AC", "CNL", "DEN"])
    target_frames = {}
    for target in TARGETS:
        part = pred_well[pred_well["target"] == target].copy()
        y_true, prediction, mean, lower, upper = _target_columns(target)
        numeric = _interp_on_grid(part, depth_grid, [y_true, prediction, mean, "total_variance"])
        if target == "PHIF":
            center = (
                1.5 * numeric[y_true].to_numpy(dtype=float)
                + 0.5 * numeric[prediction].to_numpy(dtype=float)
            ) / 2.0
            numeric[lower] = center - 0.10
            numeric[upper] = center + 0.10
        else:
            radius = Z_SCORE * np.sqrt(np.clip(numeric["total_variance"].to_numpy(dtype=float), 0.0, None))
            numeric[lower] = numeric[mean].to_numpy(dtype=float) - radius
            numeric[upper] = numeric[mean].to_numpy(dtype=float) + radius
        if target not in {"PERM", "PHIF"}:
            numeric[lower] = np.clip(numeric[lower], 0.0, 1.0)
            numeric[upper] = np.clip(numeric[upper], 0.0, 1.0)
        suspect_depths = part.loc[part["status"].astype(str).isin(["存疑", "瀛樼枒", "suspect"]), "depth"].to_numpy(dtype=float)
        numeric["prediction_suspect"] = False
        if len(suspect_depths):
            mask = np.zeros(len(numeric), dtype=bool)
            grid_depth = numeric["depth"].to_numpy(dtype=float)
            for suspect_depth in suspect_depths:
                mask |= np.abs(grid_depth - suspect_depth) <= DEPTH_STEP / 2 + 1e-9
            numeric.loc[mask, "prediction_suspect"] = True
        numeric["prediction_suspect"] = (numeric[prediction] < numeric[lower]) | (numeric[prediction] > numeric[upper])
        numeric["label_suspect"] = (numeric[y_true] < numeric[lower]) | (numeric[y_true] > numeric[upper])
        numeric["prediction_status"] = np.where(numeric["prediction_suspect"], "存疑", "可信")
        numeric["label_status"] = np.where(numeric["label_suspect"], "存疑", "可信")
        target_frames[target] = numeric
    valid_plot_mask = well_frame[["GR", "CAL", "SP", "RT", "AC", "CNL", "DEN"]].notna().any(axis=1).to_numpy(copy=True)
    for target in TARGETS:
        y_true, prediction, _mean, lower, upper = _target_columns(target)
        valid_plot_mask |= target_frames[target][[y_true, prediction, lower, upper]].notna().any(axis=1).to_numpy()
    if compress_missing:
        if not valid_plot_mask.any():
            raise ValueError("no valid samples remain after removing missing-depth intervals")
        compressed_depth = depth_min + np.arange(int(valid_plot_mask.sum()), dtype=float) * DEPTH_STEP
        well_frame = well_frame.loc[valid_plot_mask].reset_index(drop=True)
        well_frame["plot_depth"] = compressed_depth
        for target in TARGETS:
            target_frames[target] = target_frames[target].loc[valid_plot_mask].reset_index(drop=True)
            target_frames[target]["plot_depth"] = compressed_depth
    else:
        well_frame["plot_depth"] = well_frame["depth"]
        for target in TARGETS:
            target_frames[target]["plot_depth"] = target_frames[target]["depth"]
    target_limits = {target: _target_limits(target_frames[target], target) for target in TARGETS}

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
    axis_depth_min = float(well_frame["plot_depth"].min())
    axis_depth_max = float(well_frame["plot_depth"].max())
    depth_range_m = float(axis_depth_max - axis_depth_min)
    body_height_in = depth_range_m * 100.0 / scale / CM_PER_INCH
    figure_height_in = (HEADER_HEIGHT_IN + body_height_in) / (0.975 - 0.04)
    fig = plt.figure(figsize=(8.6, figure_height_in), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        7,
        height_ratios=[HEADER_HEIGHT_IN, body_height_in],
        width_ratios=[1.05, 0.85, 1.05, 0.55, 1.05, 1.05, 1.05],
        hspace=0.0,
        wspace=0.02,
        left=0.055,
        right=0.985,
        top=0.975,
        bottom=0.04,
    )
    header_axes = [fig.add_subplot(gs[0, i]) for i in range(7)]
    track_axes = [fig.add_subplot(gs[1, i]) for i in range(7)]

    _header_axis(
        header_axes[0],
        [
            ("GR", "gAPI", "0", "150", "blue", "-"),
            ("CAL", "cm", "6", "16", "#C65D4B", "--"),
            ("SP", "mV", "0", "100", "red", "-"),
        ],
    )
    _header_axis(header_axes[1], [("RT", "ohm.m", "0.1", "100", "red", "-")])
    _header_axis(
        header_axes[2],
        [
            ("AC", "us/m", "600", "100", "blue", "-"),
            ("CNL", "%", "45", "-15", "#2CA25F", "--"),
            ("DEN", "g/cm³", "1.85", "2.85", "red", "-."),
        ],
    )
    _depth_header_axis(header_axes[3])
    for axis, target in zip(header_axes[4:], TARGETS):
        if target == "PERM":
            _target_header_axis(axis, "PERM", "mD", "0.1", "1000")
        elif target == "PHIF":
            _target_header_axis(axis, "PHIF", "fraction", "0.0", "0.4")
        else:
            _target_header_axis(axis, "SW", "fraction", "0.0", "1.0")

    for ax in track_axes:
        _style_track(ax, axis_depth_min, axis_depth_max, scale=scale)
    _style_track(
        track_axes[3],
        axis_depth_min,
        axis_depth_max,
        show_y=True,
        scale=scale,
        depth_label_frame=well_frame if compress_missing else None,
    )
    _add_log_grid(track_axes[1], -1.0, 2.0)
    _add_log_grid(track_axes[5], -1.0, 3.0)

    _plot_log_curves(
        track_axes[0],
        well_frame,
        [
            ("GR", "blue", ranges["GR"], "-"),
            ("CAL", "#C65D4B", ranges["CAL"], "--"),
            ("SP", "red", ranges["SP"], "-"),
        ],
    )
    _plot_log_curves(track_axes[1], well_frame, [("RT", "red", ranges["RT"], "-")])
    _plot_log_curves(
        track_axes[2],
        well_frame,
        [
            ("AC", "blue", ranges["AC"], "-"),
            ("CNL", "#2CA25F", ranges["CNL"], "--"),
            ("DEN", "red", ranges["DEN"], "-."),
        ],
    )
    track_axes[3].set_xlim(0, 1)
    track_axes[3].set_xticks([])
    track_axes[3].tick_params(axis="y", labelsize=7)
    for ax, target in zip(track_axes[4:], TARGETS):
        _plot_target_track(ax, target_frames[target], target=target, limits=target_limits[target])
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
    interval_label = "fullwell" if full_well else "suspect"
    if compress_missing:
        interval_label = f"{interval_label}_compressed"
    scale_label = f"scale1to{int(scale):d}" if float(scale).is_integer() else f"scale1to{scale:g}"
    stem = f"field_{run_label}_well_{well_text}_composite_{interval_label}_interpretation_legend_v2_ci95_phif_3to1_label_prediction_pm0p1_{scale_label}_depth_{depth_min:.3f}_{depth_max:.3f}".replace(".", "p")
    paths = {
        "pdf": output_dir / f"{stem}.pdf",
        "svg": output_dir / f"{stem}.svg",
        "png": output_dir / f"{stem}.png",
        "data": output_dir / f"{stem}_plot_data.csv",
    }
    # Keep the canvas size fixed so the requested vertical logging scale
    # (e.g. 1:500) is preserved in the exported vector files.
    fig.savefig(paths["pdf"])
    fig.savefig(paths["svg"])
    fig.savefig(paths["png"], dpi=260)
    plt.close(fig)

    export = well_frame.copy()
    export.insert(0, "well", well_text)
    export.insert(2, "depth_step_m", DEPTH_STEP)
    export.insert(3, "run_label", run_label)
    export.insert(4, "run_dir", str(run_dir))
    for target, frame in target_frames.items():
        renamed = frame.rename(columns={column: f"{target}_{column}" for column in frame.columns if column != "depth"})
        export = export.merge(renamed, on="depth", how="left")
    export.to_csv(paths["data"], index=False, encoding="utf-8-sig")
    print(f"well={well_text}")
    print(f"run={run_label}")
    print(f"full_well={full_well}")
    print(f"compress_missing={compress_missing}")
    print(f"scale=1:{scale:g}")
    print(f"plotted_rows={len(well_frame)}")
    print(f"actual_depth_range={depth_min:.3f}-{depth_max:.3f}")
    print(f"plot_depth_range={axis_depth_min:.3f}-{axis_depth_max:.3f}")
    for target, frame in target_frames.items():
        print(
            f"{target}: rows={len(frame)}, "
            f"prediction_suspect={int(frame['prediction_suspect'].sum())}, "
            f"label_suspect={int(frame['label_suspect'].sum())}"
        )
    for path in paths.values():
        print(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot field suspect well composite using a well-log interpretation template.")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR, help="Experiment run directory containing predictions.csv and resolved_config.yaml.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for PDF/SVG/PNG/CSV outputs.")
    parser.add_argument("--full-well", action="store_true", help="Plot the full selected well interval instead of the suspect interval.")
    parser.add_argument("--scale", type=float, default=DEPTH_SCALE_DENOMINATOR, help="Vertical scale denominator, e.g. 500 for 1:500.")
    parser.add_argument("--compress-missing", action="store_true", help="Remove missing-depth intervals from the plotting coordinate and connect valid samples directly.")
    parser.add_argument("--well", help="Well identifier to plot; omit to select the most suspect PERM well.")
    args = parser.parse_args()
    plot_composite(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        full_well=args.full_well,
        scale=args.scale,
        compress_missing=args.compress_missing,
        well=args.well,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
