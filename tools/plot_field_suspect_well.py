from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from bnn_inversion.data.adapters import load_dataset
from bnn_inversion.publication_style import FIGURE_WIDTH_IN, apply_sci_style


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO / "outputs" / "field_m1_m9_multiseed" / "M9" / "N50" / "seed2"
OUTPUT_DIR = REPO / "figure" / "field_suspect_well"


def _load_run_frame(run_dir: Path, *, target: str) -> pd.DataFrame:
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    canonical = load_dataset(
        config["data"]["dataset"],
        config["data"]["root"],
        feature_profile=config["data"].get("feature_profile"),
    )
    meta = canonical.frame[[canonical.well_column, canonical.depth_column]].reset_index()
    meta = meta.rename(
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
        "interval_width",
        "status",
        "y_true_log10",
        "point_prediction_log10",
        "interval_mean_log10",
        "lower_log10",
        "upper_log10",
    ]
    predictions = pd.read_csv(run_dir / "predictions.csv", usecols=lambda c: c in usecols)
    predictions = predictions[predictions["target"] == target].copy()
    merged = predictions.merge(meta, on="source_index", how="left", validate="many_to_one")
    if merged["well"].isna().any():
        raise ValueError("some source_index values could not be mapped back to WELLNUM")
    return merged


def _select_well(frame: pd.DataFrame) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["absolute_error"] = (work["point_prediction"] - work["y_true"]).abs()
    summary = (
        work.groupby("well", as_index=False)
        .agg(
            rows=("source_index", "count"),
            suspect_count=("status", lambda s: int((s.astype(str) == "存疑").sum())),
            suspect_mae=(
                "absolute_error",
                lambda s: float(
                    s.loc[work.loc[s.index, "status"].astype(str).eq("存疑")].mean()
                )
                if work.loc[s.index, "status"].astype(str).eq("存疑").any()
                else np.nan,
            ),
            trusted_mae=(
                "absolute_error",
                lambda s: float(
                    s.loc[work.loc[s.index, "status"].astype(str).eq("可信")].mean()
                )
                if work.loc[s.index, "status"].astype(str).eq("可信").any()
                else np.nan,
            ),
        )
        .dropna(subset=["suspect_mae"])
    )
    if summary.empty:
        raise ValueError("no well contains suspect predictions")
    summary["mae_gap"] = summary["suspect_mae"] - summary["trusted_mae"]
    summary = summary.sort_values(
        ["suspect_count", "mae_gap", "suspect_mae"],
        ascending=[False, False, False],
        kind="stable",
    )
    well = summary.iloc[0]["well"]
    selected = work[work["well"] == well].sort_values("depth", kind="stable").reset_index(drop=True)
    return well, selected, summary


def _target_columns(target: str) -> tuple[str, str, str, str, str]:
    if target == "PERM":
        return (
            "y_true_log10",
            "interval_mean_log10",
            "lower_log10",
            "upper_log10",
            "log10(PERM / mD)",
        )
    return "y_true", "interval_mean", "lower", "upper", target


def _mark_suspect_spans(ax: plt.Axes, suspect_mask: np.ndarray) -> None:
    if not suspect_mask.any():
        return
    start: int | None = None
    for index, value in enumerate(np.r_[suspect_mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            ax.axvspan(start - 0.5, index - 0.5, color="#C65D4B", alpha=0.16, lw=0)
            start = None


def _suspect_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            segments.append((start, index - 1))
            start = None
    return segments


def _plot_panel(
    ax: plt.Axes,
    selected: pd.DataFrame,
    *,
    y_true: str,
    mean: str,
    lower: str,
    upper: str,
    y_label: str,
    title: str,
    xlim: tuple[int, int] | None = None,
) -> None:
    x = np.arange(len(selected))
    suspect = selected["status"].astype(str).eq("存疑").to_numpy()
    trusted = ~suspect
    _mark_suspect_spans(ax, suspect)
    lower_values = selected[lower].to_numpy(dtype=float)
    upper_values = selected[upper].to_numpy(dtype=float)
    ax.fill_between(
        x,
        lower_values,
        upper_values,
        color="#8FA8C2",
        alpha=0.24,
        label="Prediction interval",
    )
    ax.fill_between(
        x,
        lower_values,
        upper_values,
        where=suspect,
        color="#C65D4B",
        alpha=0.34,
        label="Suspect interval",
    )
    ax.plot(x, selected[y_true], color="#2E2E2E", linewidth=0.95, label="True")
    ax.plot(x, selected[mean], color="#3B5B92", linewidth=1.05, label="Predictive mean")
    if suspect.any():
        ax.scatter(
            x[suspect],
            selected.loc[suspect, mean],
            s=22,
            color="#C65D4B",
            marker="x",
            linewidths=1.0,
            label="Suspect prediction",
            zorder=5,
        )
    ax.set_ylabel(y_label)
    ax.set_title(title, loc="left", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    if xlim is not None:
        ax.set_xlim(*xlim)


def plot_suspect_well(
    run_dir: Path = DEFAULT_RUN,
    *,
    target: str = "PERM",
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _load_run_frame(run_dir, target=target)
    well, selected, summary = _select_well(frame)
    y_true, mean, lower, upper, y_label = _target_columns(target)
    selected = selected.dropna(subset=[y_true, mean, lower, upper]).reset_index(drop=True)
    x = np.arange(len(selected))
    suspect = selected["status"].astype(str).eq("存疑").to_numpy()
    segments = _suspect_segments(suspect)
    if segments:
        longest = max(segments, key=lambda item: item[1] - item[0])
        center = (longest[0] + longest[1]) // 2
        zoom_start = max(0, center - 180)
        zoom_end = min(len(selected) - 1, center + 180)
    else:
        zoom_start, zoom_end = 0, min(len(selected) - 1, 360)

    apply_sci_style()
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(FIGURE_WIDTH_IN, 5.25),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )
    well_text = str(int(well)) if float(well).is_integer() else str(well)
    _plot_panel(
        axes[0],
        selected,
        y_true=y_true,
        mean=mean,
        lower=lower,
        upper=upper,
        y_label=y_label,
        title=f"(a) Full well {well_text}: {target}",
    )
    _plot_panel(
        axes[1],
        selected,
        y_true=y_true,
        mean=mean,
        lower=lower,
        upper=upper,
        y_label=y_label,
        title=f"(b) Zoomed suspect interval: samples {zoom_start}-{zoom_end}",
        xlim=(zoom_start, zoom_end),
    )
    axes[1].set_xlabel("Sample order along selected well")
    handles, labels = axes[1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        frameon=False,
        ncol=5,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
    )

    stem = f"field_well_{well_text}_{target}_suspect_intervals".replace("/", "_").replace("\\", "_")
    paths = {
        "pdf": output_dir / f"{stem}.pdf",
        "svg": output_dir / f"{stem}.svg",
        "png": output_dir / f"{stem}.png",
        "data": output_dir / f"{stem}.csv",
        "summary": output_dir / "field_suspect_well_candidates.csv",
        "intervals": output_dir / f"{stem}_suspect_segments.csv",
    }
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["png"], dpi=260, bbox_inches="tight")
    plt.close(fig)
    selected.to_csv(paths["data"], index=False, encoding="utf-8-sig")
    summary.to_csv(paths["summary"], index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "segment_id": index + 1,
                "start_sample": start,
                "end_sample": end,
                "start_depth": selected.loc[start, "depth"],
                "end_depth": selected.loc[end, "depth"],
                "suspect_count": end - start + 1,
            }
            for index, (start, end) in enumerate(segments)
        ]
    ).to_csv(paths["intervals"], index=False, encoding="utf-8-sig")
    print(f"selected_well={well}")
    print(f"target={target}")
    print(f"suspect_count={int(suspect.sum())}")
    print(f"rows={len(selected)}")
    for path in paths.values():
        print(path)
    return paths


def main() -> int:
    plot_suspect_well()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
