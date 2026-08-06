"""Analyze and plot the completed M2--M5 uncertainty supplementary matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = REPO / "outputs" / "supplementary" / "summary" / "supplementary_metrics_summary.csv"
DEFAULT_OUTPUT = REPO / "figure" / "uncertainty_supplement"
METHODS = ("M2", "M3", "M4", "M5")
COLORS = {"M2": "#4C78A8", "M3": "#F58518", "M4": "#54A24B", "M5": "#7A5195"}
LABELS = {"M2": "MC Dropout", "M3": "BNN", "M4": "Simple fusion", "M5": "Conservative fusion"}


def _load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[(frame["section"] == "uncertainty") & frame["method"].isin(METHODS)].copy()
    if frame.empty:
        raise ValueError(f"no uncertainty supplementary results in {path}")
    return frame


def _wide(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    result = frame[frame["metric"].isin(metrics)].pivot_table(
        index=["dataset", "target", "budget_N", "method"], columns="metric", values="mean"
    ).reset_index()
    return result


def _save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(frame: pd.DataFrame, output: Path) -> None:
    wide = _wide(frame, ["rmse"])
    targets = sorted(wide["target"].unique())
    datasets = ("field", "spwla", "forward")
    fig, axes = plt.subplots(len(datasets), len(targets), figsize=(3.35 * len(targets), 2.8 * len(datasets)), sharex=True)
    axes = np.atleast_2d(axes)
    for row, dataset in enumerate(datasets):
        for column, target in enumerate(targets):
            ax = axes[row, column]
            part = wide[(wide.dataset == dataset) & (wide.target == target)]
            if column == 0:
                ax.set_ylabel(f"{dataset}\nRMSE", fontsize=13)
            if part.empty:
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.text(.5, .5, "Not available", transform=ax.transAxes, ha="center", va="center", color="#777777", fontsize=11)
                if row == 0:
                    ax.set_title(target, fontsize=16, fontweight="bold")
                continue
            for method in METHODS:
                values = part[part.method == method].sort_values("budget_N")
                if not values.empty:
                    ax.plot(values.budget_N, values.rmse, marker="o", linewidth=1.5, markersize=3.5, color=COLORS[method], label=LABELS[method])
            ax.set_xscale("symlog", linthresh=50)
            ax.grid(alpha=.25, linewidth=.45)
            if row == 0:
                ax.set_title(target, fontsize=16, fontweight="bold")
            ax.tick_params(axis="both", labelsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("M2--M5 uncertainty ablation: point-prediction RMSE", y=.992, fontsize=16, fontweight="bold")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .012), ncol=4, frameon=False, fontsize=11)
    fig.supxlabel("Label budget N", y=.105, fontsize=14)
    fig.subplots_adjust(left=.085, right=.99, bottom=.18, top=.83, wspace=.25, hspace=.28)
    _save(fig, output, "fig_u1_ablation_rmse_large_labels")


def plot_risk(frame: pd.DataFrame, output: Path) -> None:
    wide = _wide(frame, ["risk_precision", "risk_recall", "risk_error_enrichment"])
    summary = wide.groupby("method", as_index=False)[["risk_precision", "risk_recall", "risk_error_enrichment"]].mean().set_index("method").reindex(METHODS)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    positions = np.arange(len(METHODS)); width = .34
    axes[0].bar(positions - width / 2, summary.risk_precision, width, color="#4C78A8", label="Precision")
    axes[0].bar(positions + width / 2, summary.risk_recall, width, color="#E45756", label="Recall")
    axes[0].axhline(.10, color="#666666", linestyle="--", linewidth=.8, label="Random (10%)")
    axes[0].set_ylim(0, max(.30, float(summary[["risk_precision", "risk_recall"]].max().max()) * 1.25))
    axes[0].set_ylabel("Top-10% high-error identification")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(positions, summary.risk_error_enrichment, color=[COLORS[m] for m in METHODS])
    axes[1].axhline(1.0, color="#666666", linestyle="--", linewidth=.8)
    axes[1].set_ylabel("Risk-group MAE / overall MAE")
    axes[1].set_ylim(0, max(1.6, float(summary.risk_error_enrichment.max()) * 1.18))
    for ax in axes:
        ax.set_xticks(positions, METHODS); ax.grid(axis="y", alpha=.25, linewidth=.45)
    fig.suptitle("Risk-warning quality: uncertainty top 10% vs error top 10%", y=.99)
    fig.tight_layout()
    _save(fig, output, "fig_u2_risk_identification")


def plot_calibration(frame: pd.DataFrame, output: Path) -> None:
    wide = _wide(frame, ["raw_picp", "calibrated_picp", "raw_nmpiw", "calibrated_nmpiw"])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    for method in METHODS:
        part = wide[wide.method == method]
        axes[0].scatter(part.raw_nmpiw, part.raw_picp, color=COLORS[method], marker="o", alpha=.6, s=28, label=f"{method} raw")
        axes[0].scatter(part.calibrated_nmpiw, part.calibrated_picp, color=COLORS[method], marker="x", alpha=.8, s=34, label=f"{method} calibrated")
    axes[0].axhline(.95, color="#222222", linestyle="--", linewidth=.9, label="95% target")
    axes[0].set_xlabel("NMPIW"); axes[0].set_ylabel("PICP"); axes[0].set_ylim(0.55, 1.02); axes[0].grid(alpha=.25, linewidth=.45)
    method_summary = wide.groupby("method", as_index=False)[["raw_picp", "calibrated_picp", "raw_nmpiw", "calibrated_nmpiw"]].mean().set_index("method").reindex(METHODS)
    y = np.arange(len(METHODS))
    axes[1].scatter(method_summary.raw_picp, y, color="#4C78A8", marker="o", label="Raw PICP")
    axes[1].scatter(method_summary.calibrated_picp, y, color="#E45756", marker="x", s=42, label="Calibrated PICP")
    for index, (_, row) in enumerate(method_summary.iterrows()):
        axes[1].plot([row.raw_picp, row.calibrated_picp], [index, index], color="#999999", linewidth=.8)
    axes[1].axvline(.95, color="#222222", linestyle="--", linewidth=.9)
    axes[1].set_yticks(y, METHODS); axes[1].set_xlim(.75, 1.01); axes[1].set_xlabel("Mean PICP"); axes[1].grid(axis="x", alpha=.25, linewidth=.45)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Validation-only conformal recalibration: coverage--width trade-off", y=.99)
    fig.tight_layout()
    _save(fig, output, "fig_u3_conformal_calibration")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = _load(args.summary)
    plot_ablation(frame, args.output)
    plot_risk(frame, args.output)
    plot_calibration(frame, args.output)
    _wide(frame, ["rmse", "raw_picp", "calibrated_picp", "raw_nmpiw", "calibrated_nmpiw", "risk_precision", "risk_recall", "risk_error_enrichment"]).to_csv(args.output / "uncertainty_analysis_overview.csv", index=False)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
