from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bnn_inversion.publication_style import FIGURE_WIDTH_IN, PALETTE, apply_sci_style
from bnn_inversion.uncertainty.metrics import (
    active_learning_efficiency_metrics,
    interval_metrics,
    point_metrics,
    trust_metrics,
    uncertainty_metrics,
)


NUMERIC_COLUMNS = (
    "rmse",
    "mae",
    "r2",
    "picp",
    "mpiw",
    "nmpiw",
    "interval_score",
    "uce",
    "epistemic_ratio",
    "nll",
    "uncertainty_error_spearman",
)


def _parse_matrix_identity(metrics_path: Path) -> dict[str, object] | None:
    parts = metrics_path.parts
    if len(parts) < 4:
        return None
    method, budget, seed = parts[-4], parts[-3], parts[-2]
    if not (method.startswith("M") and method[1:].isdigit()):
        return None
    if not (budget.startswith("N") and budget[1:].isdigit()):
        return None
    if not (seed.startswith("seed") and seed[4:].isdigit()):
        return None
    dataset = ""
    for token in parts:
        lowered = token.lower()
        if lowered in {"field", "spwla", "forward"}:
            dataset = lowered
    return {
        "dataset": dataset,
        "method": method,
        "budget_N": int(budget[1:]),
        "seed": int(seed[4:]),
    }


def _metric_space_for_target(target: object, frame: pd.DataFrame) -> str:
    if str(target) == "PERM" and {
        "y_true_log10",
        "point_prediction_log10",
    }.issubset(frame.columns):
        return "log10"
    return "linear"


def _recompute_prediction_metrics(predictions_path: Path) -> pd.DataFrame:
    header = pd.read_csv(predictions_path, nrows=0)
    desired = {
        "target",
        "y_true",
        "point_prediction",
        "interval_mean",
        "lower",
        "upper",
        "epistemic_variance",
        "total_variance",
        "y_true_log10",
        "point_prediction_log10",
        "interval_mean_log10",
        "lower_log10",
        "upper_log10",
    }
    usecols = [column for column in header.columns if column in desired]
    predictions = pd.read_csv(predictions_path, usecols=usecols)
    if predictions.empty or "target" not in predictions:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for target, group in predictions.groupby("target", sort=False):
        metric_space = _metric_space_for_target(target, group)
        y_column = "y_true_log10" if metric_space == "log10" else "y_true"
        point_column = (
            "point_prediction_log10" if metric_space == "log10" else "point_prediction"
        )
        row: dict[str, object] = {"target": target, "metric_space": metric_space}
        row.update(point_metrics(group[y_column].to_numpy(), group[point_column].to_numpy()))
        lower_column = "lower_log10" if metric_space == "log10" and "lower_log10" in group else "lower"
        upper_column = "upper_log10" if metric_space == "log10" and "upper_log10" in group else "upper"
        mean_column = (
            "interval_mean_log10"
            if metric_space == "log10" and "interval_mean_log10" in group
            else "interval_mean"
        )
        if {lower_column, upper_column}.issubset(group.columns):
            row.update(
                interval_metrics(
                    group[y_column].to_numpy(),
                    group[lower_column].to_numpy(),
                    group[upper_column].to_numpy(),
                    confidence=0.95,
                )
            )
        if {mean_column, "epistemic_variance", "total_variance"}.issubset(
            group.columns
        ):
            row.update(
                uncertainty_metrics(
                    target=group[y_column].to_numpy(),
                    prediction=group[mean_column].to_numpy(),
                    epistemic=group["epistemic_variance"].to_numpy(),
                    total=group["total_variance"].to_numpy(),
                )
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _merge_metric_rows(stored: pd.DataFrame, recomputed: pd.DataFrame) -> pd.DataFrame:
    if recomputed.empty:
        return stored
    keys = ["target", "metric_space"]
    if not set(keys).issubset(stored.columns):
        return recomputed
    merged = stored.merge(
        recomputed,
        on=keys,
        how="outer",
        suffixes=("_stored", ""),
    )
    for column in NUMERIC_COLUMNS:
        stored_column = f"{column}_stored"
        if column not in merged and stored_column in merged:
            merged[column] = merged[stored_column]
        elif stored_column in merged:
            merged[column] = merged[column].combine_first(merged[stored_column])
    drop_columns = [column for column in merged.columns if column.endswith("_stored")]
    return merged.drop(columns=drop_columns)


def _collect_metrics(root: Path, *, recompute_predictions: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(root.rglob("metrics.csv")):
        identity = _parse_matrix_identity(metrics_path)
        if identity is None:
            continue
        stored = pd.read_csv(metrics_path)
        predictions_path = metrics_path.with_name("predictions.csv")
        recomputed = (
            _recompute_prediction_metrics(predictions_path)
            if recompute_predictions and predictions_path.exists()
            else pd.DataFrame()
        )
        metrics = _merge_metric_rows(stored, recomputed)
        for _, row in metrics.iterrows():
            record = {
                **identity,
                "target": row.get("target"),
                "metric_space": row.get("metric_space", ""),
                "metrics_path": metrics_path.as_posix(),
            }
            for column in NUMERIC_COLUMNS:
                record[column] = pd.to_numeric(
                    pd.Series([row.get(column, pd.NA)]), errors="coerce"
                ).iloc[0]
            rows.append(record)
    return pd.DataFrame(rows)


def _collect_trust(root: Path, *, recompute_predictions: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(root.rglob("metrics.csv")):
        identity = _parse_matrix_identity(metrics_path)
        if identity is None:
            continue
        trust_path = metrics_path.with_name("trust_metrics.csv")
        predictions_path = metrics_path.with_name("predictions.csv")
        if trust_path.exists():
            trust = pd.read_csv(trust_path)
        elif recompute_predictions and predictions_path.exists():
            header = pd.read_csv(predictions_path, nrows=0)
            desired = {"target", "y_true", "point_prediction", "status"}
            if "interval_width" in header.columns:
                desired.add("interval_width")
            elif {"lower", "upper"}.issubset(header.columns):
                desired.update({"lower", "upper"})
            usecols = [column for column in header.columns if column in desired]
            if "status" not in usecols:
                continue
            predictions = pd.read_csv(predictions_path, usecols=usecols)
            if "interval_width" not in predictions and {"lower", "upper"}.issubset(
                predictions.columns
            ):
                predictions["interval_width"] = predictions["upper"] - predictions["lower"]
            trust = pd.DataFrame(
                [
                    {"target": target, **trust_metrics(group)}
                    for target, group in predictions.groupby("target", sort=False)
                ]
            )
        else:
            continue
        if (
            recompute_predictions
            and predictions_path.exists()
            and not {"trusted_mae", "suspect_mae", "mae_gap"}.issubset(trust.columns)
        ):
            header = pd.read_csv(predictions_path, nrows=0)
            desired = {"target", "y_true", "point_prediction", "status"}
            if "interval_width" in header.columns:
                desired.add("interval_width")
            elif {"lower", "upper"}.issubset(header.columns):
                desired.update({"lower", "upper"})
            usecols = [column for column in header.columns if column in desired]
            if "status" not in usecols:
                continue
            predictions = pd.read_csv(predictions_path, usecols=usecols)
            if "interval_width" not in predictions and {"lower", "upper"}.issubset(
                predictions.columns
            ):
                predictions["interval_width"] = predictions["upper"] - predictions["lower"]
            trust = pd.DataFrame(
                [
                    {"target": target, **trust_metrics(group)}
                    for target, group in predictions.groupby("target", sort=False)
                ]
            )
        for _, row in trust.iterrows():
            rows.append(
                {
                    **identity,
                    "target": row.get("target"),
                    "trusted_rate": row.get("trusted_rate", pd.NA),
                    "suspect_rate": row.get("suspect_rate", pd.NA),
                    "trusted_mae": row.get("trusted_mae", pd.NA),
                    "suspect_mae": row.get("suspect_mae", pd.NA),
                    "mae_gap": row.get("mae_gap", pd.NA),
                    "trust_path": trust_path.as_posix()
                    if trust_path.exists()
                    else predictions_path.as_posix(),
                }
            )
    return pd.DataFrame(rows)


def _active_efficiency(metrics: pd.DataFrame, *, baseline_method: str = "M6") -> pd.DataFrame:
    columns = [
        "dataset",
        "target",
        "method",
        "baseline_method",
        "delta_rmse_pct",
        "relative_improvement_pct",
        "aulc",
        "baseline_aulc",
        "label_saving_rate",
    ]
    if metrics.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        metrics.groupby(["dataset", "target", "method", "budget_N"], as_index=False)["rmse"]
        .mean()
        .dropna(subset=["rmse"])
    )
    rows: list[dict[str, object]] = []
    for (dataset, target), target_frame in grouped.groupby(["dataset", "target"], sort=True):
        fallback = "M5" if (target_frame["method"] == "M5").any() else baseline_method
        target_baseline = baseline_method if (target_frame["method"] == baseline_method).any() else fallback
        baseline = target_frame[target_frame["method"] == target_baseline]
        for method, candidate in target_frame.groupby("method", sort=True):
            threshold = (
                float(baseline["rmse"].iloc[-1])
                if not baseline.empty
                else float(candidate["rmse"].iloc[-1])
            )
            values = active_learning_efficiency_metrics(
                candidate_budgets=candidate["budget_N"].to_numpy(),
                candidate_rmse=candidate["rmse"].to_numpy(),
                baseline_budgets=baseline["budget_N"].to_numpy()
                if not baseline.empty
                else None,
                baseline_rmse=baseline["rmse"].to_numpy() if not baseline.empty else None,
                threshold=threshold if math.isfinite(threshold) else None,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "method": method,
                    "baseline_method": target_baseline,
                    **values,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def evaluate_experiment_tree(
    root: Path | str, *, recompute_predictions: bool = False
) -> dict[str, pd.DataFrame]:
    root_path = Path(root)
    metrics = _collect_metrics(root_path, recompute_predictions=recompute_predictions)
    trust = _collect_trust(root_path, recompute_predictions=recompute_predictions)
    active = _active_efficiency(metrics)
    summary = (
        metrics.groupby(["dataset", "method", "budget_N", "target"], as_index=False)[
            list(NUMERIC_COLUMNS)
        ]
        .mean(numeric_only=True)
        if not metrics.empty
        else pd.DataFrame()
    )
    return {
        "metrics": metrics,
        "summary": summary,
        "trust": trust,
        "active_efficiency": active,
    }


def _save_figure(fig: plt.Figure, output: Path, stem: str) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f"{stem}.pdf", output / f"{stem}.svg"]
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


def _method_order(frame: pd.DataFrame) -> list[str]:
    methods = sorted(frame["method"].dropna().astype(str).unique())
    return sorted(methods, key=lambda value: int(value[1:]) if value.startswith("M") and value[1:].isdigit() else 999)


def _plot_metric_lines(
    ax: plt.Axes, frame: pd.DataFrame, metric: str, *, ylabel: str
) -> None:
    colors = [PALETTE["primary"], PALETTE["negative"], PALETTE["trusted"], PALETTE["secondary"], "#6F6F6F"]
    for index, method in enumerate(_method_order(frame)):
        part = frame[frame["method"] == method].sort_values("budget_N")
        if part.empty or metric not in part:
            continue
        ax.plot(
            part["budget_N"],
            part[metric],
            marker="o",
            label=method,
            color=colors[index % len(colors)],
        )
    ax.set_xlabel("Labeled samples")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22)


def write_metric_figures(
    tables: Mapping[str, pd.DataFrame], output_dir: Path | str
) -> list[Path]:
    apply_sci_style()
    output = Path(output_dir)
    paths: list[Path] = []
    metrics = tables.get("metrics", pd.DataFrame())
    if not metrics.empty:
        aggregate = (
            metrics.groupby(["method", "budget_N"], as_index=False)[
                ["rmse", "mae", "r2", "picp", "nmpiw", "uce", "uncertainty_error_spearman"]
            ]
            .mean(numeric_only=True)
            .sort_values(["method", "budget_N"])
        )
        fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_IN, 2.15), constrained_layout=True)
        _plot_metric_lines(axes[0], aggregate, "rmse", ylabel="RMSE")
        _plot_metric_lines(axes[1], aggregate, "mae", ylabel="MAE")
        _plot_metric_lines(axes[2], aggregate, "r2", ylabel="$R^2$")
        axes[0].legend(frameon=False, ncol=2)
        paths.extend(_save_figure(fig, output, "metric_point_accuracy"))

        fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_IN, 2.15), constrained_layout=True)
        axes[0].scatter(aggregate["nmpiw"], aggregate["picp"], s=28, color=PALETTE["primary"], alpha=0.82)
        axes[0].axhline(0.95, color=PALETTE["negative"], linestyle="--", linewidth=0.9)
        axes[0].set_xlabel("NMPIW")
        axes[0].set_ylabel("PICP")
        axes[0].grid(True, alpha=0.22)
        _plot_metric_lines(axes[1], aggregate, "uce", ylabel="UCE")
        _plot_metric_lines(
            axes[2],
            aggregate,
            "uncertainty_error_spearman",
            ylabel="Spearman $\\rho$",
        )
        paths.extend(_save_figure(fig, output, "metric_uncertainty_quality"))

    trust = tables.get("trust", pd.DataFrame())
    if not trust.empty:
        aggregate = (
            trust.groupby(["method", "budget_N"], as_index=False)[
                ["trusted_mae", "suspect_mae", "mae_gap", "trusted_rate"]
            ]
            .mean(numeric_only=True)
            .sort_values(["method", "budget_N"])
        )
        fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_IN, 2.15), constrained_layout=True)
        _plot_metric_lines(axes[0], aggregate, "trusted_mae", ylabel="Trusted MAE")
        _plot_metric_lines(axes[1], aggregate, "suspect_mae", ylabel="Suspect MAE")
        _plot_metric_lines(axes[2], aggregate, "mae_gap", ylabel="MAE gap")
        axes[0].legend(frameon=False, ncol=2)
        paths.extend(_save_figure(fig, output, "metric_trust_mae_gap"))

    active = tables.get("active_efficiency", pd.DataFrame())
    if not active.empty:
        plot = active.dropna(subset=["aulc"]).copy()
        if not plot.empty:
            fig, axes = plt.subplots(1, 3, figsize=(FIGURE_WIDTH_IN, 2.15), constrained_layout=True)
            for ax, metric, ylabel in [
                (axes[0], "delta_rmse_pct", "$\\Delta$RMSE (%)"),
                (axes[1], "relative_improvement_pct", "Improvement (%)"),
                (axes[2], "label_saving_rate", "Label saving"),
            ]:
                values = plot.groupby("method", as_index=False)[metric].mean(numeric_only=True)
                ax.bar(values["method"], values[metric], color=PALETTE["primary"], alpha=0.86)
                ax.set_xlabel("Method")
                ax.set_ylabel(ylabel)
                ax.grid(axis="y", alpha=0.22)
            paths.extend(_save_figure(fig, output, "metric_active_efficiency"))
    return paths


def write_tables(tables: Mapping[str, pd.DataFrame], output_dir: Path | str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/metric_evaluation"))
    parser.add_argument("--figures", type=Path, default=Path("figure/metrics"))
    parser.add_argument(
        "--recompute-predictions",
        action="store_true",
        help="Recompute NMPIW/NLL/trust MAE from predictions.csv. Slower on large output trees.",
    )
    args = parser.parse_args(argv)

    tables = evaluate_experiment_tree(
        args.root, recompute_predictions=args.recompute_predictions
    )
    write_tables(tables, args.output)
    paths = write_metric_figures(tables, args.figures)
    print(f"evaluated {len(tables['metrics'])} metric rows")
    print(f"wrote tables to {args.output}")
    print(f"wrote {len(paths)} figure files to {args.figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
