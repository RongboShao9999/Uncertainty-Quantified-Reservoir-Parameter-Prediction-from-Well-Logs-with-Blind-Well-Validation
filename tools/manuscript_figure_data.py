from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = {0, 1, 2, 3, 4}


def add_budget_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    budget = pd.to_numeric(result["budget_N"], errors="coerce").astype("Int64")
    result["final_budget"] = budget
    active_candidate = result["section"].eq("active") & result["method"].isin(["M6", "M7", "M8", "M9"])
    result["initial_budget"] = budget
    result.loc[active_candidate, "initial_budget"] = 100
    return result


def _validate_seeds(frame: pd.DataFrame, keys: list[str]) -> None:
    failures = []
    for group_key, group in frame.groupby(keys, dropna=False):
        found = set(pd.to_numeric(group["seed"], errors="coerce").dropna().astype(int))
        if found != SEEDS:
            failures.append(f"{group_key}: {sorted(found)}")
    if failures:
        raise ValueError("incomplete seed coverage: " + "; ".join(failures[:10]))


def _paired(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], comparison: str) -> pd.DataFrame:
    left_cols = keys + ["rmse", "metric_space", "metrics_path"]
    right_cols = keys + ["rmse", "metrics_path"]
    merged = left[left_cols].rename(columns={"rmse": "baseline_rmse", "metrics_path": "baseline_source_path"}).merge(
        right[right_cols].rename(columns={"rmse": "candidate_rmse", "metrics_path": "candidate_source_path"}),
        on=keys,
        validate="one_to_one",
    )
    merged["improvement_pct"] = (merged.baseline_rmse - merged.candidate_rmse) / merged.baseline_rmse * 100.0
    merged["comparison"] = comparison
    merged["source_path"] = merged.baseline_source_path + " | " + merged.candidate_source_path
    return merged


def _representative_predictions(repo: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    candidates = metrics.query("section == 'active' and dataset == 'field' and method == 'M5' and final_budget == 125").copy()
    composite = candidates.groupby("seed", as_index=False).rmse.mean()
    median = composite.rmse.median()
    seed = int(composite.assign(distance=(composite.rmse - median).abs()).sort_values(["distance", "seed"]).iloc[0].seed)
    path_text = candidates[candidates.seed == seed].iloc[0].metrics_path
    path = repo / Path(path_text).parent / "predictions.csv"
    columns = ["source_index", "target", "y_true", "interval_mean", "lower", "upper", "total_variance", "y_true_log10", "interval_mean_log10", "lower_log10", "upper_log10"]
    predictions = pd.read_csv(path, usecols=lambda name: name in columns)
    sampled = []
    for _, group in predictions.groupby("target", sort=False):
        group = group.sort_values("source_index", kind="stable")
        index = np.linspace(0, len(group) - 1, min(500, len(group)), dtype=int)
        sampled.append(group.iloc[index])
    result = pd.concat(sampled, ignore_index=True)
    result["dataset"] = "field"
    result["method"] = "M5"
    result["seed"] = seed
    result["metric_space"] = np.where(result.target.eq("PERM"), "log10", "linear")
    result["initial_budget"] = 125
    result["final_budget"] = 125
    result["source_path"] = str(path.relative_to(repo)).replace("\\", "/")
    return result


def build_figure_data(repo: Path, output_dir: Path) -> dict[str, Path]:
    summary = repo / "outputs" / "supplementary" / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = add_budget_columns(pd.read_csv(summary / "supplementary_metrics_long.csv"))
    metrics["source_path"] = metrics["metrics_path"]
    _validate_seeds(metrics.query("section in ['active','backbone','calibration']"), ["section", "dataset", "variant", "final_budget", "target"])

    pairs = []
    active = metrics.query("section == 'active'")
    for method in ["M6", "M7", "M8", "M9"]:
        pairs.append(_paired(active.query("method == 'M5'"), active[active.method == method], ["dataset", "target", "final_budget", "seed"], f"{method}_vs_M5"))
    backbone = metrics.query("section == 'backbone'")
    pairs.append(_paired(backbone.query("variant == 'bilstm'"), backbone.query("variant == 'target_aware_bilstm'"), ["dataset", "target", "final_budget", "seed"], "target_aware_vs_bilstm"))
    transfer = metrics.query("section == 'transfer'")
    pairs.append(_paired(transfer.query("method == 'M5'"), transfer.query("method == 'M9'"), ["dataset", "protocol", "target", "seed"], "M9_vs_M5_transfer"))
    paired = pd.concat(pairs, ignore_index=True)

    active_rows = []
    selection = metrics.query("section == 'active' and method in ['M6','M7','M8','M9']")
    for path_text in selection.metrics_path.drop_duplicates():
        context = selection[selection.metrics_path == path_text].iloc[0]
        path = repo / Path(path_text).parent / "active_learning_metrics.csv"
        if path.exists():
            frame = pd.read_csv(path)
            for column, value in {"dataset": context.dataset, "method": context.method, "seed": int(context.seed), "initial_budget": int(context.initial_budget), "final_budget": int(context.final_budget), "source_path": str(path.relative_to(repo)).replace("\\", "/")}.items():
                frame[column] = value
            active_rows.append(frame)
    active_round = pd.concat(active_rows, ignore_index=True)

    calibration_rows = []
    for path_text in metrics.query("section == 'calibration'").metrics_path.drop_duplicates():
        context = metrics[metrics.metrics_path == path_text].iloc[0]
        path = repo / Path(path_text).parent / "calibration.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            calibration_rows.append({"dataset": context.dataset, "method": context.method, "target": "ALL", "metric_space": "mixed", "variant": context.variant, "seed": int(context.seed), "initial_budget": int(context.initial_budget), "final_budget": int(context.final_budget), "lambda_mc": payload.get("lambda_mc"), "source_path": str(path.relative_to(repo)).replace("\\", "/")})
    calibration = pd.DataFrame(calibration_rows).drop_duplicates()

    trust_rows = []
    for path_text in metrics.query("section == 'active' and dataset == 'field' and method == 'M5' and final_budget == 125").metrics_path.drop_duplicates():
        context = metrics[metrics.metrics_path == path_text].iloc[0]
        path = repo / Path(path_text).parent / "trust_metrics.csv"
        frame = pd.read_csv(path)
        frame["dataset"] = "field"; frame["method"] = "M5"; frame["seed"] = int(context.seed)
        frame["initial_budget"] = 125; frame["final_budget"] = 125
        frame["metric_space"] = np.where(frame.target.eq("PERM"), "log10", "linear")
        frame["source_path"] = str(path.relative_to(repo)).replace("\\", "/")
        trust_rows.append(frame)
    trust = pd.concat(trust_rows, ignore_index=True)
    representative = _representative_predictions(repo, metrics)

    outputs = {
        "metrics": output_dir / "metrics_seed_long.csv",
        "paired": output_dir / "paired_improvement_seed_long.csv",
        "active_round": output_dir / "active_round_long.csv",
        "representative": output_dir / "representative_predictions.csv",
        "calibration": output_dir / "calibration_weights_long.csv",
        "trust": output_dir / "trust_status_long.csv",
    }
    for key, frame in {"metrics": metrics, "paired": paired, "active_round": active_round, "representative": representative, "calibration": calibration, "trust": trust}.items():
        frame.to_csv(outputs[key], index=False, encoding="utf-8-sig")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("figure/data"))
    args = parser.parse_args()
    outputs = build_figure_data(args.repo.resolve(), args.output.resolve())
    print(f"wrote {len(outputs)} canonical figure data files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
