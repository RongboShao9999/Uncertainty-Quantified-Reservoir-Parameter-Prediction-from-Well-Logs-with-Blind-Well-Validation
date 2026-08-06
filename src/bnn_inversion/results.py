from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NUMERIC_METRICS = (
    "rmse",
    "mae",
    "epsilon_mape",
    "mape",
    "smape",
    "r2",
    "picp",
    "mpiw",
    "nmpiw",
    "interval_score",
    "uce",
    "epistemic_ratio",
    "nll",
    "uncertainty_error_spearman",
    "trusted_rate",
    "suspect_rate",
    "trusted_rmse",
    "suspect_rmse",
    "trusted_mae",
    "suspect_mae",
    "mae_gap",
    "suspect_mean_interval_width",
    "delta_rmse_pct",
    "relative_improvement_pct",
    "aulc",
    "baseline_aulc",
    "label_saving_rate",
    "raw_picp",
    "raw_mpiw",
    "raw_nmpiw",
    "raw_interval_score",
    "raw_uce",
    "calibrated_picp",
    "calibrated_mpiw",
    "calibrated_nmpiw",
    "calibrated_interval_score",
    "calibrated_uce",
    "risk_precision",
    "risk_recall",
    "risk_f1",
    "risk_error_enrichment",
    "risk_mae",
    "nonrisk_mae",
    "risk_rate",
)
SUPPLEMENTARY_IDENTITY_COLUMNS = (
    "section",
    "dataset",
    "variant",
    "method",
    "budget_N",
    "seed",
    "protocol",
    "target",
    "metric_space",
)
TRANSFER_DATASETS = {
    "spwla_holdout": "spwla",
    "field_internal": "field",
    "field_to_forward": "forward",
}


@dataclass(frozen=True)
class MatrixSummary:
    long: pd.DataFrame
    summary: pd.DataFrame
    output_dir: Path


@dataclass(frozen=True)
class SupplementarySummary:
    long: pd.DataFrame
    summary: pd.DataFrame
    paired: pd.DataFrame
    budgets: pd.DataFrame
    missing: pd.DataFrame
    output_dir: Path


def _parse_int_token(value: str, prefix: str) -> int:
    if not value.startswith(prefix) or not value[len(prefix) :].isdigit():
        raise ValueError(f"invalid supplementary path token: {value}")
    return int(value[len(prefix) :])


def _parse_supplementary_path(
    root: Path, metrics_path: Path
) -> dict[str, object]:
    parts = metrics_path.relative_to(root).parts
    if parts[0] == "active" and len(parts) == 6:
        _, dataset, method, budget, seed, _ = parts
        return {
            "section": "active",
            "dataset": dataset,
            "variant": method,
            "method": method,
            "budget_N": _parse_int_token(budget, "N"),
            "seed": _parse_int_token(seed, "seed"),
            "protocol": "",
        }
    if parts[0] == "uncertainty" and len(parts) == 6:
        _, dataset, method, budget, seed, _ = parts
        return {
            "section": "uncertainty",
            "dataset": dataset,
            "variant": method,
            "method": method,
            "budget_N": _parse_int_token(budget, "N"),
            "seed": _parse_int_token(seed, "seed"),
            "protocol": "",
        }
    if parts[0] == "calibration" and len(parts) == 7:
        _, dataset, objective, method, budget, seed, _ = parts
        return {
            "section": "calibration",
            "dataset": dataset,
            "variant": objective,
            "method": method,
            "budget_N": _parse_int_token(budget, "N"),
            "seed": _parse_int_token(seed, "seed"),
            "protocol": "",
        }
    if parts[0] == "backbone" and len(parts) == 6:
        _, dataset, architecture, budget, seed, _ = parts
        return {
            "section": "backbone",
            "dataset": dataset,
            "variant": architecture,
            "method": "M5",
            "budget_N": _parse_int_token(budget, "N"),
            "seed": _parse_int_token(seed, "seed"),
            "protocol": "",
        }
    if parts[0] == "transfer" and len(parts) == 5:
        _, protocol, method, seed, _ = parts
        if protocol not in TRANSFER_DATASETS:
            raise ValueError(f"unknown transfer protocol: {protocol}")
        return {
            "section": "transfer",
            "dataset": TRANSFER_DATASETS[protocol],
            "variant": method,
            "method": method,
            "budget_N": pd.NA,
            "seed": _parse_int_token(seed, "seed"),
            "protocol": protocol,
        }
    raise ValueError(f"unrecognized supplementary metrics path: {metrics_path}")


def collect_supplementary_metrics(root: Path | str) -> pd.DataFrame:
    """Collect completed supplementary metrics into one validated long table."""

    root_path = Path(root)
    rows: list[dict[str, object]] = []
    patterns = (
        "active/*/*/N*/seed*/metrics.csv",
        "uncertainty/*/*/N*/seed*/metrics.csv",
        "calibration/*/*/*/N*/seed*/metrics.csv",
        "backbone/*/*/N*/seed*/metrics.csv",
        "transfer/*/*/seed*/metrics.csv",
    )
    for pattern in patterns:
        for metrics_path in sorted(root_path.glob(pattern)):
            identity = _parse_supplementary_path(root_path, metrics_path)
            metrics = pd.read_csv(metrics_path)
            missing = {"target", "rmse"} - set(metrics.columns)
            if missing:
                raise ValueError(
                    f"missing required columns {sorted(missing)} in {metrics_path}"
            )
            for _, metric_row in metrics.iterrows():
                target = str(metric_row["target"])
                raw_metric_space = metric_row.get("metric_space", "")
                metric_space = (
                    "" if pd.isna(raw_metric_space) else str(raw_metric_space).strip()
                )
                if target == "PERM" and not metric_space:
                    raise ValueError(f"PERM metric_space is required in {metrics_path}")
                row = {
                    **identity,
                    "target": target,
                    "metric_space": metric_space,
                    "metrics_path": metrics_path.as_posix(),
                }
                for column in NUMERIC_METRICS:
                    row[column] = pd.to_numeric(
                        pd.Series([metric_row.get(column, pd.NA)]), errors="coerce"
                    ).iloc[0]
                picp = row["picp"]
                row["coverage_error"] = (
                    abs(float(picp) - 0.95) if pd.notna(picp) else pd.NA
                )
                rows.append(row)
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    duplicate_mask = long.duplicated(
        subset=list(SUPPLEMENTARY_IDENTITY_COLUMNS), keep=False
    )
    if duplicate_mask.any():
        duplicate = long.loc[duplicate_mask, SUPPLEMENTARY_IDENTITY_COLUMNS]
        raise ValueError(
            "duplicate supplementary result identity: "
            f"{duplicate.iloc[0].to_dict()}"
        )
    long["budget_N"] = long["budget_N"].astype("Int64")
    return long.sort_values(
        ["section", "dataset", "variant", "method", "budget_N", "seed", "target"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def validate_supplementary_budgets(
    root: Path | str, long: pd.DataFrame
) -> pd.DataFrame:
    """Validate active-learning outputs against their declared final budgets."""

    del root
    columns = [
        "section",
        "dataset",
        "method",
        "budget_N",
        "seed",
        "initial_labels",
        "expected_added",
        "observed_added",
        "final_budget",
        "status",
        "metrics_path",
    ]
    if long.empty or "section" not in long:
        return pd.DataFrame(columns=columns)
    experiments = long[long["section"] == "active"].drop_duplicates(
        subset=["dataset", "method", "budget_N", "seed", "metrics_path"]
    )
    rows: list[dict[str, object]] = []
    for _, experiment in experiments.iterrows():
        method = str(experiment["method"])
        final_budget = int(experiment["budget_N"])
        metrics_path = Path(str(experiment["metrics_path"]))
        initial_labels = final_budget if method == "M5" else 100
        expected_added = final_budget - initial_labels
        observed_added: int | None = 0 if method == "M5" else None
        status = "valid"
        if method != "M5":
            audit_path = metrics_path.with_name("active_learning_metrics.csv")
            if not audit_path.exists():
                status = "missing"
            else:
                audit = pd.read_csv(audit_path)
                if audit.empty or "cumulative_labeled" not in audit:
                    status = "missing"
                else:
                    observed_added = int(audit["cumulative_labeled"].iloc[-1])
                    if observed_added != expected_added:
                        status = "invalid"
        rows.append(
            {
                "section": "active",
                "dataset": experiment["dataset"],
                "method": method,
                "budget_N": final_budget,
                "seed": int(experiment["seed"]),
                "initial_labels": initial_labels,
                "expected_added": expected_added,
                "observed_added": observed_added,
                "final_budget": final_budget,
                "status": status,
                "metrics_path": metrics_path.as_posix(),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _bootstrap_interval(
    values: np.ndarray, *, samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def _aggregate_supplementary(
    long: pd.DataFrame, *, bootstrap_samples: int, bootstrap_seed: int
) -> pd.DataFrame:
    columns = [
        "section",
        "dataset",
        "variant",
        "method",
        "budget_N",
        "protocol",
        "target",
        "metric_space",
        "metric",
        "mean",
        "std",
        "ci95_low",
        "ci95_high",
        "run_count",
    ]
    if long.empty:
        return pd.DataFrame(columns=columns)
    group_columns = columns[:8]
    metric_columns = (*NUMERIC_METRICS, "coverage_error")
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, object]] = []
    for identity, group in long.groupby(group_columns, dropna=False, sort=True):
        identity_row = dict(zip(group_columns, identity))
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(
                dtype=float
            )
            if not len(values):
                continue
            low, high = _bootstrap_interval(
                values, samples=bootstrap_samples, rng=rng
            )
            rows.append(
                {
                    **identity_row,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                    "ci95_low": low,
                    "ci95_high": high,
                    "run_count": len(values),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _same_nullable(series: pd.Series, value: object) -> pd.Series:
    if pd.isna(value):
        return series.isna()
    return series == value


def _baseline_mask(long: pd.DataFrame, candidate: pd.Series) -> pd.Series:
    mask = (
        (long["section"] == candidate["section"])
        & (long["dataset"] == candidate["dataset"])
        & _same_nullable(long["budget_N"], candidate["budget_N"])
        & (long["protocol"] == candidate["protocol"])
        & (long["target"] == candidate["target"])
        & (long["metric_space"] == candidate["metric_space"])
    )
    section = candidate["section"]
    if section == "active":
        return mask & (long["method"] == "M5") & (long["variant"] == "M5")
    if section == "calibration":
        return (
            mask
            & (long["method"] == candidate["method"])
            & (long["variant"] == "nll")
        )
    if section == "backbone":
        return mask & (long["variant"] == "bilstm")
    if section == "transfer":
        return mask & (long["method"] == "M5") & (long["variant"] == "M5")
    return pd.Series(False, index=long.index)


def _is_candidate(row: pd.Series) -> bool:
    if row["section"] == "active":
        return row["method"] != "M5"
    if row["section"] == "calibration":
        return row["variant"] != "nll"
    if row["section"] == "backbone":
        return row["variant"] != "bilstm"
    if row["section"] == "transfer":
        return row["method"] != "M5"
    return False


def _paired_supplementary(long: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "section",
        "dataset",
        "variant",
        "method",
        "budget_N",
        "protocol",
        "target",
        "metric_space",
        "metric",
        "candidate_mean",
        "baseline_mean",
        "paired_delta",
        "improvement_fraction",
        "win_count",
        "pair_count",
    ]
    if long.empty:
        return pd.DataFrame(columns=columns)
    identity_columns = columns[:8]
    lower_is_better = (
        "rmse",
        "mae",
        "epsilon_mape",
        "mape",
        "smape",
        "mpiw",
        "interval_score",
        "uce",
        "coverage_error",
    )
    higher_is_better = ("r2",)
    rows: list[dict[str, object]] = []
    candidates = long[long.apply(_is_candidate, axis=1)]
    for identity, candidate_group in candidates.groupby(
        identity_columns, dropna=False, sort=True
    ):
        identity_row = dict(zip(identity_columns, identity))
        baseline = long[_baseline_mask(long, candidate_group.iloc[0])]
        for metric in (*lower_is_better, *higher_is_better):
            candidate_values = candidate_group[["seed", metric]].dropna()
            baseline_values = baseline[["seed", metric]].dropna()
            pairs = candidate_values.merge(
                baseline_values, on="seed", suffixes=("_candidate", "_baseline")
            )
            if pairs.empty:
                continue
            candidate_array = pairs[f"{metric}_candidate"].to_numpy(dtype=float)
            baseline_array = pairs[f"{metric}_baseline"].to_numpy(dtype=float)
            delta = candidate_array - baseline_array
            denominator = np.abs(baseline_array)
            if metric in lower_is_better:
                improvement = np.divide(
                    baseline_array - candidate_array,
                    denominator,
                    out=np.full_like(candidate_array, np.nan),
                    where=denominator > 0,
                )
                wins = candidate_array < baseline_array
            else:
                improvement = np.divide(
                    candidate_array - baseline_array,
                    denominator,
                    out=np.full_like(candidate_array, np.nan),
                    where=denominator > 0,
                )
                wins = candidate_array > baseline_array
            rows.append(
                {
                    **identity_row,
                    "metric": metric,
                    "candidate_mean": float(candidate_array.mean()),
                    "baseline_mean": float(baseline_array.mean()),
                    "paired_delta": float(delta.mean()),
                    "improvement_fraction": (
                        float(np.nanmean(improvement))
                        if np.isfinite(improvement).any()
                        else float("nan")
                    ),
                    "win_count": int(wins.sum()),
                    "pair_count": len(pairs),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def summarize_supplementary_results(
    root: Path | str,
    output_dir: Path | str | None = None,
    *,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260629,
    expected_artifacts: Iterable[Path | str] = (),
) -> SupplementarySummary:
    """Collect, validate, compare, and write supplementary experiment results."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    root_path = Path(root)
    output_path = Path(output_dir) if output_dir is not None else root_path / "summary"
    output_path.mkdir(parents=True, exist_ok=True)
    long = collect_supplementary_metrics(root_path)
    summary = _aggregate_supplementary(
        long,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    paired = _paired_supplementary(long)
    budgets = validate_supplementary_budgets(root_path, long)
    actual = set(long.get("metrics_path", pd.Series(dtype=str)).astype(str))
    missing_rows = []
    for artifact in expected_artifacts:
        path = Path(artifact)
        if path.as_posix() not in actual and not path.exists():
            missing_rows.append(
                {"identity": path.parent.as_posix(), "expected_artifact": path.as_posix()}
            )
    missing = pd.DataFrame(
        missing_rows, columns=["identity", "expected_artifact"]
    )
    long.to_csv(output_path / "supplementary_metrics_long.csv", index=False)
    summary.to_csv(output_path / "supplementary_metrics_summary.csv", index=False)
    paired.to_csv(output_path / "paired_comparisons.csv", index=False)
    budgets.to_csv(output_path / "budget_validation.csv", index=False)
    missing.to_csv(output_path / "missing_experiments.csv", index=False)
    if not budgets.empty and (budgets["status"] == "invalid").any():
        invalid = int((budgets["status"] == "invalid").sum())
        raise ValueError(f"{invalid} active-learning budget validations failed")
    return SupplementarySummary(
        long=long,
        summary=summary,
        paired=paired,
        budgets=budgets,
        missing=missing,
        output_dir=output_path,
    )


def _parse_matrix_path(path: Path) -> tuple[str, int, int] | None:
    parts = path.parts
    if len(parts) < 4:
        return None
    method = parts[-4]
    budget = parts[-3]
    seed = parts[-2]
    if not (method.startswith("M") and method[1:].isdigit()):
        return None
    if not (budget.startswith("N") and budget[1:].isdigit()):
        return None
    if not (seed.startswith("seed") and seed[4:].isdigit()):
        return None
    return method, int(budget[1:]), int(seed[4:])


def collect_matrix_metrics(
    root: Path | str,
    *,
    exclude_seeds: Iterable[int] = (),
) -> pd.DataFrame:
    """Collect M*/N*/seed*/metrics.csv rows into one long-form table."""

    root_path = Path(root)
    excluded = set(int(seed) for seed in exclude_seeds)
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(root_path.glob("M*/N*/seed*/metrics.csv")):
        parsed = _parse_matrix_path(metrics_path)
        if parsed is None:
            continue
        method, budget, seed = parsed
        if seed in excluded:
            continue
        metrics = pd.read_csv(metrics_path)
        for _, metric_row in metrics.iterrows():
            row: dict[str, object] = {
                "method": method,
                "budget_N": budget,
                "seed": seed,
                "target": metric_row.get("target"),
                "metric_space": metric_row.get("metric_space", ""),
                "metrics_path": metrics_path.as_posix(),
            }
            for column in NUMERIC_METRICS:
                if column in metrics:
                    row[column] = metric_row.get(column)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_matrix_results(
    root: Path | str,
    output_dir: Path | str | None = None,
    *,
    exclude_seeds: Iterable[int] = (),
) -> MatrixSummary:
    """Write long-form and mean/std matrix summaries for multi-seed runs."""

    root_path = Path(root)
    output_path = Path(output_dir) if output_dir is not None else root_path
    output_path.mkdir(parents=True, exist_ok=True)
    long = collect_matrix_metrics(root_path, exclude_seeds=exclude_seeds)
    if long.empty:
        raise ValueError(f"no matrix metrics found under {root_path}")
    numeric_columns = [column for column in NUMERIC_METRICS if column in long]
    grouped = long.groupby(["method", "budget_N", "target"], as_index=False)
    summary = grouped.agg(
        **{
            f"{column}_{stat}": (column, stat)
            for column in numeric_columns
            for stat in ("mean", "std")
        },
        run_count=("seed", "nunique"),
    )
    summary = summary.sort_values(["method", "budget_N", "target"], kind="stable")
    long.to_csv(output_path / "matrix_metrics_long.csv", index=False)
    summary.to_csv(output_path / "matrix_metrics_summary.csv", index=False)
    return MatrixSummary(long=long, summary=summary, output_dir=output_path)
