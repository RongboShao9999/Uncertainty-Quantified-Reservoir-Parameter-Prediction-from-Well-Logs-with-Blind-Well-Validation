from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


MetricDict = dict[str, float | str]


def _arrays(*values: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    arrays = [np.asarray(value, dtype=float).reshape(-1) for value in values]
    lengths = {len(value) for value in arrays}
    if len(lengths) != 1:
        raise ValueError("metric arrays must have equal lengths")
    valid = np.logical_and.reduce([np.isfinite(value) for value in arrays])
    return arrays, valid


def point_metrics(
    target: np.ndarray, prediction: np.ndarray, *, epsilon: float = 1e-6
) -> MetricDict:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    (target_values, prediction_values), valid = _arrays(target, prediction)
    target_values = target_values[valid]
    prediction_values = prediction_values[valid]
    if len(target_values) == 0:
        return {
            "rmse": math.nan,
            "mae": math.nan,
            "mape": math.nan,
            "smape": math.nan,
            "epsilon_mape": math.nan,
            "r2": math.nan,
            "reason": "no finite target/prediction pairs",
        }
    errors = prediction_values - target_values
    absolute_errors = np.abs(errors)
    percentage_mask = np.abs(target_values) > epsilon
    result: MetricDict = {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(absolute_errors)),
        "smape": float(
            np.mean(
                2.0
                * absolute_errors
                / (np.abs(target_values) + np.abs(prediction_values) + epsilon)
            )
        ),
        "epsilon_mape": float(
            np.mean(absolute_errors / np.maximum(np.abs(target_values), epsilon))
        ),
    }
    if percentage_mask.any():
        result["mape"] = float(
            np.mean(absolute_errors[percentage_mask] / np.abs(target_values[percentage_mask]))
        )
    else:
        result["mape"] = math.nan
        result["mape_reason"] = "MAPE requires targets whose magnitude exceeds epsilon"
    denominator = float(np.sum((target_values - target_values.mean()) ** 2))
    if len(target_values) < 2 or denominator == 0:
        result["r2"] = math.nan
        result["r2_reason"] = "R2 requires at least two non-constant targets"
    else:
        result["r2"] = float(1.0 - np.sum(errors**2) / denominator)
    return result


def interval_metrics(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    confidence: float = 0.95,
    bins: int = 10,
) -> MetricDict:
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if bins < 1:
        raise ValueError("bins must be positive")
    (target_values, lower_values, upper_values), valid = _arrays(
        target, lower, upper
    )
    target_values = target_values[valid]
    lower_values = lower_values[valid]
    upper_values = upper_values[valid]
    if len(target_values) == 0:
        return {
            "picp": math.nan,
            "mpiw": math.nan,
            "nmpiw": math.nan,
            "interval_score": math.nan,
            "uce": math.nan,
            "reason": "no finite interval rows",
        }
    if np.any(upper_values < lower_values):
        raise ValueError("upper interval bounds must not be below lower bounds")
    inside = (target_values >= lower_values) & (target_values <= upper_values)
    widths = upper_values - lower_values
    target_range = float(np.max(target_values) - np.min(target_values))
    nmpiw = float(np.mean(widths) / target_range) if target_range > 0 else math.nan
    alpha = 1.0 - confidence
    score = widths.copy()
    below = target_values < lower_values
    above = target_values > upper_values
    score[below] += 2.0 / alpha * (lower_values[below] - target_values[below])
    score[above] += 2.0 / alpha * (target_values[above] - upper_values[above])
    ordered = np.argsort(widths, kind="stable")
    weighted_error = 0.0
    for group in np.array_split(ordered, min(bins, len(ordered))):
        if len(group):
            empirical = float(np.mean(inside[group]))
            weighted_error += len(group) / len(ordered) * abs(empirical - confidence)
    return {
        "picp": float(np.mean(inside)),
        "mpiw": float(np.mean(widths)),
        "nmpiw": nmpiw,
        "interval_score": float(np.mean(score)),
        "uce": float(weighted_error),
    }


def uncertainty_metrics(
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    epistemic: np.ndarray,
    total: np.ndarray,
) -> MetricDict:
    (target_values, prediction_values, epistemic_values, total_values), valid = _arrays(
        target, prediction, epistemic, total
    )
    valid &= total_values > 0
    if not valid.any():
        return {
            "epistemic_ratio": math.nan,
            "nll": math.nan,
            "uncertainty_error_spearman": math.nan,
            "reason": "no finite rows with positive total variance",
        }
    target_values = target_values[valid]
    prediction_values = prediction_values[valid]
    epistemic_values = epistemic_values[valid]
    total_values = total_values[valid]
    absolute_error = np.abs(prediction_values - target_values)
    ratio = float(np.mean(epistemic_values / total_values))
    nll = float(
        np.mean(
            0.5 * np.log(2.0 * np.pi * total_values)
            + (target_values - prediction_values) ** 2 / (2.0 * total_values)
        )
    )
    if len(target_values) < 2 or np.all(total_values == total_values[0]):
        correlation = math.nan
        reason = "Spearman correlation requires varying data"
    else:
        correlation = float(spearmanr(total_values, absolute_error).statistic)
        reason = ""
    result: MetricDict = {
        "epistemic_ratio": ratio,
        "nll": nll,
        "uncertainty_error_spearman": correlation,
    }
    if reason:
        result["spearman_reason"] = reason
    return result


def conformal_scale(
    *,
    target: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    confidence: float = 0.95,
    epsilon: float = 1e-8,
) -> tuple[float, int]:
    """Fit a split-conformal scale to validation residuals only."""

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    (target_values, mean_values, variance_values), valid = _arrays(
        target, mean, variance
    )
    valid &= variance_values >= 0
    if not valid.any():
        raise ValueError("conformal calibration requires finite non-negative variance")
    scores = np.abs(target_values[valid] - mean_values[valid]) / np.sqrt(
        np.maximum(variance_values[valid], epsilon)
    )
    scores.sort(kind="stable")
    rank = min(int(math.ceil((len(scores) + 1) * confidence)), len(scores))
    return float(scores[rank - 1]), int(len(scores))


def risk_metrics(
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    total: np.ndarray,
    fraction: float = 0.1,
) -> MetricDict:
    """Evaluate whether the most uncertain samples capture the largest errors."""

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    (target_values, prediction_values, total_values), valid = _arrays(
        target, prediction, total
    )
    valid &= total_values >= 0
    if not valid.any():
        return {"risk_precision": math.nan, "risk_recall": math.nan, "risk_f1": math.nan,
                "risk_error_enrichment": math.nan, "risk_mae": math.nan,
                "nonrisk_mae": math.nan, "risk_rate": math.nan,
                "risk_count": math.nan, "high_error_count": math.nan,
                "reason": "no finite rows with non-negative total variance"}
    errors = np.abs(prediction_values[valid] - target_values[valid])
    uncertainty = total_values[valid]
    count = max(1, int(math.ceil(len(errors) * fraction)))
    risk_indices = np.argsort(-uncertainty, kind="stable")[:count]
    high_error_indices = np.argsort(-errors, kind="stable")[:count]
    risk = np.zeros(len(errors), dtype=bool)
    high_error = np.zeros(len(errors), dtype=bool)
    risk[risk_indices] = True
    high_error[high_error_indices] = True
    overlap = int(np.sum(risk & high_error))
    precision = overlap / int(risk.sum())
    recall = overlap / int(high_error.sum())
    risk_mae = float(np.mean(errors[risk]))
    all_mae = float(np.mean(errors))
    return {
        "risk_precision": float(precision),
        "risk_recall": float(recall),
        "risk_f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "risk_error_enrichment": risk_mae / all_mae if all_mae > 0 else math.nan,
        "risk_mae": risk_mae,
        "nonrisk_mae": float(np.mean(errors[~risk])) if (~risk).any() else math.nan,
        "risk_rate": float(risk.mean()),
        "risk_count": float(risk.sum()),
        "high_error_count": float(high_error.sum()),
    }


def trust_metrics(rows: Iterable[Mapping[str, Any]] | pd.DataFrame) -> MetricDict:
    """Summarize trusted/suspect prediction quality for a single target group."""

    frame = pd.DataFrame(rows)
    required = {"y_true", "point_prediction", "status"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"trust metrics require columns: {missing}")
    if frame.empty:
        return {
            "trusted_rate": math.nan,
            "suspect_rate": math.nan,
            "trusted_rmse": math.nan,
            "suspect_rmse": math.nan,
            "trusted_mae": math.nan,
            "suspect_mae": math.nan,
            "mae_gap": math.nan,
            "suspect_mean_interval_width": math.nan,
            "reason": "no prediction rows",
        }
    status = frame["status"].astype(str)
    trusted = status.isin({"可信", "鍙俊", "trusted"})
    suspect = status.isin({"存疑", "瀛樼枒", "suspect"})

    def _rmse(mask: pd.Series) -> float:
        values = frame.loc[mask, ["y_true", "point_prediction"]].to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1)
        if not valid.any():
            return math.nan
        errors = values[valid, 1] - values[valid, 0]
        return float(np.sqrt(np.mean(errors**2)))

    def _mae(mask: pd.Series) -> float:
        values = frame.loc[mask, ["y_true", "point_prediction"]].to_numpy(dtype=float)
        valid = np.isfinite(values).all(axis=1)
        if not valid.any():
            return math.nan
        errors = values[valid, 1] - values[valid, 0]
        return float(np.mean(np.abs(errors)))

    if "interval_width" in frame:
        suspect_width = pd.to_numeric(
            frame.loc[suspect, "interval_width"], errors="coerce"
        )
        suspect_mean_width = float(suspect_width.mean()) if len(suspect_width) else math.nan
    else:
        suspect_mean_width = math.nan
    total = len(frame)
    trusted_mae = _mae(trusted)
    suspect_mae = _mae(suspect)
    return {
        "trusted_rate": float(trusted.sum() / total),
        "suspect_rate": float(suspect.sum() / total),
        "trusted_count": float(trusted.sum()),
        "suspect_count": float(suspect.sum()),
        "trusted_rmse": _rmse(trusted),
        "suspect_rmse": _rmse(suspect),
        "trusted_mae": trusted_mae,
        "suspect_mae": suspect_mae,
        "mae_gap": (
            suspect_mae - trusted_mae
            if np.isfinite(suspect_mae) and np.isfinite(trusted_mae)
            else math.nan
        ),
        "suspect_mean_interval_width": suspect_mean_width,
    }


def _clean_learning_curve(
    budgets: np.ndarray, rmse: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    (budget_values, rmse_values), valid = _arrays(budgets, rmse)
    valid &= budget_values >= 0
    budget_values = budget_values[valid]
    rmse_values = rmse_values[valid]
    if len(budget_values) == 0:
        return budget_values, rmse_values
    order = np.argsort(budget_values, kind="stable")
    frame = pd.DataFrame(
        {"budget": budget_values[order], "rmse": rmse_values[order]}
    )
    grouped = frame.groupby("budget", as_index=False)["rmse"].mean()
    return grouped["budget"].to_numpy(dtype=float), grouped["rmse"].to_numpy(
        dtype=float
    )


def _curve_aulc(budgets: np.ndarray, rmse: np.ndarray) -> float:
    if len(budgets) < 2:
        return math.nan
    return float(np.trapezoid(rmse, budgets))


def _first_budget_at_threshold(
    budgets: np.ndarray, rmse: np.ndarray, threshold: float
) -> float:
    reached = budgets[rmse <= threshold]
    return float(reached[0]) if len(reached) else math.nan


def active_learning_efficiency_metrics(
    *,
    candidate_budgets: np.ndarray,
    candidate_rmse: np.ndarray,
    baseline_budgets: np.ndarray | None = None,
    baseline_rmse: np.ndarray | None = None,
    threshold: float | None = None,
) -> MetricDict:
    """Evaluate active-learning efficiency from RMSE learning curves."""

    budgets, rmse = _clean_learning_curve(candidate_budgets, candidate_rmse)
    if len(budgets) == 0:
        return {
            "delta_rmse_pct": math.nan,
            "relative_improvement_pct": math.nan,
            "aulc": math.nan,
            "baseline_aulc": math.nan,
            "label_saving_rate": math.nan,
            "reason": "no finite learning-curve rows",
        }
    initial = rmse[0]
    final = rmse[-1]
    result: MetricDict = {
        "delta_rmse_pct": float((initial - final) / abs(initial) * 100.0)
        if initial != 0
        else math.nan,
        "relative_improvement_pct": math.nan,
        "aulc": _curve_aulc(budgets, rmse),
        "baseline_aulc": math.nan,
        "label_saving_rate": math.nan,
    }
    if baseline_budgets is None or baseline_rmse is None:
        return result
    base_budgets, base_rmse = _clean_learning_curve(baseline_budgets, baseline_rmse)
    result["baseline_aulc"] = _curve_aulc(base_budgets, base_rmse)
    if len(base_rmse):
        baseline_final = base_rmse[-1]
        result["relative_improvement_pct"] = (
            float((baseline_final - final) / abs(baseline_final) * 100.0)
            if baseline_final != 0
            else math.nan
        )
    if threshold is not None:
        baseline_needed = _first_budget_at_threshold(base_budgets, base_rmse, threshold)
        candidate_needed = _first_budget_at_threshold(budgets, rmse, threshold)
        if (
            np.isfinite(baseline_needed)
            and np.isfinite(candidate_needed)
            and baseline_needed > 0
        ):
            result["label_saving_rate"] = float(
                (baseline_needed - candidate_needed) / baseline_needed
            )
    return result


def calibration_bin_metrics(
    *,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    total: np.ndarray,
    bins: int = 10,
) -> list[MetricDict]:
    """Report empirical coverage and error by ascending uncertainty bins."""

    if bins < 1:
        raise ValueError("bins must be positive")
    (target_values, lower_values, upper_values, total_values), valid = _arrays(
        target, lower, upper, total
    )
    target_values = target_values[valid]
    lower_values = lower_values[valid]
    upper_values = upper_values[valid]
    total_values = total_values[valid]
    if len(target_values) == 0:
        return []
    if np.any(upper_values < lower_values):
        raise ValueError("upper interval bounds must not be below lower bounds")
    ordered = np.argsort(total_values, kind="stable")
    rows: list[MetricDict] = []
    for index, group in enumerate(np.array_split(ordered, min(bins, len(ordered))), start=1):
        if len(group) == 0:
            continue
        inside = (target_values[group] >= lower_values[group]) & (
            target_values[group] <= upper_values[group]
        )
        absolute_error = np.abs(
            0.5 * (lower_values[group] + upper_values[group]) - target_values[group]
        )
        rows.append(
            {
                "bin": float(index),
                "count": float(len(group)),
                "uncertainty_min": float(np.min(total_values[group])),
                "uncertainty_max": float(np.max(total_values[group])),
                "uncertainty_mean": float(np.mean(total_values[group])),
                "coverage": float(np.mean(inside)),
                "mean_interval_width": float(
                    np.mean(upper_values[group] - lower_values[group])
                ),
                "mean_absolute_error": float(np.mean(absolute_error)),
            }
        )
    return rows
