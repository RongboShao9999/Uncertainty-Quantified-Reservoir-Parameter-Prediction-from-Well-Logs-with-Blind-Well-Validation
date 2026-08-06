from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from bnn_inversion.config import ExperimentConfig
from bnn_inversion.experiments import ExperimentResult, run_experiment


Runner = Callable[[ExperimentConfig], ExperimentResult]


DEFAULT_SEARCH_SPACE: dict[str, list[object]] = {
    "model.hidden_size": [32, 64, 128],
    "model.deterministic_dropout": [0.1, 0.2, 0.3],
    "model.mc_dropout": [0.05, 0.1, 0.2],
    "training.learning_rate": [0.001, 0.0005],
    "training.bnn_learning_rate": [0.0005, 0.0001],
    "uncertainty.calibration_objective": ["nll", "interval_score"],
}


@dataclass(frozen=True)
class OptimizationResult:
    best_config: ExperimentConfig
    results: pd.DataFrame
    output_dir: Path


def _replace_nested(config: ExperimentConfig, dotted_key: str, value: object) -> ExperimentConfig:
    parts = dotted_key.split(".")
    if len(parts) != 2:
        raise ValueError(f"optimization keys must be section.field: {dotted_key}")
    section, field = parts
    nested = getattr(config, section)
    return replace(config, **{section: replace(nested, **{field: value})})


def _apply_parameters(config: ExperimentConfig, parameters: dict[str, object]) -> ExperimentConfig:
    result = config
    for key, value in parameters.items():
        result = _replace_nested(result, key, value)
    return result


def _trials(search_space: dict[str, list[object]], max_trials: int | None) -> list[dict[str, object]]:
    keys = list(search_space)
    values = [search_space[key] for key in keys]
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    if max_trials is not None:
        if max_trials < 1:
            raise ValueError("max_trials must be positive")
        combinations = combinations[:max_trials]
    return combinations


def _metric_score(metrics: pd.DataFrame) -> float:
    if metrics.empty:
        return float("inf")

    priority_metrics = metrics
    if "target" in metrics:
        target_values = metrics["target"].astype(str).str.upper()
        priority = metrics.loc[target_values.isin({"PHIF", "SW"})]
        if not priority.empty:
            priority_metrics = priority

    def _mean(column: str, default: float = 0.0, *, frame: pd.DataFrame = metrics) -> float:
        if column not in frame:
            return default
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if len(finite) else default

    mae = _mean("mae", _mean("rmse", 0.0, frame=priority_metrics), frame=priority_metrics)
    epsilon_mape = _mean("epsilon_mape", 0.0, frame=priority_metrics)
    rmse = _mean("rmse", mae, frame=priority_metrics)
    interval_score = _mean("interval_score", 0.0, frame=priority_metrics)
    uce = _mean("uce", 0.0, frame=priority_metrics)
    picp = _mean("picp", 0.95, frame=priority_metrics)
    coverage_penalty = abs(picp - 0.95)
    return mae + epsilon_mape + 0.25 * rmse + 0.10 * interval_score + 0.10 * uce + 0.10 * coverage_penalty


def optimize_config(
    config: ExperimentConfig,
    output_dir: Path | str,
    *,
    runner: Runner = run_experiment,
    search_space: dict[str, list[object]] | None = None,
    max_trials: int | None = None,
    seeds: list[int] | tuple[int, ...] | None = None,
) -> OptimizationResult:
    """Run a reproducible grid search and persist ranked optimization artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    space = search_space or DEFAULT_SEARCH_SPACE
    rows: list[dict[str, object]] = []
    best_score = float("inf")
    best_config: ExperimentConfig | None = None
    seed_values = tuple(seeds) if seeds is not None else (config.seed,)
    if not seed_values:
        raise ValueError("optimization requires at least one seed")
    trial_index = 0
    stop = False
    for parameters in _trials(space, None):
        for seed in seed_values:
            trial_index += 1
            if max_trials is not None and trial_index > max_trials:
                stop = True
                break
            trial_output = output / f"trial_{trial_index:03d}"
            trial_config = _apply_parameters(
                replace(
                    config,
                    seed=int(seed),
                    runtime=replace(config.runtime, output_dir=trial_output),
                ),
                parameters,
            )
            metrics_path = trial_output / "metrics.csv"
            if metrics_path.exists():
                metrics = pd.read_csv(metrics_path)
                result_output = trial_output
            else:
                result = runner(trial_config)
                metrics = result.metrics
                result_output = result.output_dir
            score = _metric_score(metrics)
            row: dict[str, object] = {
                "trial": trial_index,
                "seed": int(seed),
                "score": score,
                "output_dir": result_output.as_posix(),
            }
            row.update(parameters)
            for column in ("rmse", "mae", "mape", "epsilon_mape", "smape", "interval_score", "uce", "picp"):
                if column in metrics:
                    values = pd.to_numeric(metrics[column], errors="coerce")
                    row[f"mean_{column}"] = float(values.mean())
            rows.append(row)
            if score < best_score:
                best_score = score
                best_config = trial_config
        if stop:
            break
    if best_config is None:
        raise RuntimeError("optimization did not run any trials")
    results = pd.DataFrame(rows).sort_values("score", kind="stable").reset_index(drop=True)
    results.to_csv(output / "optimization_results.csv", index=False)
    (output / "best_config.yaml").write_text(
        yaml.safe_dump(best_config.to_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (output / "optimization_summary.json").write_text(
        json.dumps(
            {
                "best_score": best_score,
                "best_trial": int(results.loc[0, "trial"]),
                "trials": len(rows),
                "best_output_dir": best_config.runtime.output_dir.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return OptimizationResult(best_config=best_config, results=results, output_dir=output)
