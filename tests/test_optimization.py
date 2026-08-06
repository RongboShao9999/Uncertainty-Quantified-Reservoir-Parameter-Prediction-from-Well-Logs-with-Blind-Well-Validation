from pathlib import Path

import pandas as pd
import yaml

from bnn_inversion import cli
from bnn_inversion.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RuntimeConfig,
    TrainingConfig,
)
from bnn_inversion.experiments import ExperimentResult
from bnn_inversion.optimization import OptimizationResult, optimize_config
from bnn_inversion.optimization import _metric_score


def test_metric_score_prioritizes_phif_sw_mae_and_mape() -> None:
    better_reservoir = pd.DataFrame(
        {
            "target": ["PHIF", "SW", "VSH"],
            "rmse": [0.2, 0.2, 0.01],
            "mae": [0.03, 0.06, 0.2],
            "epsilon_mape": [0.12, 0.10, 0.8],
            "interval_score": [0.2, 0.2, 0.2],
            "uce": [0.05, 0.05, 0.05],
            "picp": [0.95, 0.95, 0.95],
        }
    )
    worse_reservoir = pd.DataFrame(
        {
            "target": ["PHIF", "SW", "VSH"],
            "rmse": [0.1, 0.1, 0.01],
            "mae": [0.08, 0.12, 0.01],
            "epsilon_mape": [0.30, 0.35, 0.01],
            "interval_score": [0.1, 0.1, 0.1],
            "uce": [0.01, 0.01, 0.01],
            "picp": [0.99, 0.99, 0.99],
        }
    )

    assert _metric_score(better_reservoir) < _metric_score(worse_reservoir)


def test_optimize_config_writes_ranked_results_and_best_config(tmp_path: Path) -> None:
    base = ExperimentConfig(
        data=DataConfig(initial_labels=5),
        model=ModelConfig(hidden_size=32, deterministic_dropout=0.1, mc_dropout=0.05),
        training=TrainingConfig(epochs=1, learning_rate=0.001, bnn_learning_rate=0.0005),
        runtime=RuntimeConfig(device="cpu", output_dir=tmp_path),
    )

    def fake_runner(config: ExperimentConfig) -> ExperimentResult:
        score = (
            0.5
            if config.model.hidden_size == 64
            and config.model.mc_dropout == 0.1
            and config.uncertainty.calibration_objective == "interval_score"
            else 2.0
        )
        metrics = pd.DataFrame(
            {
                "target": ["PHIF"],
                "rmse": [score],
                "interval_score": [score],
                "uce": [0.1],
                "picp": [0.95],
            }
        )
        output = config.runtime.output_dir
        output.mkdir(parents=True, exist_ok=True)
        return ExperimentResult(
            predictions=pd.DataFrame(),
            metrics=metrics,
            metrics_path=output / "metrics.csv",
            output_dir=output,
        )

    result = optimize_config(
        base,
        tmp_path / "optimization",
        runner=fake_runner,
        search_space={
            "model.hidden_size": [32, 64],
            "model.mc_dropout": [0.05, 0.1],
            "uncertainty.calibration_objective": ["nll", "interval_score"],
        },
        max_trials=8,
    )

    assert result.best_config.model.hidden_size == 64
    assert result.best_config.model.mc_dropout == 0.1
    assert result.best_config.uncertainty.calibration_objective == "interval_score"
    assert (tmp_path / "optimization" / "optimization_results.csv").exists()
    assert (tmp_path / "optimization" / "best_config.yaml").exists()
    assert (tmp_path / "optimization" / "optimization_summary.json").exists()


def test_optimize_config_honors_max_trials(tmp_path: Path) -> None:
    base = ExperimentConfig(runtime=RuntimeConfig(device="cpu", output_dir=tmp_path))
    seen: list[int] = []

    def fake_runner(config: ExperimentConfig) -> ExperimentResult:
        seen.append(config.model.hidden_size)
        return ExperimentResult(
            predictions=pd.DataFrame(),
            metrics=pd.DataFrame({"target": ["PHIF"], "rmse": [1.0]}),
            metrics_path=config.runtime.output_dir / "metrics.csv",
            output_dir=config.runtime.output_dir,
        )

    optimize_config(
        base,
        tmp_path / "optimization",
        runner=fake_runner,
        search_space={"model.hidden_size": [16, 32, 64]},
        max_trials=2,
    )

    assert len(seen) == 2


def test_optimize_config_reuses_completed_trial_metrics(tmp_path: Path) -> None:
    base = ExperimentConfig(runtime=RuntimeConfig(device="cpu", output_dir=tmp_path))
    completed = tmp_path / "optimization" / "trial_001"
    completed.mkdir(parents=True)
    pd.DataFrame(
        {
            "target": ["PHIF"],
            "rmse": [0.1],
            "interval_score": [0.2],
            "uce": [0.01],
            "picp": [0.95],
        }
    ).to_csv(completed / "metrics.csv", index=False)
    seen: list[Path] = []

    def fake_runner(config: ExperimentConfig) -> ExperimentResult:
        seen.append(config.runtime.output_dir)
        output = config.runtime.output_dir
        output.mkdir(parents=True, exist_ok=True)
        metrics = pd.DataFrame(
            {
                "target": ["PHIF"],
                "rmse": [1.0],
                "interval_score": [1.0],
                "uce": [0.1],
                "picp": [0.8],
            }
        )
        metrics.to_csv(output / "metrics.csv", index=False)
        return ExperimentResult(
            predictions=pd.DataFrame(),
            metrics=metrics,
            metrics_path=output / "metrics.csv",
            output_dir=output,
        )

    result = optimize_config(
        base,
        tmp_path / "optimization",
        runner=fake_runner,
        search_space={"model.hidden_size": [16, 32]},
        max_trials=2,
    )

    assert seen == [tmp_path / "optimization" / "trial_002"]
    assert result.results["trial"].tolist() == [1, 2]


def test_cli_optimize_uses_configured_output_and_trial_limit(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "runtime": {
                    "output_dir": (tmp_path / "optimized").as_posix(),
                    "device": "cpu",
                },
                "training": {"epochs": 1},
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_optimize(config, output_dir, *, max_trials=None, seeds=None):
        captured["output_dir"] = Path(output_dir)
        captured["max_trials"] = max_trials
        captured["seeds"] = seeds
        return OptimizationResult(
            best_config=config,
            results=pd.DataFrame({"trial": [1], "score": [0.0]}),
            output_dir=Path(output_dir),
        )

    monkeypatch.setattr(cli, "optimize_config", fake_optimize)

    assert (
        cli.main(
            [
                "optimize",
                "--config",
                str(config_path),
                "--max-trials",
                "3",
                "--seeds",
                "1,2",
            ]
        )
        == 0
    )
    assert captured["output_dir"] == tmp_path / "optimized"
    assert captured["max_trials"] == 3
    assert captured["seeds"] == [1, 2]


def test_cli_export_cleaned_uses_configured_data_root(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.yaml"
    data_root = tmp_path / "data"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "root": data_root.as_posix(),
                    "dataset": "field",
                    "feature_profile": "main_7",
                },
                "runtime": {"device": "cpu"},
            }
        ),
        encoding="utf-8",
    )
    captured: list[tuple[str, Path, str | None, str]] = []

    def fake_export(dataset, root, *, feature_profile=None, folder_name="processed_cleaned"):
        captured.append((dataset, Path(root), feature_profile, folder_name))
        return Path(root) / folder_name / f"{dataset}_cleaned.csv"

    monkeypatch.setattr(cli, "export_cleaned_dataset", fake_export)

    assert (
        cli.main(
            [
                "export-cleaned",
                "--config",
                str(config_path),
                "--dataset",
                "all",
                "--folder-name",
                "cleaned_csv",
            ]
        )
        == 0
    )
    assert captured == [
        ("field", data_root, "main_7", "cleaned_csv"),
        ("spwla", data_root, "main_7", "cleaned_csv"),
        ("forward", data_root, "main_7", "cleaned_csv"),
    ]


def test_run_matrix_uses_large_label_counts_and_m9_terminal_budgets(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "runtime": {"output_dir": (tmp_path / "matrix").as_posix(), "device": "cpu"},
                "training": {"epochs": 1},
                "data": {"validation_size": 4},
            }
        ),
        encoding="utf-8",
    )
    captured: list[tuple[str, int, int, int]] = []

    def fake_run(config):
        captured.append(
            (
                config.method,
                config.data.initial_labels,
                config.active_learning.rounds,
                config.active_learning.batch_budget,
            )
        )
        output = config.runtime.output_dir
        output.mkdir(parents=True, exist_ok=True)
        metrics = pd.DataFrame({"target": ["PHIF"], "rmse": [1.0]})
        metrics.to_csv(output / "metrics.csv", index=False)
        return ExperimentResult(
            predictions=pd.DataFrame(),
            metrics=metrics,
            metrics_path=output / "metrics.csv",
            output_dir=output,
        )

    monkeypatch.setattr(cli, "run_experiment", fake_run)

    assert (
        cli.main(
            [
                "run-matrix",
                "--config",
                str(config_path),
                "--methods",
                "M1,M9",
                "--seeds",
                "0",
            ]
        )
        == 0
    )

    assert [item[1] for item in captured if item[0] == "M1"] == [50, 100, 200, 500]
    assert [item[1] for item in captured if item[0] == "M9"] == [50, 100, 100, 100]
    assert [(item[2], item[3]) for item in captured if item[0] == "M9"] == [
        (0, 0),
        (0, 0),
        (1, 100),
        (4, 100),
    ]


def test_run_matrix_default_seeds_skip_seed_zero(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "runtime": {"output_dir": (tmp_path / "matrix").as_posix(), "device": "cpu"},
                "training": {"epochs": 1},
                "data": {"validation_size": 4},
            }
        ),
        encoding="utf-8",
    )
    seen_seeds: list[int] = []

    def fake_run(config):
        seen_seeds.append(config.seed)
        output = config.runtime.output_dir
        output.mkdir(parents=True, exist_ok=True)
        metrics = pd.DataFrame({"target": ["PHIF"], "rmse": [1.0]})
        metrics.to_csv(output / "metrics.csv", index=False)
        return ExperimentResult(
            predictions=pd.DataFrame(),
            metrics=metrics,
            metrics_path=output / "metrics.csv",
            output_dir=output,
        )

    monkeypatch.setattr(cli, "run_experiment", fake_run)

    assert (
        cli.main(
            [
                "run-matrix",
                "--config",
                str(config_path),
                "--methods",
                "M1",
            ]
        )
        == 0
    )

    assert sorted(set(seen_seeds)) == [1, 2, 3, 4]


def test_cli_summarize_matrix_writes_multiseed_summary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "matrix"
    for seed, mae in [(1, 0.1), (2, 0.3)]:
        output = root / "M1" / "N50" / f"seed{seed}"
        output.mkdir(parents=True)
        pd.DataFrame(
            {
                "target": ["PHIF"],
                "metric_space": ["linear"],
                "mae": [mae],
                "epsilon_mape": [mae * 10],
            }
        ).to_csv(output / "metrics.csv", index=False)

    assert (
        cli.main(
            [
                "summarize-matrix",
                "--root",
                str(root),
                "--output",
                str(tmp_path / "summary"),
                "--exclude-seeds",
                "0",
            ]
        )
        == 0
    )

    summary = pd.read_csv(tmp_path / "summary" / "matrix_metrics_summary.csv")
    assert summary.loc[0, "mae_mean"] == 0.2


def test_cli_summarize_matrix_includes_seed_zero_by_default(
    tmp_path: Path,
) -> None:
    root = tmp_path / "matrix"
    for seed, mae in [(0, 0.5), (1, 0.1)]:
        output = root / "M1" / "N50" / f"seed{seed}"
        output.mkdir(parents=True)
        pd.DataFrame(
            {
                "target": ["PHIF"],
                "metric_space": ["linear"],
                "mae": [mae],
                "epsilon_mape": [mae * 10],
            }
        ).to_csv(output / "metrics.csv", index=False)

    assert (
        cli.main(
            [
                "summarize-matrix",
                "--root",
                str(root),
                "--output",
                str(tmp_path / "summary"),
            ]
        )
        == 0
    )

    summary = pd.read_csv(tmp_path / "summary" / "matrix_metrics_summary.csv")
    assert summary.loc[0, "mae_mean"] == 0.3
    assert summary.loc[0, "run_count"] == 2
