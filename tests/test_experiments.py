import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from bnn_inversion.config import (
    ActiveLearningConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RuntimeConfig,
    TrainingConfig,
    UncertaintyConfig,
)
from bnn_inversion.data.adapters import CanonicalFrame
from bnn_inversion.experiments import (
    METHOD_REGISTRY,
    _metric_frame,
    _make_point_model,
    _prepare_transfer_source,
    _prediction_frame,
    _reservoir_sample_weights,
    _select_targets,
    run_cross_domain_experiment,
    run_experiment,
)
from bnn_inversion.models.bilstm import BiLSTMRegressor, TargetAwareBiLSTMRegressor
from bnn_inversion.types import PredictiveDistribution
from bnn_inversion.cli import parse_overrides


def _synthetic_frame() -> CanonicalFrame:
    rows = []
    rng = np.random.default_rng(5)
    for well in range(5):
        for depth in range(12):
            gr = 20.0 + well + depth
            rt = 1.0 + depth
            rows.append(
                {
                    "WELLNUM": well,
                    "DEPTH": float(depth),
                    "GR": gr,
                    "RT": rt,
                    "PHIF": 0.05 + 0.005 * gr,
                    "SW": 0.2 + 0.01 * depth,
                }
            )
    return CanonicalFrame(
        frame=pd.DataFrame(rows),
        feature_columns=("GR", "RT"),
        target_columns=("PHIF", "SW"),
        well_column="WELLNUM",
        depth_column="DEPTH",
        audit=(),
        dataset="synthetic",
    )


def _tiny_config(output: Path, method: str = "M5") -> ExperimentConfig:
    return ExperimentConfig(
        seed=11,
        method=method,
        data=DataConfig(
            root=Path("unused"),
            dataset="field",
            feature_profile="main_7",
            split_mode="group_well",
            test_fraction=0.2,
            validation_size=6,
            initial_labels=8,
            window_size=3,
        ),
        model=ModelConfig(hidden_size=4, layers=1, deterministic_dropout=0.1, mc_dropout=0.2),
        training=TrainingConfig(
            epochs=2,
            batch_size=4,
            learning_rate=0.01,
            bnn_learning_rate=0.005,
            early_stopping_patience=2,
        ),
        uncertainty=UncertaintyConfig(mc_samples=3, bnn_samples=3, confidence=0.95),
        runtime=RuntimeConfig(device="cpu", output_dir=output),
    )


def _spwla_transfer_frame() -> CanonicalFrame:
    rows = []
    for well in range(9):
        for depth in range(8):
            gr = 30.0 + well + depth
            rows.append(
                {
                    "WELLNUM": well,
                    "DEPTH": float(depth),
                    "GR": gr,
                    "CALI": 8.0 + 0.01 * depth,
                    "NEU": 0.1 + 0.01 * well,
                    "DEN": 2.3 + 0.01 * depth,
                    "RDEP": 3.0 + 0.02 * depth,
                    "RMED": 4.0 + 0.02 * depth,
                    "PHIF": 0.08 + 0.002 * gr,
                    "SW": 0.3 + 0.01 * depth,
                    "VSH": 0.2 + 0.001 * gr,
                }
            )
    return CanonicalFrame(
        frame=pd.DataFrame(rows),
        feature_columns=("CALI", "DEN", "GR", "NEU", "RDEP", "RMED"),
        target_columns=("PHIF", "SW", "VSH"),
        well_column="WELLNUM",
        depth_column="DEPTH",
        audit=(),
        dataset="spwla",
    )


def _field_transfer_frame() -> CanonicalFrame:
    rows = []
    for well in range(205):
        for depth in range(6):
            gr = 50.0 + well * 0.01 + depth
            rows.append(
                {
                    "WELLNUM": well,
                    "DEPTH": float(depth),
                    "GR": gr,
                    "CAL": 20.0 + 0.01 * depth,
                    "SP": 40.0 + 0.1 * depth,
                    "AC": 220.0 + depth,
                    "CNL": 15.0 + 0.01 * well,
                    "DEN": 2.4 + 0.01 * depth,
                    "RT": 1.0 + 0.02 * depth,
                    "PHIF": 0.08 + 0.0002 * well,
                    "SW": 0.3 + 0.01 * depth,
                    "PERM": 0.4 + 0.001 * well + 0.01 * depth,
                }
            )
    return CanonicalFrame(
        frame=pd.DataFrame(rows),
        feature_columns=("GR", "CAL", "SP", "AC", "CNL", "DEN", "RT"),
        target_columns=("PHIF", "SW", "PERM"),
        well_column="WELLNUM",
        depth_column="DEPTH",
        audit=(),
        dataset="field",
    )


def _subset_spwla_without_well8() -> CanonicalFrame:
    frame = _spwla_transfer_frame()
    subset = frame.frame[frame.frame["WELLNUM"] != 8].copy().reset_index(drop=True)
    return CanonicalFrame(
        frame=subset,
        feature_columns=frame.feature_columns,
        target_columns=frame.target_columns,
        well_column=frame.well_column,
        depth_column=frame.depth_column,
        audit=(),
        dataset="spwla",
    )


def _subset_spwla_well8() -> CanonicalFrame:
    frame = _spwla_transfer_frame()
    subset = frame.frame[frame.frame["WELLNUM"] == 8].copy().reset_index(drop=True)
    return CanonicalFrame(
        frame=subset,
        feature_columns=frame.feature_columns,
        target_columns=frame.target_columns,
        well_column=frame.well_column,
        depth_column=frame.depth_column,
        audit=(),
        dataset="spwla:well8",
    )


def test_method_registry_contains_exact_m1_to_m9() -> None:
    assert set(METHOD_REGISTRY) == {f"M{number}" for number in range(1, 10)}


def test_m5_auto_point_architecture_uses_target_aware_model() -> None:
    m1 = _make_point_model(3, 2, _tiny_config(Path("unused"), method="M1"))
    m5 = _make_point_model(3, 2, _tiny_config(Path("unused"), method="M5"))

    assert isinstance(m1, BiLSTMRegressor)
    assert isinstance(m5, TargetAwareBiLSTMRegressor)


def test_cli_overrides_build_nested_mapping() -> None:
    assert parse_overrides(
        ["training.epochs=2", "runtime.device=cpu", "data.test_fraction=0.25"]
    ) == {
        "training": {"epochs": 2},
        "runtime": {"device": "cpu"},
        "data": {"test_fraction": 0.25},
    }


def test_permeability_predictions_are_exported_in_md() -> None:
    distribution = PredictiveDistribution(
        mean=torch.tensor([[0.0]]),
        epistemic_variance=torch.tensor([[0.01]]),
        aleatoric_variance=torch.tensor([[0.03]]),
    )
    predictions = _prediction_frame(
        ("PERM",),
        target=torch.tensor([[0.0]]),
        mask=torch.tensor([[True]]),
        indices=np.array([4]),
        point=torch.tensor([[1.0]]),
        distribution=distribution,
        confidence=0.95,
    )

    assert predictions.loc[0, "y_true"] == pytest.approx(1.0)
    assert predictions.loc[0, "point_prediction"] == pytest.approx(10.0)
    assert predictions.loc[0, "unit"] == "mD"
    assert predictions.loc[0, "y_true_log10"] == pytest.approx(0.0)
    assert predictions.loc[0, "point_prediction_log10"] == pytest.approx(1.0)
    assert predictions.loc[0, "lower"] > 0
    assert predictions.loc[0, "lower_log10"] < predictions.loc[0, "upper_log10"]
    assert predictions.loc[0, "interval_width"] > 0
    assert predictions.loc[0, "epistemic_ratio"] == pytest.approx(0.25)
    assert predictions.loc[0, "uncertainty_dominance"] == "aleatoric"


def test_fraction_targets_are_clipped_to_physical_range() -> None:
    distribution = PredictiveDistribution(
        mean=torch.tensor([[1.5, -0.2, 0.4]]),
        epistemic_variance=torch.full((1, 3), 0.01),
        aleatoric_variance=torch.full((1, 3), 0.01),
    )
    predictions = _prediction_frame(
        ("PHIF", "SW", "VSH"),
        target=torch.tensor([[0.2, 0.8, 0.5]]),
        mask=torch.tensor([[True, True, True]]),
        indices=np.array([1]),
        point=torch.tensor([[1.4, -0.3, 0.7]]),
        distribution=distribution,
        confidence=0.95,
    )

    bounded = predictions.set_index("target")
    assert bounded.loc["PHIF", "point_prediction"] == 1.0
    assert bounded.loc["SW", "point_prediction"] == 0.0
    assert bounded.loc["PHIF", "upper"] == 1.0
    assert bounded.loc["SW", "lower"] == 0.0
    assert bounded.loc["VSH", "point_prediction"] == pytest.approx(0.7)


def test_perm_metrics_are_computed_in_log10_space() -> None:
    predictions = pd.DataFrame(
        {
            "target": ["PERM", "PERM"],
            "y_true": [1.0, 100.0],
            "point_prediction": [10.0, 10.0],
            "y_true_log10": [0.0, 2.0],
            "point_prediction_log10": [1.0, 1.0],
        }
    )

    metrics = _metric_frame(predictions)

    assert metrics.loc[0, "target"] == "PERM"
    assert metrics.loc[0, "rmse"] == pytest.approx(1.0)
    assert metrics.loc[0, "mae"] == pytest.approx(1.0)
    assert metrics.loc[0, "metric_space"] == "log10"


def test_fraction_metrics_use_reservoir_epsilon_for_mape() -> None:
    predictions = pd.DataFrame(
        {
            "target": ["PHIF"],
            "y_true": [0.02],
            "point_prediction": [0.04],
        }
    )

    metrics = _metric_frame(predictions)

    assert metrics.loc[0, "epsilon_mape"] == pytest.approx(0.4)


def test_configured_target_columns_select_target_specific_frame() -> None:
    selected = _select_targets(
        _field_transfer_frame(),
        DataConfig(target_columns=["SW"], root=Path("unused")),
    )

    assert selected.target_columns == ("SW",)
    assert "PHIF" in selected.frame.columns
    assert "PERM" in selected.frame.columns


def test_reservoir_sample_weights_emphasize_reservoir_and_high_sw_rows() -> None:
    frame = pd.DataFrame(
        {
            "PHIF": [0.03, 0.06, 0.08],
            "SW": [0.4, 0.7, 0.9],
        }
    )
    weights = _reservoir_sample_weights(
        frame,
        np.array([0, 1, 2]),
        reservoir_sample_weight=2.0,
        reservoir_phif_threshold=0.05,
        high_sw_sample_weight=1.5,
        high_sw_threshold=0.8,
    )

    assert weights.tolist() == pytest.approx([1.0, 3.0, 4.5])


def test_m5_smoke_run_writes_reproducible_artifacts(tmp_path: Path) -> None:
    result = run_experiment(_tiny_config(tmp_path), frame=_synthetic_frame())

    assert result.metrics_path.exists()
    assert (tmp_path / "resolved_config.yaml").exists()
    assert (tmp_path / "cleaning_audit.json").exists()
    assert (tmp_path / "predictions.csv").exists()
    assert (tmp_path / "calibration.json").exists()
    assert (tmp_path / "trust_metrics.csv").exists()
    assert (tmp_path / "calibration_bins.csv").exists()
    assert (tmp_path / "figures" / "trust_status.pdf").exists()
    assert (tmp_path / "preprocessor.joblib").exists()
    assert (tmp_path / "training_history.json").exists()
    assert (tmp_path / "best_point_model.pt").exists()
    history = json.loads((tmp_path / "training_history.json").read_text(encoding="utf-8"))
    assert history["point"]["best_epoch"] >= 0
    assert history["point"]["best_validation_loss"] >= 0
    assert set(result.predictions["status"].unique()) <= {"可信", "存疑"}
    assert set(result.predictions["target"].unique()) == {"PHIF", "SW"}


def test_m1_does_not_claim_interval_outputs(tmp_path: Path) -> None:
    result = run_experiment(_tiny_config(tmp_path, method="M1"), frame=_synthetic_frame())
    assert "status" not in result.predictions
    assert not (tmp_path / "calibration.json").exists()


def test_active_learning_methods_require_positive_labels(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path, method="M9")
    config = replace(config, data=replace(config.data, initial_labels=0))
    with pytest.raises(ValueError, match="initial_labels must be positive"):
        run_experiment(config, frame=_synthetic_frame())


def test_m9_runs_one_active_round_and_records_selection(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path, method="M9")
    config = replace(
        config,
        data=replace(config.data, initial_labels=6),
        active_learning=ActiveLearningConfig(
            batch_budget=2,
            rounds=1,
            inconsistency_penalty=2.0,
            random_fraction=0.1,
        ),
    )

    run_experiment(config, frame=_synthetic_frame())

    selections = pd.read_csv(tmp_path / "active_learning.csv")
    assert selections["round"].unique().tolist() == [1]
    assert len(selections) == 2
    assert {
        "score",
        "epistemic_component",
        "inconsistency_component",
        "valid_target_count",
        "selection_mode",
    }.issubset(selections.columns)


def test_cross_domain_run_uses_only_shared_targets_and_source_preprocessor(
    tmp_path: Path,
) -> None:
    source = _synthetic_frame()
    target_frame = source.frame.copy()
    target_frame["GR"] += 100.0
    target = CanonicalFrame(
        frame=target_frame,
        feature_columns=source.feature_columns,
        target_columns=("PHIF", "SW"),
        well_column="WELLNUM",
        depth_column="DEPTH",
        audit=(),
        dataset="target-domain",
    )

    result = run_cross_domain_experiment(
        _tiny_config(tmp_path), source=source, target=target
    )

    assert set(result.predictions["target"]) == {"PHIF", "SW"}
    assert (tmp_path / "domain_summary.json").exists()


def test_spwla_transfer_uses_well8_as_target_without_field_alignment(
    tmp_path: Path, monkeypatch
) -> None:
    frame = _spwla_transfer_frame()

    def fake_load_dataset(dataset, root, *, feature_profile=None):
        assert dataset == "spwla"
        return frame

    monkeypatch.setattr(
        "bnn_inversion.experiments.load_dataset", fake_load_dataset
    )
    config = replace(
        _tiny_config(tmp_path),
        data=replace(
            _tiny_config(tmp_path).data,
            dataset="spwla",
            feature_profile="transfer_5",
            initial_labels=6,
        ),
    )

    result = run_cross_domain_experiment(config, target_dataset="spwla")

    summary = json.loads((tmp_path / "domain_summary.json").read_text(encoding="utf-8"))
    assert summary["source"] == "spwla"
    assert summary["target"] == "spwla:well8"
    assert summary["target_wells"] == [8]
    assert summary["train_wells"] == list(range(8))
    assert summary["targets"] == ["PHIF", "SW", "VSH"]
    assert set(result.predictions["target"]) == {"PHIF", "SW", "VSH"}


def test_field_transfer_samples_20_wells_before_200_and_one_after_200(
    tmp_path: Path, monkeypatch
) -> None:
    frame = _field_transfer_frame()

    def fake_load_dataset(dataset, root, *, feature_profile=None):
        assert dataset == "field"
        return frame

    monkeypatch.setattr(
        "bnn_inversion.experiments.load_dataset", fake_load_dataset
    )
    config = replace(
        _tiny_config(tmp_path, method="M1"),
        data=replace(
            _tiny_config(tmp_path).data,
            dataset="field",
            feature_profile="main_7",
            initial_labels=6,
        ),
    )

    result = run_cross_domain_experiment(config, target_dataset="field")

    summary = json.loads((tmp_path / "domain_summary.json").read_text(encoding="utf-8"))
    assert summary["protocol"] == "field_internal_random_well_transfer"
    assert summary["source"] == "field:first200_random20"
    assert len(summary["selected_source_wells"]) == 20
    assert all(0 <= well < 200 for well in summary["selected_source_wells"])
    assert len(summary["target_wells"]) == 1
    assert summary["target_wells"][0] >= 200
    assert summary["targets"] == ["PHIF", "SW", "PERM"]
    assert set(result.predictions["target"]) == {"PHIF", "SW", "PERM"}


def test_transfer_source_split_keeps_all_source_wells_for_training_pool() -> None:
    source = _subset_spwla_without_well8()
    config = replace(
        _tiny_config(Path("unused")),
        data=replace(_tiny_config(Path("unused")).data, initial_labels=6),
    )

    _, _, split, _ = _prepare_transfer_source(source, config)

    covered_indices = np.union1d(split.train, split.validation)
    covered_wells = sorted(source.frame.iloc[covered_indices]["WELLNUM"].unique().tolist())
    assert covered_wells == list(range(8))
    assert len(split.test) == 0


def test_m9_cross_domain_runs_active_learning_and_records_selection(
    tmp_path: Path,
) -> None:
    source = _subset_spwla_without_well8()
    target = _subset_spwla_well8()
    config = replace(
        _tiny_config(tmp_path, method="M9"),
        data=replace(_tiny_config(tmp_path).data, initial_labels=6),
        active_learning=ActiveLearningConfig(
            batch_budget=2,
            rounds=1,
            inconsistency_penalty=2.0,
            random_fraction=0.1,
        ),
    )

    result = run_cross_domain_experiment(config, source=source, target=target)

    assert set(result.predictions["target"]) == {"PHIF", "SW", "VSH"}
    selections = pd.read_csv(tmp_path / "active_learning.csv")
    assert selections["round"].unique().tolist() == [1]
    assert len(selections) == 2
