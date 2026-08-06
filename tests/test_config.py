from pathlib import Path

import pytest
import torch

from bnn_inversion.config import load_config
from bnn_inversion.types import PredictiveDistribution


def test_load_config_resolves_external_data_path(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "data:\n  root: D:/coding/BNN/DATASET\nseed: 7\n",
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.data.root.as_posix() == "D:/coding/BNN/DATASET"
    assert cfg.seed == 7


def test_runtime_loader_and_fast_figure_options_are_configurable(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            [
                "runtime:",
                "  num_workers: 2",
                "  pin_memory: true",
                "  persistent_workers: true",
                "  figure_format: svg",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.runtime.num_workers == 2
    assert cfg.runtime.pin_memory is True
    assert cfg.runtime.persistent_workers is True
    assert cfg.runtime.figure_format == "svg"


def test_runtime_device_defaults_to_cuda(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("seed: 3\n", encoding="utf-8")

    cfg = load_config(path)

    assert cfg.runtime.device == "cuda"


def test_precision_oriented_defaults_for_reservoir_prediction(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("seed: 3\n", encoding="utf-8")

    cfg = load_config(path)

    assert cfg.data.initial_labels == 50
    assert cfg.data.window_size == 15
    assert cfg.model.hidden_size == 96
    assert cfg.model.layers == 3
    assert cfg.model.target_head_hidden_sizes == [96, 80, 128]
    assert cfg.training.learning_rate == pytest.approx(0.0007)
    assert cfg.training.relative_loss_weight == pytest.approx(0.0)
    assert cfg.training.relative_loss_epsilon == pytest.approx(0.05)


def test_relative_loss_options_are_configurable(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            [
                "model:",
                "  point_architecture: reservoir_bilstm",
                "training:",
                "  relative_loss_weight: 0.4",
                "  relative_loss_epsilon: 0.08",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.model.point_architecture == "reservoir_bilstm"
    assert cfg.training.relative_loss_weight == pytest.approx(0.4)
    assert cfg.training.relative_loss_epsilon == pytest.approx(0.08)


def test_target_specific_and_reservoir_sampling_options_are_configurable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "\n".join(
            [
                "data:",
                "  target_columns: [SW]",
                "training:",
                "  reservoir_sample_weight: 2.5",
                "  reservoir_phif_threshold: 0.05",
                "  high_sw_sample_weight: 1.5",
                "  high_sw_threshold: 0.8",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.data.target_columns == ["SW"]
    assert cfg.training.reservoir_sample_weight == pytest.approx(2.5)
    assert cfg.training.reservoir_phif_threshold == pytest.approx(0.05)
    assert cfg.training.high_sw_sample_weight == pytest.approx(1.5)
    assert cfg.training.high_sw_threshold == pytest.approx(0.8)


def test_predictive_distribution_rejects_negative_variance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        PredictiveDistribution(
            torch.zeros(1),
            torch.tensor([-1.0]),
            torch.zeros(1),
        )
