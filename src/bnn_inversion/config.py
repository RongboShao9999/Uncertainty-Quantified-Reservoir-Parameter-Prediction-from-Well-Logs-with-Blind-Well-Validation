from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml


@dataclass(frozen=True)
class DataConfig:
    root: Path = Path("D:/coding/BNN/DATASET")
    dataset: str = "field"
    feature_profile: str = "main_7"
    split_mode: str = "group_well"
    test_fraction: float = 0.2
    validation_size: int = 200
    initial_labels: int = 50
    window_size: int = 15
    scaler: str = "robust"
    winsorize_low: float | None = 0.01
    winsorize_high: float | None = 0.99
    target_columns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelConfig:
    point_architecture: str = "auto"
    hidden_size: int = 96
    layers: int = 3
    deterministic_dropout: float = 0.12
    mc_dropout: float = 0.1
    target_head_hidden_sizes: list[int] = field(default_factory=lambda: [96, 80, 128])
    target_dropouts: list[float] = field(default_factory=lambda: [0.08, 0.08, 0.12])
    bounded_targets: list[str] = field(default_factory=lambda: ["PHIF", "SW", "VSH"])


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 500
    batch_size: int = 64
    learning_rate: float = 0.0007
    bnn_learning_rate: float = 0.0005
    early_stopping_patience: int = 20
    target_loss_scaling: bool = True
    point_loss: str = "huber"
    huber_delta: float = 1.0
    target_loss_weights: list[float] = field(default_factory=lambda: [1.6, 2.0, 1.0])
    relative_loss_weight: float = 0.0
    relative_loss_epsilon: float = 0.05
    reservoir_sample_weight: float = 0.0
    reservoir_phif_threshold: float = 0.05
    high_sw_sample_weight: float = 0.0
    high_sw_threshold: float = 0.8
    weight_decay: float = 0.0001
    lr_patience: int = 4
    lr_factor: float = 0.5


@dataclass(frozen=True)
class UncertaintyConfig:
    mc_samples: int = 50
    bnn_samples: int = 50
    confidence: float = 0.95
    calibration_objective: str = "nll"


@dataclass(frozen=True)
class ActiveLearningConfig:
    batch_budget: int = 5
    rounds: int = 5
    inconsistency_penalty: float = 2.0
    random_fraction: float = 0.1


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cuda"
    output_dir: Path = Path("outputs")
    figure_format: str = "pdf"
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    method: str = "M5"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    active_learning: ActiveLearningConfig = field(default_factory=ActiveLearningConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        return _paths_to_strings(raw)


def _paths_to_strings(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {key: _paths_to_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_paths_to_strings(item) for item in value]
    return value


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _dataclass_type(annotation: Any) -> type[Any] | None:
    if isinstance(annotation, type) and is_dataclass(annotation):
        return annotation
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        for candidate in get_args(annotation):
            if isinstance(candidate, type) and is_dataclass(candidate):
                return candidate
    return None


def _construct(cls: type[Any], values: dict[str, Any], context: str) -> Any:
    allowed = {item.name: item for item in fields(cls)}
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise ValueError(f"unknown configuration keys at {context}: {unknown}")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        annotation = hints.get(name, allowed[name].type)
        nested = _dataclass_type(annotation)
        if nested is not None:
            if not isinstance(value, dict):
                raise ValueError(f"{context}.{name} must be a mapping")
            kwargs[name] = _construct(nested, value, f"{context}.{name}")
        elif annotation is Path:
            kwargs[name] = Path(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(
    path: Path | str,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load a YAML experiment configuration with strict key validation."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError("configuration root must be a mapping")
    merged = _merge(loaded, overrides or {})
    config = _construct(ExperimentConfig, merged, "config")
    _validate(config)
    return config


def _validate(config: ExperimentConfig) -> None:
    if not 0 < config.data.test_fraction < 1:
        raise ValueError("data.test_fraction must be between 0 and 1")
    if config.data.window_size < 1 or config.data.window_size % 2 == 0:
        raise ValueError("data.window_size must be a positive odd integer")
    if config.model.point_architecture not in {"auto", "bilstm", "attentive_bilstm", "target_aware_bilstm", "reservoir_bilstm"}:
        raise ValueError(
            "model.point_architecture must be 'auto', 'bilstm', 'attentive_bilstm', 'target_aware_bilstm', or 'reservoir_bilstm'"
        )
    if any(value < 1 for value in config.model.target_head_hidden_sizes):
        raise ValueError("model.target_head_hidden_sizes must contain positive integers")
    if any(not 0 <= value < 1 for value in config.model.target_dropouts):
        raise ValueError("model.target_dropouts must contain values in [0, 1)")
    if config.data.scaler not in {"minmax", "robust", "quantile"}:
        raise ValueError("data.scaler must be 'minmax', 'robust', or 'quantile'")
    if any(not str(value).strip() for value in config.data.target_columns):
        raise ValueError("data.target_columns must contain non-empty names")
    if (config.data.winsorize_low is None) != (config.data.winsorize_high is None):
        raise ValueError("data.winsorize_low and data.winsorize_high must be set together")
    if config.data.winsorize_low is not None and config.data.winsorize_high is not None:
        if not 0 <= config.data.winsorize_low < config.data.winsorize_high <= 1:
            raise ValueError("winsorize quantiles must satisfy 0 <= low < high <= 1")
    if not 0 < config.uncertainty.confidence < 1:
        raise ValueError("uncertainty.confidence must be between 0 and 1")
    if config.training.point_loss not in {"mse", "huber"}:
        raise ValueError("training.point_loss must be 'mse' or 'huber'")
    if config.training.huber_delta <= 0:
        raise ValueError("training.huber_delta must be positive")
    if any(value <= 0 for value in config.training.target_loss_weights):
        raise ValueError("training.target_loss_weights must contain positive values")
    if config.training.relative_loss_weight < 0:
        raise ValueError("training.relative_loss_weight must be non-negative")
    if config.training.relative_loss_epsilon <= 0:
        raise ValueError("training.relative_loss_epsilon must be positive")
    if config.training.reservoir_sample_weight < 0:
        raise ValueError("training.reservoir_sample_weight must be non-negative")
    if config.training.reservoir_phif_threshold < 0:
        raise ValueError("training.reservoir_phif_threshold must be non-negative")
    if config.training.high_sw_sample_weight < 0:
        raise ValueError("training.high_sw_sample_weight must be non-negative")
    if not 0 <= config.training.high_sw_threshold <= 1:
        raise ValueError("training.high_sw_threshold must be in [0, 1]")
    if config.training.weight_decay < 0:
        raise ValueError("training.weight_decay must be non-negative")
    if config.training.lr_patience < 1:
        raise ValueError("training.lr_patience must be positive")
    if not 0 < config.training.lr_factor < 1:
        raise ValueError("training.lr_factor must be between 0 and 1")
    if config.uncertainty.calibration_objective not in {"nll", "interval_score"}:
        raise ValueError(
            "uncertainty.calibration_objective must be 'nll' or 'interval_score'"
        )
    if config.runtime.figure_format not in {"pdf", "svg"}:
        raise ValueError("runtime.figure_format must be 'pdf' or 'svg'")
    if config.runtime.num_workers < 0:
        raise ValueError("runtime.num_workers must be non-negative")
    if config.runtime.persistent_workers and config.runtime.num_workers == 0:
        raise ValueError("runtime.persistent_workers requires num_workers > 0")
    if config.method not in {f"M{index}" for index in range(1, 10)}:
        raise ValueError("method must be one of M1 through M9")
