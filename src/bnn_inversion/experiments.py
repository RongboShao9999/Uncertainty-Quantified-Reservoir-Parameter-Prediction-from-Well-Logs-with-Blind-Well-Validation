from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from bnn_inversion.active_learning import score_pool_components, select_batch
from bnn_inversion.config import DataConfig, ExperimentConfig
from bnn_inversion.data.adapters import CanonicalFrame, load_dataset
from bnn_inversion.data.preprocessing import TabularPreprocessor
from bnn_inversion.data.splits import SplitIndices, make_splits
from bnn_inversion.data.windows import WindowDataset
from bnn_inversion.models.bayesian import (
    BayesianRegressor,
    bnn_predict,
    elbo_loss,
)
from bnn_inversion.models.bilstm import (
    AttentiveBiLSTMRegressor,
    BiLSTMRegressor,
    ReservoirBiLSTMRegressor,
    TargetAwareBiLSTMRegressor,
)
from bnn_inversion.models.common import (
    heteroscedastic_nll,
    masked_huber,
    masked_mse,
    scaled_masked_huber,
    scaled_masked_mse,
    weighted_relative_masked_huber,
    weighted_scaled_masked_huber,
)
from bnn_inversion.models.mc_dropout import MCDropoutRegressor, mc_predict
from bnn_inversion.training import Trainer, TrainingHistory
from bnn_inversion.types import PredictiveDistribution
from bnn_inversion.uncertainty.fusion import (
    confidence_interval,
    fit_aleatoric_weight,
    fuse_distributions,
    trust_status,
)
from bnn_inversion.uncertainty.metrics import (
    calibration_bin_metrics,
    conformal_scale,
    interval_metrics,
    point_metrics,
    risk_metrics,
    trust_metrics,
    uncertainty_metrics,
)
from bnn_inversion.visualization import write_visualizations


METHOD_REGISTRY = {
    "M1": {"interval": "none", "active": None},
    "M2": {"interval": "mc", "active": None},
    "M3": {"interval": "bnn", "active": None},
    "M4": {"interval": "simple", "active": None},
    "M5": {"interval": "conservative", "active": None},
    "M6": {"interval": "conservative", "active": "random"},
    "M7": {"interval": "conservative", "active": "epistemic"},
    "M8": {"interval": "conservative", "active": "inconsistency"},
    "M9": {"interval": "conservative", "active": "mixed"},
}

FRACTION_TARGETS = {"PHIF", "SW", "VSH"}
FRACTION_METRIC_EPSILON = 0.05


@dataclass(frozen=True)
class ExperimentResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    metrics_path: Path
    output_dir: Path


@dataclass
class _Models:
    point: (
        BiLSTMRegressor
        | AttentiveBiLSTMRegressor
        | TargetAwareBiLSTMRegressor
        | ReservoirBiLSTMRegressor
    )
    mc: MCDropoutRegressor | None = None
    bnn: BayesianRegressor | None = None
    histories: dict[str, TrainingHistory] | None = None


def _point_architecture(config: ExperimentConfig) -> str:
    if config.model.point_architecture != "auto":
        return config.model.point_architecture
    return "target_aware_bilstm" if config.method in {"M5", "M6", "M7", "M8", "M9"} else "bilstm"


def _make_point_model(
    input_size: int,
    targets: int | tuple[str, ...],
    config: ExperimentConfig,
) -> (
    BiLSTMRegressor
    | AttentiveBiLSTMRegressor
    | TargetAwareBiLSTMRegressor
    | ReservoirBiLSTMRegressor
):
    target_names = tuple(targets) if not isinstance(targets, int) else tuple()
    target_count = len(target_names) if target_names else int(targets)
    arguments = dict(
        input_size=input_size,
        targets=target_count,
        hidden_size=config.model.hidden_size,
        layers=config.model.layers,
        dropout=config.model.deterministic_dropout,
    )
    architecture = _point_architecture(config)
    if architecture == "bilstm":
        return BiLSTMRegressor(**arguments)
    if architecture == "attentive_bilstm":
        target_head_hidden_sizes = config.model.target_head_hidden_sizes[:target_count]
        if len(target_head_hidden_sizes) < target_count:
            target_head_hidden_sizes = target_head_hidden_sizes + [
                config.model.hidden_size
            ] * (target_count - len(target_head_hidden_sizes))
        return AttentiveBiLSTMRegressor(
            **arguments,
            target_head_hidden_sizes=target_head_hidden_sizes,
        )
    if architecture == "target_aware_bilstm":
        if not target_names:
            target_names = tuple(f"target_{index}" for index in range(target_count))
        head_sizes = config.model.target_head_hidden_sizes[:target_count]
        if len(head_sizes) < target_count:
            head_sizes = head_sizes + [config.model.hidden_size] * (target_count - len(head_sizes))
        dropouts = config.model.target_dropouts[:target_count]
        if len(dropouts) < target_count:
            dropouts = dropouts + [config.model.deterministic_dropout] * (target_count - len(dropouts))
        return TargetAwareBiLSTMRegressor(
            input_size=input_size,
            targets=target_names,
            hidden_size=config.model.hidden_size,
            layers=config.model.layers,
            dropout=config.model.deterministic_dropout,
            target_head_hidden_sizes=head_sizes,
            target_dropouts=dropouts,
            bounded_targets=tuple(config.model.bounded_targets),
        )
    if architecture == "reservoir_bilstm":
        if not target_names:
            target_names = tuple(f"target_{index}" for index in range(target_count))
        head_sizes = config.model.target_head_hidden_sizes[:target_count]
        if len(head_sizes) < target_count:
            head_sizes = head_sizes + [config.model.hidden_size] * (target_count - len(head_sizes))
        dropouts = config.model.target_dropouts[:target_count]
        if len(dropouts) < target_count:
            dropouts = dropouts + [config.model.deterministic_dropout] * (target_count - len(dropouts))
        return ReservoirBiLSTMRegressor(
            input_size=input_size,
            targets=target_names,
            hidden_size=config.model.hidden_size,
            layers=config.model.layers,
            dropout=config.model.deterministic_dropout,
            target_head_hidden_sizes=head_sizes,
            target_dropouts=dropouts,
            bounded_targets=tuple(config.model.bounded_targets),
        )
    raise ValueError(f"unknown point architecture: {architecture}")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _loader(
    dataset,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    sample_weights: np.ndarray | None = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if sample_weights is not None:
        if len(sample_weights) != len(dataset):
            raise ValueError("sample_weights must match dataset length")
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "generator": generator,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
        kwargs.pop("shuffle")
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(dataset, **kwargs)


def _loader_for_config(
    dataset,
    config: ExperimentConfig,
    *,
    shuffle: bool,
    seed: int,
    sample_weights: np.ndarray | None = None,
) -> DataLoader:
    return _loader(
        dataset,
        config.training.batch_size,
        shuffle=shuffle,
        seed=seed,
        num_workers=config.runtime.num_workers,
        pin_memory=config.runtime.pin_memory,
        persistent_workers=config.runtime.persistent_workers,
        sample_weights=sample_weights,
    )


def _select_targets(canonical: CanonicalFrame, data_config: DataConfig) -> CanonicalFrame:
    if not data_config.target_columns:
        return canonical
    requested = tuple(str(value) for value in data_config.target_columns)
    missing = sorted(set(requested) - set(canonical.target_columns))
    if missing:
        raise ValueError(f"configured target columns are not available: {missing}")
    return CanonicalFrame(
        frame=canonical.frame,
        feature_columns=canonical.feature_columns,
        target_columns=requested,
        well_column=canonical.well_column,
        depth_column=canonical.depth_column,
        audit=canonical.audit,
        dataset=canonical.dataset,
        cleaning_audit=canonical.cleaning_audit,
    )


def _reservoir_sample_weights(
    frame: pd.DataFrame,
    center_indices: np.ndarray,
    *,
    reservoir_sample_weight: float,
    reservoir_phif_threshold: float,
    high_sw_sample_weight: float,
    high_sw_threshold: float,
) -> np.ndarray:
    weights = np.ones(len(center_indices), dtype=np.float32)
    if len(center_indices) == 0:
        return weights
    rows = frame.iloc[np.asarray(center_indices, dtype=int)]
    if reservoir_sample_weight > 0 and "PHIF" in rows:
        phif = pd.to_numeric(rows["PHIF"], errors="coerce").to_numpy(dtype=float)
        weights += np.where(
            np.isfinite(phif) & (phif >= reservoir_phif_threshold),
            reservoir_sample_weight,
            0.0,
        ).astype(np.float32)
    if high_sw_sample_weight > 0 and "SW" in rows:
        sw = pd.to_numeric(rows["SW"], errors="coerce").to_numpy(dtype=float)
        weights += np.where(
            np.isfinite(sw) & (sw >= high_sw_threshold),
            high_sw_sample_weight,
            0.0,
        ).astype(np.float32)
    return weights


def _fit_interval_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    loss_function: Callable[[nn.Module, dict[str, torch.Tensor]], torch.Tensor],
    learning_rate: float,
    epochs: int,
    patience: int,
    device: torch.device,
) -> TrainingHistory:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    training_history: list[float] = []
    validation_history: list[float] = []
    for epoch in range(epochs):
        model.train()
        saw_batch = False
        training_losses: list[float] = []
        for batch in train_loader:
            saw_batch = True
            moved = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model, moved)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite interval-model loss encountered")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            training_losses.append(float(loss.detach().item()))
        if not saw_batch:
            raise ValueError("training loader is empty")
        training_history.append(float(np.mean(training_losses)))
        model.eval()
        validation_losses: list[float] = []
        with torch.no_grad():
            for batch in validation_loader:
                moved = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                value = loss_function(model, moved)
                if not torch.isfinite(value):
                    raise FloatingPointError(
                        "non-finite validation loss encountered"
                    )
                validation_losses.append(float(value.item()))
        if not validation_losses:
            raise ValueError("validation loader is empty")
        validation_loss = float(np.mean(validation_losses))
        validation_history.append(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("interval training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return TrainingHistory(
        training_losses=tuple(training_history),
        validation_losses=tuple(validation_history),
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
    )


def _collect(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    predictor: str,
    samples: int,
) -> tuple[torch.Tensor | PredictiveDistribution, torch.Tensor, torch.Tensor, np.ndarray]:
    means: list[torch.Tensor] = []
    epistemic: list[torch.Tensor] = []
    aleatoric: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    indices: list[np.ndarray] = []
    model.to(device)
    for batch in loader:
        x = batch["x"].to(device)
        if predictor == "point":
            model.eval()
            with torch.no_grad():
                means.append(model(x).cpu())
        elif predictor == "mc":
            distribution = mc_predict(model, x, samples=samples)
            means.append(distribution.mean.cpu())
            epistemic.append(distribution.epistemic_variance.cpu())
            aleatoric.append(distribution.aleatoric_variance.cpu())
        elif predictor == "bnn":
            distribution = bnn_predict(model, x, samples=samples)
            means.append(distribution.mean.cpu())
            epistemic.append(distribution.epistemic_variance.cpu())
            aleatoric.append(distribution.aleatoric_variance.cpu())
        else:
            raise ValueError(f"unknown predictor: {predictor}")
        targets.append(batch["y"].cpu())
        masks.append(batch["mask"].cpu())
        indices.append(batch["index"].cpu().numpy())
    if not means:
        raise ValueError("prediction loader is empty")
    mean = torch.cat(means)
    target = torch.cat(targets)
    mask = torch.cat(masks)
    row_indices = np.concatenate(indices)
    if predictor == "point":
        return mean, target, mask, row_indices
    return (
        PredictiveDistribution(
            mean,
            torch.cat(epistemic),
            torch.cat(aleatoric),
        ),
        target,
        mask,
        row_indices,
    )


def _prepare(
    canonical: CanonicalFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, tuple[str, ...], SplitIndices, TabularPreprocessor]:
    split = make_splits(
        canonical.frame,
        mode=config.data.split_mode,
        seed=config.seed,
        test_fraction=config.data.test_fraction,
        validation_size=config.data.validation_size,
        well_column=canonical.well_column,
    )
    preprocessor = TabularPreprocessor(
        canonical.feature_columns,
        (),
        contamination=None,
        scaler=config.data.scaler,
        winsorize_quantiles=(
            (config.data.winsorize_low, config.data.winsorize_high)
            if config.data.winsorize_low is not None
            and config.data.winsorize_high is not None
            else None
        ),
        random_state=config.seed,
    ).fit(canonical.frame.iloc[split.train])
    transformed = preprocessor.transform(canonical.frame)
    working = canonical.frame.copy().reset_index(drop=True)
    for column in transformed.columns:
        working[column] = transformed[column].to_numpy()
    return working, tuple(transformed.columns), split, preprocessor


def _prepare_transfer_source(
    canonical: CanonicalFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, tuple[str, ...], SplitIndices, TabularPreprocessor]:
    if canonical.well_column not in canonical.frame:
        raise ValueError(f"missing well column: {canonical.well_column}")
    positions = np.arange(len(canonical.frame), dtype=int)
    validation: list[int] = []
    remaining = config.data.validation_size
    grouped = (
        canonical.frame.reset_index()
        .sort_values([canonical.well_column, canonical.depth_column])
        .groupby(canonical.well_column, sort=False)
    )
    for _, group in grouped:
        if remaining <= 0:
            break
        if len(group) <= config.data.window_size:
            continue
        chunk_size = min(config.data.window_size, len(group) - config.data.window_size, remaining)
        if chunk_size <= 0:
            continue
        validation.extend(group["index"].to_numpy(dtype=int)[-chunk_size:].tolist())
        remaining -= chunk_size
    validation_array = np.sort(np.asarray(validation, dtype=int))
    train_array = np.sort(np.setdiff1d(positions, validation_array, assume_unique=False))
    split = SplitIndices(
        train=train_array,
        validation=validation_array,
        test=np.asarray([], dtype=int),
        train_wells=tuple(
            _jsonable_wells(pd.unique(canonical.frame.iloc[train_array][canonical.well_column]))
        ),
        validation_wells=tuple(
            _jsonable_wells(pd.unique(canonical.frame.iloc[validation_array][canonical.well_column]))
        ),
        test_wells=(),
    )
    preprocessor = TabularPreprocessor(
        canonical.feature_columns,
        (),
        contamination=None,
        scaler=config.data.scaler,
        winsorize_quantiles=(
            (config.data.winsorize_low, config.data.winsorize_high)
            if config.data.winsorize_low is not None
            and config.data.winsorize_high is not None
            else None
        ),
        random_state=config.seed,
    ).fit(canonical.frame.iloc[split.train])
    transformed = preprocessor.transform(canonical.frame)
    working = canonical.frame.copy().reset_index(drop=True)
    for column in transformed.columns:
        working[column] = transformed[column].to_numpy()
    return working, tuple(transformed.columns), split, preprocessor


def _make_window_sets(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    targets: tuple[str, ...],
    split: SplitIndices,
    config: ExperimentConfig,
    canonical: CanonicalFrame,
) -> tuple[WindowDataset, WindowDataset, WindowDataset]:
    arguments = dict(
        frame=frame,
        feature_columns=features,
        target_columns=targets,
        window_size=config.data.window_size,
        well_column=canonical.well_column,
        depth_column=canonical.depth_column,
    )
    return (
        WindowDataset(indices=split.train, **arguments),
        WindowDataset(indices=split.validation, **arguments),
        WindowDataset(indices=split.test, **arguments),
    )


def _train_models(
    train_dataset: WindowDataset,
    validation_dataset: WindowDataset,
    labeled_positions: np.ndarray,
    config: ExperimentConfig,
    input_size: int,
    targets: int | tuple[str, ...],
) -> _Models:
    device = _device(config.runtime.device)
    target_count = len(targets) if not isinstance(targets, int) else targets
    train_subset = Subset(train_dataset, labeled_positions.tolist())
    sample_weights = None
    if (
        config.training.reservoir_sample_weight > 0
        or config.training.high_sw_sample_weight > 0
    ):
        center_indices_for_weights = np.asarray(train_dataset.center_indices, dtype=int)[
            labeled_positions
        ]
        sample_weights = _reservoir_sample_weights(
            train_dataset.frame,
            center_indices_for_weights,
            reservoir_sample_weight=config.training.reservoir_sample_weight,
            reservoir_phif_threshold=config.training.reservoir_phif_threshold,
            high_sw_sample_weight=config.training.high_sw_sample_weight,
            high_sw_threshold=config.training.high_sw_threshold,
        )
    train_loader = _loader_for_config(
        train_subset,
        config,
        shuffle=True,
        seed=config.seed,
        sample_weights=sample_weights,
    )
    validation_loader = _loader_for_config(
        validation_dataset,
        config,
        shuffle=False,
        seed=config.seed,
    )
    point = _make_point_model(input_size, targets, config)
    target_scale = torch.ones(target_count, dtype=torch.float32)
    if config.training.target_loss_scaling:
        center_indices = np.asarray(train_dataset.center_indices, dtype=int)[
            labeled_positions
        ]
        labeled_targets = train_dataset.targets_array[center_indices]
        labeled_masks = train_dataset.target_mask_array[center_indices]
        scales: list[float] = []
        for target_index in range(target_count):
            values = labeled_targets[labeled_masks[:, target_index], target_index]
            scale = float(np.std(values)) if len(values) > 1 else 1.0
            scales.append(scale if np.isfinite(scale) and scale > 1e-6 else 1.0)
        target_scale = torch.tensor(scales, dtype=torch.float32)
    target_weights = config.training.target_loss_weights[:target_count]
    if len(target_weights) < target_count:
        target_weights = target_weights + [1.0] * (target_count - len(target_weights))
    target_weight_tensor = torch.tensor(target_weights, dtype=torch.float32)

    def point_loss(prediction, target, mask):
        if config.training.point_loss == "huber":
            if config.training.target_loss_scaling:
                base_loss = weighted_scaled_masked_huber(
                    prediction,
                    target,
                    mask,
                    target_scale,
                    target_weight_tensor,
                    delta=config.training.huber_delta,
                )
            else:
                base_loss = masked_huber(
                    prediction, target, mask, delta=config.training.huber_delta
                )
        elif config.training.target_loss_scaling:
            base_loss = scaled_masked_mse(prediction, target, mask, target_scale)
        else:
            base_loss = masked_mse(prediction, target, mask)
        if config.training.relative_loss_weight <= 0:
            return base_loss
        relative_loss = weighted_relative_masked_huber(
            prediction,
            target,
            mask,
            target_weight_tensor,
            epsilon=config.training.relative_loss_epsilon,
            delta=config.training.huber_delta,
        )
        return base_loss + config.training.relative_loss_weight * relative_loss

    histories: dict[str, TrainingHistory] = {}
    histories["point"] = Trainer(
        epochs=config.training.epochs,
        patience=config.training.early_stopping_patience,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        lr_patience=config.training.lr_patience,
        lr_factor=config.training.lr_factor,
        device=str(device),
    ).fit(
        point,
        train_loader,
        validation_loader,
        point_loss,
    )
    interval = METHOD_REGISTRY[config.method]["interval"]
    models = _Models(point=point, histories=histories)
    if interval in {"mc", "simple", "conservative"}:
        models.mc = MCDropoutRegressor(
            input_size,
            target_count,
            hidden_size=config.model.hidden_size,
            layers=config.model.layers,
            dropout=config.model.mc_dropout,
        )

        def mc_loss(model, batch):
            mean, log_variance = model(batch["x"])
            return heteroscedastic_nll(
                mean, log_variance, batch["y"], batch["mask"]
            )

        histories["mc_dropout"] = _fit_interval_model(
            models.mc,
            train_loader,
            validation_loader,
            loss_function=mc_loss,
            learning_rate=config.training.learning_rate,
            epochs=config.training.epochs,
            patience=config.training.early_stopping_patience,
            device=device,
        )
    if interval in {"bnn", "simple", "conservative"}:
        models.bnn = BayesianRegressor(
            input_size,
            target_count,
            window_size=config.data.window_size,
            hidden_size=config.model.hidden_size,
        )

        def bayesian_loss(model, batch):
            mean, log_variance = model(batch["x"])
            return elbo_loss(
                model,
                mean,
                log_variance,
                batch["y"],
                batch["mask"],
                train_size=len(train_subset),
            )

        histories["bnn"] = _fit_interval_model(
            models.bnn,
            train_loader,
            validation_loader,
            loss_function=bayesian_loss,
            learning_rate=config.training.bnn_learning_rate,
            epochs=config.training.epochs,
            patience=config.training.early_stopping_patience,
            device=device,
        )
    return models


def _interval_prediction(
    models: _Models,
    loader: DataLoader,
    config: ExperimentConfig,
    *,
    lambda_mc: float,
) -> tuple[PredictiveDistribution | None, torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
    device = _device(config.runtime.device)
    point, target, mask, indices = _collect(
        models.point, loader, device=device, predictor="point", samples=1
    )
    interval = METHOD_REGISTRY[config.method]["interval"]
    if interval == "none":
        return None, target, mask, indices, point
    if interval == "mc":
        distribution, _, _, _ = _collect(
            models.mc,
            loader,
            device=device,
            predictor="mc",
            samples=config.uncertainty.mc_samples,
        )
    elif interval == "bnn":
        distribution, _, _, _ = _collect(
            models.bnn,
            loader,
            device=device,
            predictor="bnn",
            samples=config.uncertainty.bnn_samples,
        )
    else:
        mc, _, _, _ = _collect(
            models.mc,
            loader,
            device=device,
            predictor="mc",
            samples=config.uncertainty.mc_samples,
        )
        bnn, _, _, _ = _collect(
            models.bnn,
            loader,
            device=device,
            predictor="bnn",
            samples=config.uncertainty.bnn_samples,
        )
        distribution = fuse_distributions(
            mc,
            bnn,
            lambda_mc=lambda_mc,
            strategy="simple" if interval == "simple" else "conservative",
        )
    return distribution, target, mask, indices, point


def _calibrate(
    models: _Models,
    validation_loader: DataLoader,
    config: ExperimentConfig,
) -> float:
    interval = METHOD_REGISTRY[config.method]["interval"]
    if interval != "conservative":
        return 0.5
    device = _device(config.runtime.device)
    mc, target, mask, _ = _collect(
        models.mc,
        validation_loader,
        device=device,
        predictor="mc",
        samples=config.uncertainty.mc_samples,
    )
    bnn, _, _, _ = _collect(
        models.bnn,
        validation_loader,
        device=device,
        predictor="bnn",
        samples=config.uncertainty.bnn_samples,
    )
    masked_target = target.clone()
    masked_target[~mask] = torch.nan
    return fit_aleatoric_weight(
        mc,
        bnn,
        masked_target,
        objective=config.uncertainty.calibration_objective,
        confidence=config.uncertainty.confidence,
    )


def _uncertainty_dominance(epistemic: float, aleatoric: float) -> str:
    total = epistemic + aleatoric
    if total <= 0 or not np.isfinite(total):
        return "undefined"
    ratio = epistemic / total
    if ratio > 0.6:
        return "epistemic"
    if ratio < 0.4:
        return "aleatoric"
    return "balanced"


def _clip_fraction(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _prediction_frame(
    targets: tuple[str, ...],
    target: torch.Tensor,
    mask: torch.Tensor,
    indices: np.ndarray,
    point: torch.Tensor,
    distribution: PredictiveDistribution | None,
    confidence: float,
    conformal_scales: torch.Tensor | None = None,
    well_ids: np.ndarray | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if distribution is not None:
        lower, upper = confidence_interval(
            distribution.mean, distribution.total_variance, confidence=confidence
        )
        if conformal_scales is not None:
            calibrated_lower = distribution.mean - conformal_scales * torch.sqrt(
                distribution.total_variance.clamp_min(1e-8)
            )
            calibrated_upper = distribution.mean + conformal_scales * torch.sqrt(
                distribution.total_variance.clamp_min(1e-8)
            )
        statuses = trust_status(point, lower, upper)
    for row_number, source_index in enumerate(indices):
        for target_number, target_name in enumerate(targets):
            if not bool(mask[row_number, target_number]):
                continue
            truth_value = float(target[row_number, target_number])
            point_value = float(point[row_number, target_number])
            unit = "fraction"
            variance_space = "fraction^2"
            extra_columns: dict[str, float] = {}
            if target_name == "PERM":
                truth_log10 = truth_value
                point_log10 = point_value
                truth_value = float(np.power(10.0, truth_value))
                point_value = float(np.power(10.0, point_value))
                extra_columns = {
                    "y_true_log10": truth_log10,
                    "point_prediction_log10": point_log10,
                }
                unit = "mD"
                variance_space = "log10_mD^2"
            elif target_name in FRACTION_TARGETS:
                point_value = _clip_fraction(point_value)
            row: dict[str, object] = {
                "source_index": int(source_index),
                "target": target_name,
                "y_true": truth_value,
                "point_prediction": point_value,
                "unit": unit,
                **extra_columns,
            }
            if well_ids is not None:
                row["well"] = str(well_ids[row_number])
            if distribution is not None:
                interval_mean = float(distribution.mean[row_number, target_number])
                lower_value = float(lower[row_number, target_number])
                upper_value = float(upper[row_number, target_number])
                if target_name == "PERM":
                    interval_mean_log10 = interval_mean
                    lower_log10 = lower_value
                    upper_log10 = upper_value
                    interval_mean = float(np.power(10.0, interval_mean))
                    lower_value = float(np.power(10.0, lower_value))
                    upper_value = float(np.power(10.0, upper_value))
                    row.update(
                        {
                            "interval_mean_log10": interval_mean_log10,
                            "lower_log10": lower_log10,
                            "upper_log10": upper_log10,
                        }
                    )
                elif target_name in FRACTION_TARGETS:
                    interval_mean = _clip_fraction(interval_mean)
                    lower_value = _clip_fraction(lower_value)
                    upper_value = _clip_fraction(upper_value)
                epistemic_value = float(
                    distribution.epistemic_variance[row_number, target_number]
                )
                aleatoric_value = float(
                    distribution.aleatoric_variance[row_number, target_number]
                )
                total_value = float(
                    distribution.total_variance[row_number, target_number]
                )
                epistemic_ratio = (
                    epistemic_value / total_value if total_value > 0 else np.nan
                )
                row.update(
                    {
                        "raw_lower": lower_value,
                        "raw_upper": upper_value,
                        "interval_mean": interval_mean,
                        "lower": lower_value,
                        "upper": upper_value,
                        "interval_width": upper_value - lower_value,
                        "epistemic_variance": epistemic_value,
                        "aleatoric_variance": aleatoric_value,
                        "total_variance": total_value,
                        "epistemic_ratio": epistemic_ratio,
                        "uncertainty_dominance": _uncertainty_dominance(
                            epistemic_value, aleatoric_value
                        ),
                        "variance_space": variance_space,
                        "status": str(statuses[row_number, target_number]),
                    }
                )
                if conformal_scales is not None:
                    calibrated_lower_value = float(calibrated_lower[row_number, target_number])
                    calibrated_upper_value = float(calibrated_upper[row_number, target_number])
                    if target_name == "PERM":
                        row.update({
                            "calibrated_lower_log10": calibrated_lower_value,
                            "calibrated_upper_log10": calibrated_upper_value,
                            "calibrated_lower": float(np.power(10.0, calibrated_lower_value)),
                            "calibrated_upper": float(np.power(10.0, calibrated_upper_value)),
                        })
                    else:
                        if target_name in FRACTION_TARGETS:
                            calibrated_lower_value = _clip_fraction(calibrated_lower_value)
                            calibrated_upper_value = _clip_fraction(calibrated_upper_value)
                        row.update({"calibrated_lower": calibrated_lower_value, "calibrated_upper": calibrated_upper_value})
            rows.append(row)
    return pd.DataFrame(rows)


def _metric_frame(
    predictions: pd.DataFrame, *, confidence: float = 0.95
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target_name, group in predictions.groupby("target", sort=False):
        metric_space = "log10" if target_name == "PERM" and {
            "y_true_log10",
            "point_prediction_log10",
        }.issubset(group.columns) else "linear"
        target_values = (
            group["y_true_log10"].to_numpy()
            if metric_space == "log10"
            else group["y_true"].to_numpy()
        )
        point_values = (
            group["point_prediction_log10"].to_numpy()
            if metric_space == "log10"
            else group["point_prediction"].to_numpy()
        )
        row: dict[str, object] = {"target": target_name, "metric_space": metric_space}
        row.update(
            point_metrics(
                target_values,
                point_values,
                epsilon=FRACTION_METRIC_EPSILON
                if target_name in FRACTION_TARGETS
                else 1e-6,
            )
        )
        lower_column = "lower_log10" if metric_space == "log10" and "lower_log10" in group else "lower"
        upper_column = "upper_log10" if metric_space == "log10" and "upper_log10" in group else "upper"
        interval_mean_column = (
            "interval_mean_log10"
            if metric_space == "log10" and "interval_mean_log10" in group
            else "interval_mean"
        )
        if lower_column in group:
            raw_metrics = interval_metrics(target_values, group[lower_column].to_numpy(), group[upper_column].to_numpy(), confidence=confidence)
            row.update(raw_metrics)
            row.update({f"raw_{key}": value for key, value in raw_metrics.items()})
            row.update(
                uncertainty_metrics(
                    target=target_values,
                    prediction=group[interval_mean_column].to_numpy(),
                    epistemic=group["epistemic_variance"].to_numpy(),
                    total=group["total_variance"].to_numpy(),
                )
            )
            risk_groups = group.groupby("well", sort=False) if "well" in group else [("all", group)]
            risk_rows = [risk_metrics(
                target=(part["y_true_log10"].to_numpy() if metric_space == "log10" else part["y_true"].to_numpy()),
                prediction=part[interval_mean_column].to_numpy(),
                total=part["total_variance"].to_numpy(),
            ) for _, part in risk_groups]
            for key in risk_rows[0]:
                values = [item[key] for item in risk_rows if isinstance(item.get(key), (float, int, np.floating))]
                if values:
                    row[key] = float(np.nanmean(values))
            calibrated_lower_column = "calibrated_lower_log10" if metric_space == "log10" else "calibrated_lower"
            calibrated_upper_column = "calibrated_upper_log10" if metric_space == "log10" else "calibrated_upper"
            if calibrated_lower_column in group:
                calibrated = interval_metrics(target_values, group[calibrated_lower_column].to_numpy(), group[calibrated_upper_column].to_numpy(), confidence=confidence)
                row.update({f"calibrated_{key}": value for key, value in calibrated.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_conformal_scales(
    distribution: PredictiveDistribution | None,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    confidence: float,
) -> tuple[torch.Tensor | None, list[int]]:
    if distribution is None:
        return None, []
    scales: list[float] = []
    counts: list[int] = []
    for index in range(target.shape[1]):
        valid = mask[:, index].detach().cpu().numpy().astype(bool)
        scale, count = conformal_scale(
            target=target[:, index].detach().cpu().numpy()[valid],
            mean=distribution.mean[:, index].detach().cpu().numpy()[valid],
            variance=distribution.total_variance[:, index].detach().cpu().numpy()[valid],
            confidence=confidence,
        )
        scales.append(scale)
        counts.append(count)
    return torch.tensor(scales, dtype=distribution.mean.dtype), counts


def _trust_metric_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    if "status" not in predictions:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for target_name, group in predictions.groupby("target", sort=False):
        row: dict[str, object] = {"target": target_name}
        row.update(trust_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _calibration_bin_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"lower", "upper", "total_variance"}
    if not required.issubset(predictions.columns):
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for target_name, group in predictions.groupby("target", sort=False):
        for row in calibration_bin_metrics(
            target=group["y_true"].to_numpy(),
            lower=group["lower"].to_numpy(),
            upper=group["upper"].to_numpy(),
            total=group["total_variance"].to_numpy(),
            bins=10,
        ):
            rows.append({"target": target_name, **row})
    return pd.DataFrame(rows)


def _active_learning_metric_frame(selection_rows: list[dict[str, object]]) -> pd.DataFrame:
    if not selection_rows:
        return pd.DataFrame()
    frame = pd.DataFrame(selection_rows)
    grouped = frame.groupby("round", as_index=False).agg(
        selected_count=("window_position", "count"),
        mean_score=("score", "mean"),
        mean_epistemic_component=("epistemic_component", "mean"),
        mean_inconsistency_component=("inconsistency_component", "mean"),
        random_exploration_count=(
            "selection_mode",
            lambda values: int((values == "random_exploration").sum()),
        ),
    )
    grouped["cumulative_labeled"] = grouped["selected_count"].cumsum()
    grouped["random_exploration_rate"] = (
        grouped["random_exploration_count"] / grouped["selected_count"]
    )
    return grouped


def _write_paper_outputs(
    output: Path,
    predictions: pd.DataFrame,
    selection_rows: list[dict[str, object]] | None = None,
    figure_format: str = "pdf",
) -> None:
    trust = _trust_metric_frame(predictions)
    if not trust.empty:
        trust.to_csv(output / "trust_metrics.csv", index=False)
    calibration = _calibration_bin_frame(predictions)
    if not calibration.empty:
        calibration.to_csv(output / "calibration_bins.csv", index=False)
    active_frame = _active_learning_metric_frame(selection_rows or [])
    active_for_figures: pd.DataFrame | None = None
    if selection_rows:
        active_for_figures = pd.DataFrame(selection_rows)
    if not active_frame.empty:
        active_frame.to_csv(output / "active_learning_metrics.csv", index=False)
    write_visualizations(
        predictions,
        output / "figures",
        active_learning=active_for_figures,
        figure_format=figure_format,
    )


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _history_payload(models: _Models) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, history in (models.histories or {}).items():
        result[name] = {
            "training_losses": list(history.training_losses),
            "validation_losses": list(history.validation_losses),
            "best_epoch": history.best_epoch,
            "best_validation_loss": history.best_validation_loss,
            "learning_rates": list(history.learning_rates),
        }
    return result


def _save_best_models(output: Path, models: _Models) -> None:
    torch.save(models.point.state_dict(), output / "point_model.pt")
    torch.save(models.point.state_dict(), output / "best_point_model.pt")
    if models.mc is not None:
        torch.save(models.mc.state_dict(), output / "mc_dropout_model.pt")
        torch.save(models.mc.state_dict(), output / "best_mc_dropout_model.pt")
    if models.bnn is not None:
        torch.save(models.bnn.state_dict(), output / "bnn_model.pt")
        torch.save(models.bnn.state_dict(), output / "best_bnn_model.pt")


def run_experiment(
    config: ExperimentConfig,
    *,
    frame: CanonicalFrame | None = None,
) -> ExperimentResult:
    if config.method not in METHOD_REGISTRY:
        raise ValueError(f"unknown method: {config.method}")
    if config.data.initial_labels < 1:
        raise ValueError("initial_labels must be positive")
    _set_seed(config.seed)
    canonical = frame or load_dataset(
        config.data.dataset,
        config.data.root,
        feature_profile=config.data.feature_profile,
    )
    canonical = _select_targets(canonical, config.data)
    working, feature_columns, split, preprocessor = _prepare(canonical, config)
    train_dataset, validation_dataset, test_dataset = _make_window_sets(
        working,
        feature_columns,
        canonical.target_columns,
        split,
        config,
        canonical,
    )
    if len(validation_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("split produced no validation or test windows")
    if config.data.initial_labels > len(train_dataset):
        raise ValueError(
            f"initial_labels={config.data.initial_labels} exceeds {len(train_dataset)} training windows"
        )
    rng = np.random.default_rng(config.seed)
    labeled = np.sort(
        rng.choice(
            len(train_dataset),
            size=config.data.initial_labels,
            replace=False,
        )
    )
    validation_loader = _loader_for_config(
        validation_dataset,
        config,
        shuffle=False,
        seed=config.seed,
    )
    selection_rows: list[dict[str, object]] = []
    active_strategy = METHOD_REGISTRY[config.method]["active"]
    if active_strategy:
        for round_number in range(1, config.active_learning.rounds + 1):
            models = _train_models(
                train_dataset,
                validation_dataset,
                labeled,
                config,
                input_size=len(feature_columns),
                targets=canonical.target_columns,
            )
            lambda_mc = _calibrate(models, validation_loader, config)
            pool = np.setdiff1d(
                np.arange(len(train_dataset), dtype=int), labeled, assume_unique=True
            )
            if len(pool) < config.active_learning.batch_budget:
                raise ValueError(
                    "active-learning pool is smaller than the configured batch budget"
                )
            pool_loader = _loader_for_config(
                Subset(train_dataset, pool.tolist()),
                config,
                shuffle=False,
                seed=config.seed + round_number,
            )
            distribution, _, pool_mask, _, point = _interval_prediction(
                models, pool_loader, config, lambda_mc=lambda_mc
            )
            if distribution is None:
                raise RuntimeError("active learning requires an interval distribution")
            lower, upper = confidence_interval(
                distribution.mean,
                distribution.total_variance,
                confidence=config.uncertainty.confidence,
            )
            inside = (point >= lower) & (point <= upper)
            score_details = score_pool_components(
                distribution.epistemic_variance.numpy(),
                inside.numpy(),
                strategy=active_strategy,
                penalty=config.active_learning.inconsistency_penalty,
                mask=pool_mask.numpy(),
            )
            scores = score_details["score"]
            if active_strategy == "random":
                exploration = 1.0
            elif active_strategy == "mixed":
                exploration = config.active_learning.random_fraction
            else:
                exploration = 0.0
            selected = select_batch(
                scores,
                pool,
                budget=config.active_learning.batch_budget,
                random_fraction=exploration,
                seed=config.seed + round_number,
            )
            score_by_position = dict(zip(pool.tolist(), scores.tolist()))
            epistemic_by_position = dict(
                zip(pool.tolist(), score_details["epistemic_component"].tolist())
            )
            inconsistency_by_position = dict(
                zip(pool.tolist(), score_details["inconsistency_component"].tolist())
            )
            count_by_position = dict(
                zip(pool.tolist(), score_details["valid_target_count"].tolist())
            )
            exploration_start = len(selected) - int(
                round(len(selected) * exploration)
            )
            for position in selected:
                selection_index = len(selection_rows) % len(selected)
                selection_rows.append(
                    {
                        "round": round_number,
                        "window_position": int(position),
                        "source_index": int(train_dataset.center_indices[int(position)]),
                        "score": float(score_by_position[int(position)]),
                        "epistemic_component": float(
                            epistemic_by_position[int(position)]
                        ),
                        "inconsistency_component": float(
                            inconsistency_by_position[int(position)]
                        ),
                        "valid_target_count": int(count_by_position[int(position)]),
                        "selection_mode": (
                            "random_exploration"
                            if selection_index >= exploration_start
                            else "high_score"
                        ),
                        "strategy": active_strategy,
                    }
                )
            labeled = np.sort(np.concatenate([labeled, selected.astype(int)]))
    models = _train_models(
        train_dataset,
        validation_dataset,
        labeled,
        config,
        input_size=len(feature_columns),
        targets=canonical.target_columns,
    )
    lambda_mc = _calibrate(models, validation_loader, config)
    validation_distribution, validation_target, validation_mask, _, _ = _interval_prediction(
        models, validation_loader, config, lambda_mc=lambda_mc
    )
    conformal_scales, conformal_counts = _fit_conformal_scales(
        validation_distribution,
        validation_target,
        validation_mask,
        confidence=config.uncertainty.confidence,
    )
    test_loader = _loader_for_config(
        test_dataset,
        config,
        shuffle=False,
        seed=config.seed,
    )
    distribution, target, mask, indices, point = _interval_prediction(
        models, test_loader, config, lambda_mc=lambda_mc
    )
    predictions = _prediction_frame(
        canonical.target_columns,
        target,
        mask,
        indices,
        point,
        distribution,
        config.uncertainty.confidence,
        conformal_scales,
        canonical.frame.iloc[indices][canonical.well_column].to_numpy(),
    )
    metrics = _metric_frame(
        predictions, confidence=config.uncertainty.confidence
    )
    output = config.runtime.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    _write_json(output / "audit.json", [record.to_dict() for record in canonical.audit])
    _write_json(
        output / "cleaning_audit.json",
        [record.to_dict() for record in canonical.cleaning_audit],
    )
    _write_json(
        output / "splits.json",
        {
            "train": split.train.tolist(),
            "validation": split.validation.tolist(),
            "test": split.test.tolist(),
            "initial_labeled_windows": labeled.tolist(),
        },
    )
    preprocessor.save(output / "preprocessor.joblib")
    _save_best_models(output, models)
    _write_json(output / "training_history.json", _history_payload(models))
    if distribution is not None:
        _write_json(
            output / "calibration.json",
            {
                "lambda_mc": lambda_mc,
                "objective": config.uncertainty.calibration_objective,
                "conformal_confidence": config.uncertainty.confidence,
                "conformal_scales": conformal_scales.tolist() if conformal_scales is not None else [],
                "conformal_calibration_count": conformal_counts,
            },
        )
    predictions.to_csv(output / "predictions.csv", index=False)
    if selection_rows:
        pd.DataFrame(selection_rows).to_csv(
            output / "active_learning.csv", index=False
        )
    metrics_path = output / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    _write_paper_outputs(
        output,
        predictions,
        selection_rows,
        figure_format=config.runtime.figure_format,
    )
    _write_json(
        output / "summary.json",
        {
            "method": config.method,
            "seed": config.seed,
            "rows": len(predictions),
            "targets": list(canonical.target_columns),
        },
    )
    return ExperimentResult(predictions, metrics, metrics_path, output)


def _shared_target_frame(
    canonical: CanonicalFrame, targets: tuple[str, ...] = ("PHIF", "SW")
) -> CanonicalFrame:
    missing = sorted(set(targets) - set(canonical.target_columns))
    if missing:
        raise ValueError(
            f"{canonical.dataset} lacks cross-domain targets: {missing}"
        )
    return CanonicalFrame(
        frame=canonical.frame,
        feature_columns=canonical.feature_columns,
        target_columns=targets,
        well_column=canonical.well_column,
        depth_column=canonical.depth_column,
        audit=canonical.audit,
        dataset=canonical.dataset,
        cleaning_audit=canonical.cleaning_audit,
    )


def _shared_targets_between(
    source: CanonicalFrame, target: CanonicalFrame
) -> tuple[str, ...]:
    target_set = set(target.target_columns)
    shared = tuple(column for column in source.target_columns if column in target_set)
    if not shared:
        raise ValueError(
            f"{source.dataset} and {target.dataset} do not share target columns"
        )
    return shared


def _subset_canonical(
    canonical: CanonicalFrame,
    mask: pd.Series | np.ndarray,
    *,
    dataset: str,
) -> CanonicalFrame:
    subset = canonical.frame.loc[mask].copy().reset_index(drop=True)
    if subset.empty:
        raise ValueError(f"{dataset} produced no rows")
    return CanonicalFrame(
        frame=subset,
        feature_columns=canonical.feature_columns,
        target_columns=canonical.target_columns,
        well_column=canonical.well_column,
        depth_column=canonical.depth_column,
        audit=canonical.audit,
        dataset=dataset,
        cleaning_audit=canonical.cleaning_audit,
    )


def _jsonable_wells(values: Iterable[object]) -> list[object]:
    result: list[object] = []
    for value in values:
        result.append(value.item() if hasattr(value, "item") else value)
    return result


def _transfer_frames(
    config: ExperimentConfig, target_dataset: str
) -> tuple[CanonicalFrame, CanonicalFrame, dict[str, object]]:
    if target_dataset == "spwla":
        canonical = load_dataset(
            "spwla",
            config.data.root,
            feature_profile=config.data.feature_profile,
        )
        wells = pd.unique(canonical.frame[canonical.well_column])
        target_wells = (8,) if 8 in set(wells.tolist()) else (wells[-1],)
        target_mask = canonical.frame[canonical.well_column].isin(target_wells)
        source = _subset_canonical(
            canonical,
            ~target_mask,
            dataset="spwla",
        )
        target = _subset_canonical(
            canonical,
            target_mask,
            dataset=f"spwla:well{target_wells[0]}",
        )
        train_wells = pd.unique(source.frame[source.well_column])
        return source, target, {
            "protocol": "same_dataset_holdout",
            "train_wells": _jsonable_wells(train_wells),
            "target_wells": _jsonable_wells(target_wells),
        }
    if target_dataset == "field":
        canonical = load_dataset(
            "field",
            config.data.root,
            feature_profile=config.data.feature_profile,
        )
        wells = sorted(pd.unique(canonical.frame[canonical.well_column]).tolist())
        if len(wells) <= 200:
            raise ValueError(
                "field transfer requires more than 200 wells for post-200 holdout"
            )
        train_candidates = wells[:200]
        target_candidates = wells[200:]
        if len(train_candidates) < 20:
            raise ValueError("field transfer requires at least 20 wells before well 200")
        rng = np.random.default_rng(config.seed)
        selected_source_wells = tuple(
            sorted(rng.choice(train_candidates, size=20, replace=False).tolist())
        )
        target_wells = tuple(rng.choice(target_candidates, size=1, replace=False).tolist())
        source = _subset_canonical(
            canonical,
            canonical.frame[canonical.well_column].isin(selected_source_wells),
            dataset="field:first200_random20",
        )
        target = _subset_canonical(
            canonical,
            canonical.frame[canonical.well_column].isin(target_wells),
            dataset=f"field:post200_well{target_wells[0]}",
        )
        return source, target, {
            "protocol": "field_internal_random_well_transfer",
            "source_candidate_rule": "first_200_wells_sorted_by_well_id",
            "target_candidate_rule": "wells_after_first_200_sorted_by_well_id",
            "selected_source_wells": _jsonable_wells(selected_source_wells),
            "target_wells": _jsonable_wells(target_wells),
        }
    if target_dataset == "forward":
        source = _shared_target_frame(
            load_dataset(
                "field",
                config.data.root,
                feature_profile=config.data.feature_profile,
            )
        )
        target = _shared_target_frame(
            load_dataset(
                "forward",
                config.data.root,
                feature_profile=config.data.feature_profile,
            )
        )
        return source, target, {
            "protocol": "forward_generalization",
            "train_wells": _jsonable_wells(pd.unique(source.frame[source.well_column])),
            "target_wells": _jsonable_wells(pd.unique(target.frame[target.well_column])),
        }
    raise ValueError(f"unknown transfer target dataset: {target_dataset}")


def run_cross_domain_experiment(
    config: ExperimentConfig,
    *,
    source: CanonicalFrame | None = None,
    target: CanonicalFrame | None = None,
    target_dataset: str = "spwla",
) -> ExperimentResult:
    """Train once and evaluate a held-out transfer domain without tuning."""

    if config.method not in METHOD_REGISTRY:
        raise ValueError(f"unknown method: {config.method}")
    if config.data.initial_labels < 1:
        raise ValueError("initial_labels must be positive")
    _set_seed(config.seed)
    transfer_summary: dict[str, object] = {"protocol": "explicit_source_target"}
    if source is None and target is None:
        source_data, target_data, transfer_summary = _transfer_frames(
            config, target_dataset
        )
    else:
        source_loaded = source or load_dataset(
            config.data.dataset,
            config.data.root,
            feature_profile=config.data.feature_profile,
        )
        target_loaded = target or load_dataset(
            target_dataset,
            config.data.root,
            feature_profile=config.data.feature_profile,
        )
        shared_targets = _shared_targets_between(source_loaded, target_loaded)
        source_data = _shared_target_frame(source_loaded, shared_targets)
        target_data = _shared_target_frame(target_loaded, shared_targets)
    if source_data.feature_columns != target_data.feature_columns:
        raise ValueError(
            "source and target feature columns must align exactly for zero-shot transfer"
        )
    source_frame, transformed_features, split, preprocessor = _prepare_transfer_source(
        source_data, config
    )
    train_dataset, validation_dataset, _ = _make_window_sets(
        source_frame,
        transformed_features,
        source_data.target_columns,
        split,
        config,
        source_data,
    )
    if len(validation_dataset) == 0:
        raise ValueError("source split produced no validation windows")
    if config.data.initial_labels > len(train_dataset):
        raise ValueError("initial label count exceeds source training windows")
    rng = np.random.default_rng(config.seed)
    labeled = np.sort(
        rng.choice(
            len(train_dataset), config.data.initial_labels, replace=False
        )
    )
    validation_loader = _loader_for_config(
        validation_dataset,
        config,
        shuffle=False,
        seed=config.seed,
    )
    selection_rows: list[dict[str, object]] = []
    active_strategy = METHOD_REGISTRY[config.method]["active"]
    if active_strategy:
        for round_number in range(1, config.active_learning.rounds + 1):
            models = _train_models(
                train_dataset,
                validation_dataset,
                labeled,
                config,
                input_size=len(transformed_features),
                targets=source_data.target_columns,
            )
            lambda_mc = _calibrate(models, validation_loader, config)
            pool = np.setdiff1d(
                np.arange(len(train_dataset), dtype=int), labeled, assume_unique=True
            )
            if len(pool) < config.active_learning.batch_budget:
                raise ValueError(
                    "active-learning pool is smaller than the configured batch budget"
                )
            pool_loader = _loader_for_config(
                Subset(train_dataset, pool.tolist()),
                config,
                shuffle=False,
                seed=config.seed + round_number,
            )
            distribution, _, pool_mask, _, point = _interval_prediction(
                models, pool_loader, config, lambda_mc=lambda_mc
            )
            if distribution is None:
                raise RuntimeError("active learning requires an interval distribution")
            lower, upper = confidence_interval(
                distribution.mean,
                distribution.total_variance,
                confidence=config.uncertainty.confidence,
            )
            inside = (point >= lower) & (point <= upper)
            score_details = score_pool_components(
                distribution.epistemic_variance.numpy(),
                inside.numpy(),
                strategy=active_strategy,
                penalty=config.active_learning.inconsistency_penalty,
                mask=pool_mask.numpy(),
            )
            scores = score_details["score"]
            if active_strategy == "random":
                exploration = 1.0
            elif active_strategy == "mixed":
                exploration = config.active_learning.random_fraction
            else:
                exploration = 0.0
            selected = select_batch(
                scores,
                pool,
                budget=config.active_learning.batch_budget,
                random_fraction=exploration,
                seed=config.seed + round_number,
            )
            score_by_position = dict(zip(pool.tolist(), scores.tolist()))
            epistemic_by_position = dict(
                zip(pool.tolist(), score_details["epistemic_component"].tolist())
            )
            inconsistency_by_position = dict(
                zip(pool.tolist(), score_details["inconsistency_component"].tolist())
            )
            count_by_position = dict(
                zip(pool.tolist(), score_details["valid_target_count"].tolist())
            )
            exploration_start = len(selected) - int(
                round(len(selected) * exploration)
            )
            for position in selected:
                selection_index = len(selection_rows) % len(selected)
                selection_rows.append(
                    {
                        "round": round_number,
                        "window_position": int(position),
                        "source_index": int(train_dataset.center_indices[int(position)]),
                        "score": float(score_by_position[int(position)]),
                        "epistemic_component": float(
                            epistemic_by_position[int(position)]
                        ),
                        "inconsistency_component": float(
                            inconsistency_by_position[int(position)]
                        ),
                        "valid_target_count": int(count_by_position[int(position)]),
                        "selection_mode": (
                            "random_exploration"
                            if selection_index >= exploration_start
                            else "high_score"
                        ),
                        "strategy": active_strategy,
                    }
                )
            labeled = np.sort(np.concatenate([labeled, selected.astype(int)]))
    models = _train_models(
        train_dataset,
        validation_dataset,
        labeled,
        config,
        input_size=len(transformed_features),
        targets=source_data.target_columns,
    )
    lambda_mc = _calibrate(models, validation_loader, config)
    transformed_target = preprocessor.transform(target_data.frame)
    target_frame = target_data.frame.copy().reset_index(drop=True)
    for column in transformed_target.columns:
        target_frame[column] = transformed_target[column].to_numpy()
    target_windows = WindowDataset(
        target_frame,
        np.arange(len(target_frame), dtype=int),
        transformed_features,
        target_data.target_columns,
        config.data.window_size,
        well_column=target_data.well_column,
        depth_column=target_data.depth_column,
    )
    if len(target_windows) == 0:
        raise ValueError("target domain produced no valid sequence windows")
    target_loader = _loader_for_config(
        target_windows,
        config,
        shuffle=False,
        seed=config.seed,
    )
    distribution, truth, mask, indices, point = _interval_prediction(
        models, target_loader, config, lambda_mc=lambda_mc
    )
    predictions = _prediction_frame(
        target_data.target_columns,
        truth,
        mask,
        indices,
        point,
        distribution,
        config.uncertainty.confidence,
    )
    metrics = _metric_frame(
        predictions, confidence=config.uncertainty.confidence
    )
    output = config.runtime.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    preprocessor.save(output / "preprocessor.joblib")
    predictions.to_csv(output / "predictions.csv", index=False)
    if selection_rows:
        pd.DataFrame(selection_rows).to_csv(
            output / "active_learning.csv", index=False
        )
    metrics_path = output / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    _write_paper_outputs(
        output,
        predictions,
        selection_rows,
        figure_format=config.runtime.figure_format,
    )
    _write_json(
        output / "calibration.json",
        {
            "lambda_mc": lambda_mc,
            "objective": config.uncertainty.calibration_objective,
        },
    )
    _write_json(
        output / "domain_summary.json",
        {
            "source": source_data.dataset,
            "target": target_data.dataset,
            "targets": list(target_data.target_columns),
            "fine_tuned_on_target": False,
            "target_prediction_rows": len(predictions),
            **transfer_summary,
        },
    )
    _save_best_models(output, models)
    _write_json(output / "training_history.json", _history_payload(models))
    return ExperimentResult(predictions, metrics, metrics_path, output)
