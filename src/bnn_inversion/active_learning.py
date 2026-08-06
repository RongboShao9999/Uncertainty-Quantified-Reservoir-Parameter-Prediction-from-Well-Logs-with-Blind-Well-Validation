from __future__ import annotations

from typing import Literal

import numpy as np


Strategy = Literal["random", "epistemic", "inconsistency", "mixed"]


def _normalize_targets(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    normalized = np.full(values.shape, np.nan, dtype=float)
    for column in range(values.shape[1]):
        valid = mask[:, column] & np.isfinite(values[:, column])
        if not valid.any():
            continue
        selected = values[valid, column]
        minimum = float(selected.min())
        maximum = float(selected.max())
        if maximum > minimum:
            normalized[valid, column] = (selected - minimum) / (maximum - minimum)
        else:
            normalized[valid, column] = 0.0
    return normalized


def score_pool(
    epistemic: np.ndarray,
    in_interval: np.ndarray,
    *,
    strategy: Strategy,
    penalty: float = 2.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Score unlabeled samples with per-target normalization and macro averaging."""

    return score_pool_components(
        epistemic,
        in_interval,
        strategy=strategy,
        penalty=penalty,
        mask=mask,
    )["score"]


def score_pool_components(
    epistemic: np.ndarray,
    in_interval: np.ndarray,
    *,
    strategy: Strategy,
    penalty: float = 2.0,
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return active-learning scores plus the macro-averaged score terms."""

    uncertainty = np.asarray(epistemic, dtype=float)
    inside = np.asarray(in_interval, dtype=bool)
    if uncertainty.ndim == 1:
        uncertainty = uncertainty[:, None]
    if inside.ndim == 1:
        inside = inside[:, None]
    if uncertainty.shape != inside.shape:
        raise ValueError("epistemic and in_interval must have identical shapes")
    if penalty < 0:
        raise ValueError("penalty must be non-negative")
    valid = np.isfinite(uncertainty) if mask is None else np.asarray(mask, dtype=bool)
    if valid.shape != uncertainty.shape:
        raise ValueError("mask must have the same shape as epistemic")
    valid &= np.isfinite(uncertainty)
    normalized = _normalize_targets(uncertainty, valid)
    if strategy == "random":
        components = np.where(valid, 0.0, np.nan)
    elif strategy == "epistemic":
        components = normalized
    elif strategy == "inconsistency":
        components = np.where(valid, (~inside).astype(float), np.nan)
    elif strategy == "mixed":
        components = normalized + np.where(
            valid, penalty * (~inside).astype(float), np.nan
        )
    else:
        raise ValueError(f"unknown active-learning strategy: {strategy}")
    counts = np.sum(np.isfinite(components), axis=1)
    totals = np.nansum(components, axis=1)
    scores = np.full(len(uncertainty), -np.inf, dtype=float)
    available = counts > 0
    scores[available] = totals[available] / counts[available]

    epistemic_totals = np.nansum(normalized, axis=1)
    epistemic_component = np.full(len(uncertainty), np.nan, dtype=float)
    epistemic_component[available] = epistemic_totals[available] / counts[available]
    inconsistency_terms = np.where(valid, (~inside).astype(float), np.nan)
    inconsistency_totals = np.nansum(inconsistency_terms, axis=1)
    inconsistency_component = np.full(len(uncertainty), np.nan, dtype=float)
    inconsistency_component[available] = (
        penalty * inconsistency_totals[available] / counts[available]
    )
    return {
        "score": scores,
        "epistemic_component": epistemic_component,
        "inconsistency_component": inconsistency_component,
        "valid_target_count": counts.astype(int),
    }


def select_batch(
    scores: np.ndarray,
    pool_indices: np.ndarray,
    *,
    budget: int,
    random_fraction: float = 0.1,
    seed: int,
) -> np.ndarray:
    """Select high-score and exploration samples without replacement."""

    score_values = np.asarray(scores, dtype=float).reshape(-1)
    indices = np.asarray(pool_indices).reshape(-1)
    if len(score_values) != len(indices):
        raise ValueError("scores and pool_indices must have equal lengths")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("pool_indices must be unique")
    if budget < 1 or budget > len(indices):
        raise ValueError("budget must be between one and the pool size")
    if not 0 <= random_fraction <= 1:
        raise ValueError("random_fraction must be between 0 and 1")
    random_count = min(budget, int(round(budget * random_fraction)))
    high_count = budget - random_count
    order = np.argsort(-score_values, kind="stable")
    high_positions = order[:high_count]
    high_indices = indices[high_positions]
    remaining_mask = np.ones(len(indices), dtype=bool)
    remaining_mask[high_positions] = False
    remaining = indices[remaining_mask]
    rng = np.random.default_rng(seed)
    random_indices = (
        rng.choice(remaining, size=random_count, replace=False)
        if random_count
        else np.asarray([], dtype=indices.dtype)
    )
    return np.concatenate([high_indices, random_indices])
