from __future__ import annotations

from statistics import NormalDist
from typing import Literal

import numpy as np
import torch

from bnn_inversion.types import PredictiveDistribution


def _validate_pair(
    mc: PredictiveDistribution, bnn: PredictiveDistribution
) -> None:
    if mc.mean.shape != bnn.mean.shape:
        raise ValueError("MC Dropout and BNN distributions must have identical shapes")


def fuse_distributions(
    mc: PredictiveDistribution,
    bnn: PredictiveDistribution,
    *,
    lambda_mc: float = 0.5,
    strategy: Literal["simple", "conservative"] = "conservative",
) -> PredictiveDistribution:
    _validate_pair(mc, bnn)
    if not 0 <= lambda_mc <= 1:
        raise ValueError("lambda_mc must be between 0 and 1")
    mean = 0.5 * (mc.mean + bnn.mean)
    if strategy == "simple":
        epistemic = 0.5 * (mc.epistemic_variance + bnn.epistemic_variance)
        aleatoric = 0.5 * (mc.aleatoric_variance + bnn.aleatoric_variance)
    elif strategy == "conservative":
        epistemic = torch.maximum(mc.epistemic_variance, bnn.epistemic_variance)
        aleatoric = (
            lambda_mc * mc.aleatoric_variance
            + (1.0 - lambda_mc) * bnn.aleatoric_variance
        )
    else:
        raise ValueError(f"unknown fusion strategy: {strategy}")
    return PredictiveDistribution(mean, epistemic, aleatoric)


def fit_aleatoric_weight(
    mc: PredictiveDistribution,
    bnn: PredictiveDistribution,
    target: torch.Tensor,
    *,
    grid_size: int = 101,
    objective: Literal["nll", "interval_score"] = "nll",
    confidence: float = 0.95,
) -> float:
    _validate_pair(mc, bnn)
    if target.shape != mc.mean.shape:
        raise ValueError("target and predictive distributions must have identical shapes")
    if grid_size < 2:
        raise ValueError("grid_size must be at least two")
    if objective not in {"nll", "interval_score"}:
        raise ValueError("objective must be either 'nll' or 'interval_score'")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    valid = torch.isfinite(target)
    if not valid.any():
        raise ValueError("calibration target contains no finite values")
    best_weight = 0.0
    best_score = float("inf")
    alpha = 1.0 - confidence
    for weight in np.linspace(0.0, 1.0, grid_size):
        fused = fuse_distributions(
            mc, bnn, lambda_mc=float(weight), strategy="conservative"
        )
        variance = fused.total_variance.clamp_min(1e-8)
        if objective == "nll":
            score = 0.5 * (
                torch.log(variance) + (target - fused.mean).square() / variance
            )
        else:
            lower, upper = confidence_interval(
                fused.mean, variance, confidence=confidence
            )
            score = upper - lower
            below = target < lower
            above = target > upper
            score = score + torch.where(
                below, 2.0 / alpha * (lower - target), torch.zeros_like(score)
            )
            score = score + torch.where(
                above, 2.0 / alpha * (target - upper), torch.zeros_like(score)
            )
        value = float(score[valid].mean().item())
        if value < best_score - 1e-12:
            best_score = value
            best_weight = float(weight)
    return best_weight


def confidence_interval(
    mean: torch.Tensor,
    variance: torch.Tensor,
    *,
    confidence: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mean.shape != variance.shape:
        raise ValueError("mean and variance must have identical shapes")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if torch.any(variance < 0):
        raise ValueError("variance must be non-negative")
    z_score = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    radius = z_score * torch.sqrt(variance)
    return mean - radius, mean + radius


def trust_status(
    point_prediction: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> np.ndarray:
    if not (point_prediction.shape == lower.shape == upper.shape):
        raise ValueError("point prediction and interval bounds must have identical shapes")
    inside = (point_prediction >= lower) & (point_prediction <= upper)
    return np.where(inside.detach().cpu().numpy(), "可信", "存疑")

