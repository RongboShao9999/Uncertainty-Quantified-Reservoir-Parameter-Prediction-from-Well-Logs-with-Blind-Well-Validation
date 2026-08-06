from __future__ import annotations

import torch


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    boolean_mask = mask.to(dtype=torch.bool)
    if values.shape != boolean_mask.shape:
        raise ValueError("loss values and mask must have identical shapes")
    selected = values[boolean_mask]
    if selected.numel() == 0:
        raise ValueError("loss mask selects no targets")
    return selected.mean()


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    return _masked_mean((prediction - target).square(), mask)


def masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if delta <= 0:
        raise ValueError("delta must be positive")
    error = (prediction - target).abs()
    quadratic = torch.minimum(error, torch.tensor(delta, device=error.device, dtype=error.dtype))
    linear = error - quadratic
    return _masked_mean(0.5 * quadratic.square() + delta * linear, mask)


def scaled_masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if scale.ndim != 1 or scale.shape[0] != prediction.shape[-1]:
        raise ValueError("scale must have one positive value per target")
    safe_scale = scale.to(device=prediction.device, dtype=prediction.dtype).clamp_min(1e-6)
    return _masked_mean(((prediction - target) / safe_scale).square(), mask)


def scaled_masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if scale.ndim != 1 or scale.shape[0] != prediction.shape[-1]:
        raise ValueError("scale must have one positive value per target")
    if delta <= 0:
        raise ValueError("delta must be positive")
    safe_scale = scale.to(device=prediction.device, dtype=prediction.dtype).clamp_min(1e-6)
    scaled_error = ((prediction - target) / safe_scale).abs()
    quadratic = torch.minimum(
        scaled_error,
        torch.tensor(delta, device=scaled_error.device, dtype=scaled_error.dtype),
    )
    linear = scaled_error - quadratic
    return _masked_mean(0.5 * quadratic.square() + delta * linear, mask)


def weighted_scaled_masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scale: torch.Tensor,
    weights: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if scale.ndim != 1 or scale.shape[0] != prediction.shape[-1]:
        raise ValueError("scale must have one positive value per target")
    if weights.ndim != 1 or weights.shape[0] != prediction.shape[-1]:
        raise ValueError("weights must have one positive value per target")
    if torch.any(weights <= 0):
        raise ValueError("weights must be positive")
    if delta <= 0:
        raise ValueError("delta must be positive")
    safe_scale = scale.to(device=prediction.device, dtype=prediction.dtype).clamp_min(1e-6)
    safe_weights = weights.to(device=prediction.device, dtype=prediction.dtype)
    scaled_error = ((prediction - target) / safe_scale).abs()
    quadratic = torch.minimum(
        scaled_error,
        torch.tensor(delta, device=scaled_error.device, dtype=scaled_error.dtype),
    )
    linear = scaled_error - quadratic
    elementwise = (0.5 * quadratic.square() + delta * linear) * safe_weights
    return _masked_mean(elementwise, mask)


def weighted_relative_masked_huber(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    *,
    epsilon: float = 0.05,
    delta: float = 1.0,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    if weights.ndim != 1 or weights.shape[0] != prediction.shape[-1]:
        raise ValueError("weights must have one positive value per target")
    if torch.any(weights <= 0):
        raise ValueError("weights must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if delta <= 0:
        raise ValueError("delta must be positive")
    safe_weights = weights.to(device=prediction.device, dtype=prediction.dtype)
    denominator = target.abs().clamp_min(epsilon)
    relative_error = ((prediction - target) / denominator).abs()
    quadratic = torch.minimum(
        relative_error,
        torch.tensor(delta, device=relative_error.device, dtype=relative_error.dtype),
    )
    linear = relative_error - quadratic
    elementwise = (0.5 * quadratic.square() + delta * linear) * safe_weights
    return _masked_mean(elementwise, mask)


def heteroscedastic_nll(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if not (mean.shape == log_variance.shape == target.shape):
        raise ValueError("mean, log_variance, and target must have identical shapes")
    bounded = log_variance.clamp(-10.0, 5.0)
    elementwise = 0.5 * (bounded + (target - mean).square() * torch.exp(-bounded))
    return _masked_mean(elementwise, mask)

