from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PredictiveDistribution:
    """Mean prediction with separated epistemic and aleatoric variance."""

    mean: torch.Tensor
    epistemic_variance: torch.Tensor
    aleatoric_variance: torch.Tensor

    def __post_init__(self) -> None:
        if not (
            self.mean.shape
            == self.epistemic_variance.shape
            == self.aleatoric_variance.shape
        ):
            raise ValueError("mean and variances must have identical shapes")
        if not all(
            torch.isfinite(value).all()
            for value in (
                self.mean,
                self.epistemic_variance,
                self.aleatoric_variance,
            )
        ):
            raise ValueError("mean and variances must be finite")
        if torch.any(self.epistemic_variance < 0) or torch.any(
            self.aleatoric_variance < 0
        ):
            raise ValueError("variances must be non-negative")

    @property
    def total_variance(self) -> torch.Tensor:
        return self.epistemic_variance + self.aleatoric_variance

