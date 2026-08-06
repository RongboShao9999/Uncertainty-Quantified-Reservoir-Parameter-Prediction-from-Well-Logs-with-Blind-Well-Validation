import numpy as np
import torch

from bnn_inversion.types import PredictiveDistribution
from bnn_inversion.uncertainty.fusion import (
    confidence_interval,
    fit_aleatoric_weight,
    fuse_distributions,
    trust_status,
)


def _distribution(mean, epistemic, aleatoric) -> PredictiveDistribution:
    return PredictiveDistribution(
        torch.as_tensor(mean, dtype=torch.float32).clone(),
        torch.as_tensor(epistemic, dtype=torch.float32).clone(),
        torch.as_tensor(aleatoric, dtype=torch.float32).clone(),
    )


def test_conservative_fusion_uses_max_epistemic_and_weighted_aleatoric() -> None:
    mc = _distribution([[1.0]], [[2.0]], [[4.0]])
    bnn = _distribution([[3.0]], [[5.0]], [[8.0]])

    fused = fuse_distributions(mc, bnn, lambda_mc=0.25, strategy="conservative")

    torch.testing.assert_close(fused.mean, torch.tensor([[2.0]]))
    torch.testing.assert_close(fused.epistemic_variance, torch.tensor([[5.0]]))
    torch.testing.assert_close(fused.aleatoric_variance, torch.tensor([[7.0]]))


def test_simple_fusion_averages_both_variances() -> None:
    mc = _distribution([[1.0]], [[2.0]], [[4.0]])
    bnn = _distribution([[3.0]], [[6.0]], [[8.0]])
    fused = fuse_distributions(mc, bnn, strategy="simple")
    torch.testing.assert_close(fused.epistemic_variance, torch.tensor([[4.0]]))
    torch.testing.assert_close(fused.aleatoric_variance, torch.tensor([[6.0]]))


def test_interval_uses_standard_deviation_and_status_is_per_target() -> None:
    lower, upper = confidence_interval(
        mean=torch.tensor([10.0]), variance=torch.tensor([4.0]), confidence=0.95
    )
    torch.testing.assert_close(lower, torch.tensor([6.080072]), atol=1e-5, rtol=0)
    torch.testing.assert_close(upper, torch.tensor([13.919928]), atol=1e-5, rtol=0)
    assert trust_status(torch.tensor([10.0, 20.0]), torch.tensor([9.0, 0.0]), torch.tensor([11.0, 10.0])).tolist() == ["可信", "存疑"]


def test_calibration_weight_prefers_better_validation_likelihood() -> None:
    target = torch.zeros(10, 1)
    mc = _distribution(torch.zeros(10, 1), torch.zeros(10, 1), torch.full((10, 1), 0.1))
    bnn = _distribution(torch.zeros(10, 1), torch.zeros(10, 1), torch.full((10, 1), 4.0))
    weight = fit_aleatoric_weight(mc, bnn, target, grid_size=21)
    assert weight == 1.0


def test_calibration_weight_can_use_interval_score_objective() -> None:
    target = torch.zeros(10, 1)
    mc = _distribution(torch.zeros(10, 1), torch.zeros(10, 1), torch.full((10, 1), 4.0))
    bnn = _distribution(torch.zeros(10, 1), torch.zeros(10, 1), torch.full((10, 1), 0.1))
    weight = fit_aleatoric_weight(
        mc,
        bnn,
        target,
        grid_size=21,
        objective="interval_score",
        confidence=0.95,
    )
    assert weight == 0.0
