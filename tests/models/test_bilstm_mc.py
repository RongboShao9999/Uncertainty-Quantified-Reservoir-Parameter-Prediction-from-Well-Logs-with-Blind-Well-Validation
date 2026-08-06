import torch

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


def test_bilstm_returns_batch_by_target() -> None:
    model = BiLSTMRegressor(input_size=7, targets=3, hidden_size=8, layers=2)
    output = model(torch.randn(4, 9, 7))
    assert output.shape == (4, 3)


def test_attentive_bilstm_returns_batch_by_target_and_attention() -> None:
    model = AttentiveBiLSTMRegressor(input_size=7, targets=3, hidden_size=8, layers=2)
    x = torch.randn(4, 9, 7)

    output = model(x)
    weights = model.attention_weights(x)

    assert output.shape == (4, 3)
    assert weights.shape == (4, 9)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(4))


def test_attentive_bilstm_uses_target_specific_head_widths() -> None:
    model = AttentiveBiLSTMRegressor(
        input_size=7,
        targets=3,
        hidden_size=8,
        layers=3,
        dropout=0.1,
        target_head_hidden_sizes=[6, 10, 14],
    )

    assert [head[0].out_features for head in model.heads] == [6, 10, 14]
    assert model(torch.randn(2, 11, 7)).shape == (2, 3)


def test_target_aware_bilstm_bounds_fraction_targets_only() -> None:
    model = TargetAwareBiLSTMRegressor(
        input_size=7,
        targets=("PHIF", "SW", "PERM"),
        hidden_size=8,
        layers=2,
        dropout=0.1,
        target_head_hidden_sizes=[6, 6, 6],
    )

    output = model(torch.randn(5, 11, 7))

    assert output.shape == (5, 3)
    assert torch.all((0.0 <= output[:, :2]) & (output[:, :2] <= 1.0))
    assert torch.isfinite(output[:, 2]).all()


def test_target_aware_bilstm_can_bound_selected_targets() -> None:
    model = TargetAwareBiLSTMRegressor(
        input_size=7,
        targets=("PHIF", "SW"),
        hidden_size=8,
        layers=2,
        dropout=0.1,
        target_head_hidden_sizes=[6, 6],
        bounded_targets=("SW",),
    )

    output = model(torch.randn(4, 11, 7))

    assert torch.isfinite(output[:, 0]).all()
    assert torch.all((0.0 <= output[:, 1]) & (output[:, 1] <= 1.0))


def test_reservoir_bilstm_uses_multiscale_target_heads_and_bounds_fraction_targets() -> None:
    model = ReservoirBiLSTMRegressor(
        input_size=7,
        targets=("PHIF", "SW", "PERM"),
        hidden_size=8,
        layers=2,
        dropout=0.1,
        target_head_hidden_sizes=[10, 12, 8],
        target_dropouts=[0.0, 0.0, 0.0],
        bounded_targets=("PHIF", "SW"),
    )
    x = torch.randn(5, 13, 7)

    output = model(x)
    attention = model.attention_weights(x)

    assert output.shape == (5, 3)
    assert attention.shape == (5, 3, 13)
    torch.testing.assert_close(attention.sum(dim=-1), torch.ones(5, 3))
    assert torch.all((0.0 <= output[:, :2]) & (output[:, :2] <= 1.0))
    assert torch.isfinite(output[:, 2]).all()


def test_masked_losses_ignore_missing_targets() -> None:
    truth = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[True, False]])
    point = torch.tensor([[1.0, 9.0]])
    assert masked_mse(point, truth, mask).item() == 1.0

    mean = torch.tensor([[1.0, 9.0]])
    log_variance = torch.zeros_like(mean)
    assert heteroscedastic_nll(mean, log_variance, truth, mask).item() == 0.5


def test_scaled_masked_mse_balances_target_scales() -> None:
    prediction = torch.tensor([[2.0, 20.0]])
    truth = torch.tensor([[1.0, 10.0]])
    mask = torch.tensor([[True, True]])
    scale = torch.tensor([1.0, 10.0])

    assert scaled_masked_mse(prediction, truth, mask, scale).item() == 1.0


def test_scaled_huber_is_less_sensitive_to_large_errors_than_mse() -> None:
    prediction = torch.tensor([[11.0]])
    truth = torch.tensor([[1.0]])
    mask = torch.tensor([[True]])
    scale = torch.tensor([1.0])

    assert masked_huber(prediction, truth, mask, delta=1.0).item() < masked_mse(
        prediction, truth, mask
    ).item()
    assert scaled_masked_huber(prediction, truth, mask, scale, delta=1.0).item() < scaled_masked_mse(
        prediction, truth, mask, scale
    ).item()


def test_weighted_scaled_huber_emphasizes_target_weights() -> None:
    prediction = torch.tensor([[2.0, 2.0]])
    truth = torch.tensor([[1.0, 1.0]])
    mask = torch.tensor([[True, True]])
    scale = torch.tensor([1.0, 1.0])
    weights = torch.tensor([1.0, 3.0])

    weighted = weighted_scaled_masked_huber(
        prediction,
        truth,
        mask,
        scale,
        weights,
        delta=1.0,
    )
    unweighted = scaled_masked_huber(prediction, truth, mask, scale, delta=1.0)

    assert weighted.item() > unweighted.item()


def test_relative_huber_emphasizes_fraction_targets_with_small_denominators() -> None:
    prediction = torch.tensor([[0.08, 0.60, 0.30]])
    truth = torch.tensor([[0.04, 0.50, 0.30]])
    mask = torch.tensor([[True, True, True]])
    weights = torch.tensor([2.0, 2.0, 0.5])

    loss = weighted_relative_masked_huber(
        prediction,
        truth,
        mask,
        weights,
        epsilon=0.05,
        delta=1.0,
    )

    assert loss.item() > 0.0
    assert loss.item() > masked_huber(prediction, truth, mask, delta=1.0).item()


def test_mc_prediction_separates_variances_and_restores_mode() -> None:
    torch.manual_seed(1)
    model = MCDropoutRegressor(
        input_size=7,
        targets=2,
        hidden_size=8,
        layers=1,
        dropout=0.5,
    )
    model.train()
    distribution = mc_predict(model, torch.randn(8, 9, 7), samples=12)

    assert distribution.mean.shape == (8, 2)
    assert torch.any(distribution.epistemic_variance > 0)
    assert torch.all(distribution.aleatoric_variance > 0)
    assert model.training is True


def test_models_reject_wrong_input_shape() -> None:
    model = BiLSTMRegressor(input_size=7, targets=2, hidden_size=8)
    try:
        model(torch.randn(4, 7))
    except ValueError as exc:
        assert "batch, sequence, features" in str(exc)
    else:
        raise AssertionError("wrong input rank was accepted")

