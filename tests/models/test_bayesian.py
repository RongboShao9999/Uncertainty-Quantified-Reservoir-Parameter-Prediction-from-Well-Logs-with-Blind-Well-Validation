import torch

from bnn_inversion.models.bayesian import (
    BayesianLinear,
    BayesianRegressor,
    bnn_predict,
    elbo_loss,
)


def test_bayesian_linear_samples_and_reports_positive_kl() -> None:
    torch.manual_seed(3)
    layer = BayesianLinear(3, 2)
    x = torch.ones(4, 3)

    first = layer(x)
    second = layer(x)

    assert not torch.equal(first, second)
    assert layer.kl_divergence().item() > 0


def test_bayesian_regressor_and_prediction_shapes() -> None:
    model = BayesianRegressor(
        input_size=7,
        targets=2,
        window_size=5,
        hidden_size=12,
    )
    x = torch.randn(6, 5, 7)
    mean, log_variance = model(x)
    distribution = bnn_predict(model, x, samples=8)

    assert mean.shape == log_variance.shape == (6, 2)
    assert distribution.mean.shape == (6, 2)
    assert torch.any(distribution.epistemic_variance > 0)
    assert torch.all(distribution.aleatoric_variance > 0)


def test_elbo_adds_kl_scaled_by_train_size() -> None:
    model = BayesianRegressor(2, 1, window_size=3, hidden_size=4)
    x = torch.zeros(2, 3, 2)
    target = torch.zeros(2, 1)
    mask = torch.ones_like(target, dtype=torch.bool)
    mean, log_variance = model(x)

    small = elbo_loss(model, mean, log_variance, target, mask, train_size=2)
    large = elbo_loss(model, mean, log_variance, target, mask, train_size=20)

    assert small > large


def test_bayesian_model_rejects_wrong_window_shape() -> None:
    model = BayesianRegressor(2, 1, window_size=3, hidden_size=4)
    try:
        model(torch.zeros(2, 4, 2))
    except ValueError as exc:
        assert "expected input shape" in str(exc)
    else:
        raise AssertionError("wrong window shape was accepted")
