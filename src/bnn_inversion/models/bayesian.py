from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from bnn_inversion.types import PredictiveDistribution
from .common import heteroscedastic_nll


class BayesianLinear(nn.Module):
    """Mean-field Gaussian linear layer with a standard-normal prior."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        if in_features < 1 or out_features < 1:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)

    @staticmethod
    def _sample(mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(rho)
        return mu + sigma * torch.randn_like(mu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._sample(self.weight_mu, self.weight_rho)
        bias = self._sample(self.bias_mu, self.bias_rho)
        return F.linear(x, weight, bias)

    @staticmethod
    def _standard_normal_kl(mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        sigma = F.softplus(rho)
        variance = sigma.square()
        return 0.5 * (mu.square() + variance - 1.0 - torch.log(variance)).sum()

    def kl_divergence(self) -> torch.Tensor:
        return self._standard_normal_kl(
            self.weight_mu, self.weight_rho
        ) + self._standard_normal_kl(self.bias_mu, self.bias_rho)


class BayesianRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        targets: int,
        *,
        window_size: int,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        if min(input_size, targets, window_size, hidden_size) < 1:
            raise ValueError("model dimensions must be positive")
        self.input_size = input_size
        self.targets = targets
        self.window_size = window_size
        flattened = input_size * window_size
        self.hidden1 = BayesianLinear(flattened, hidden_size)
        self.hidden2 = BayesianLinear(hidden_size, hidden_size)
        self.output = BayesianLinear(hidden_size, 2 * targets)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected = (self.window_size, self.input_size)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                "expected input shape "
                f"(batch, {self.window_size}, {self.input_size}), got {tuple(x.shape)}"
            )
        hidden = F.silu(self.hidden1(x.flatten(start_dim=1)))
        hidden = F.silu(self.hidden2(hidden))
        output = self.output(hidden)
        mean, log_variance = output.split(self.targets, dim=-1)
        return mean, log_variance.clamp(-10.0, 5.0)

    def kl_divergence(self) -> torch.Tensor:
        return sum(
            (
                module.kl_divergence()
                for module in self.modules()
                if isinstance(module, BayesianLinear)
            ),
            start=torch.zeros((), device=next(self.parameters()).device),
        )


def elbo_loss(
    model: BayesianRegressor,
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    train_size: int,
) -> torch.Tensor:
    if train_size < 1:
        raise ValueError("train_size must be positive")
    likelihood = heteroscedastic_nll(mean, log_variance, target, mask)
    return likelihood + model.kl_divergence() / train_size


def bnn_predict(
    model: BayesianRegressor,
    x: torch.Tensor,
    *,
    samples: int = 50,
) -> PredictiveDistribution:
    if samples < 2:
        raise ValueError("BNN prediction requires at least two samples")
    training = model.training
    means: list[torch.Tensor] = []
    variances: list[torch.Tensor] = []
    try:
        model.eval()
        with torch.no_grad():
            for _ in range(samples):
                mean, log_variance = model(x)
                means.append(mean)
                variances.append(torch.exp(log_variance))
    finally:
        model.train(training)
    stacked_means = torch.stack(means)
    return PredictiveDistribution(
        mean=stacked_means.mean(dim=0),
        epistemic_variance=stacked_means.var(dim=0, unbiased=False),
        aleatoric_variance=torch.stack(variances).mean(dim=0),
    )
