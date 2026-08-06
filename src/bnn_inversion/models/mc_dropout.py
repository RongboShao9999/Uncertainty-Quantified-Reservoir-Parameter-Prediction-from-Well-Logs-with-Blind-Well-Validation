from __future__ import annotations

import torch
from torch import nn

from bnn_inversion.types import PredictiveDistribution
from .bilstm import BiLSTMEncoder


class MCDropoutRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        targets: int,
        hidden_size: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if targets < 1:
            raise ValueError("targets must be positive")
        self.encoder = BiLSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            layers=layers,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.output_size, 2 * targets)
        self.targets = targets

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.head(self.dropout(self.encoder(x)))
        mean, log_variance = output.split(self.targets, dim=-1)
        return mean, log_variance.clamp(-10.0, 5.0)


def mc_predict(
    model: MCDropoutRegressor,
    x: torch.Tensor,
    *,
    samples: int = 50,
) -> PredictiveDistribution:
    if samples < 2:
        raise ValueError("MC prediction requires at least two samples")
    states = {module: module.training for module in model.modules()}
    means: list[torch.Tensor] = []
    variances: list[torch.Tensor] = []
    try:
        model.eval()
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        with torch.no_grad():
            for _ in range(samples):
                mean, log_variance = model(x)
                means.append(mean)
                variances.append(torch.exp(log_variance))
    finally:
        for module, state in states.items():
            module.training = state
    stacked_means = torch.stack(means)
    stacked_variances = torch.stack(variances)
    return PredictiveDistribution(
        mean=stacked_means.mean(dim=0),
        epistemic_variance=stacked_means.var(dim=0, unbiased=False),
        aleatoric_variance=stacked_variances.mean(dim=0),
    )

