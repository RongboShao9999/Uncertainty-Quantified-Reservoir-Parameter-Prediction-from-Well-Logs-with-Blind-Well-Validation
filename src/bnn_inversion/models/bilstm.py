from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


FRACTION_TARGETS = {"PHIF", "SW", "VSH"}


class BiLSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1 or layers < 1:
            raise ValueError("input_size, hidden_size, and layers must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.input_size = input_size
        self.output_size = 2 * hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("input must have shape (batch, sequence, features)")
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"expected {self.input_size} features, received {x.shape[-1]}"
            )
        encoded, _ = self.lstm(x)
        return encoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_sequence(x)
        return encoded[:, encoded.shape[1] // 2, :]


class BiLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        targets: int,
        hidden_size: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if targets < 1:
            raise ValueError("targets must be positive")
        self.encoder = BiLSTMEncoder(input_size, hidden_size, layers, dropout)
        self.head = nn.Linear(self.encoder.output_size, targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class AttentiveBiLSTMRegressor(nn.Module):
    """Residual Bi-LSTM point model with multi-context pooling and target heads."""

    def __init__(
        self,
        input_size: int,
        targets: int,
        hidden_size: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        target_head_hidden_sizes: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if targets < 1:
            raise ValueError("targets must be positive")
        if target_head_hidden_sizes is not None and len(target_head_hidden_sizes) != targets:
            raise ValueError("target_head_hidden_sizes must match targets")
        self.encoder = BiLSTMEncoder(input_size, hidden_size, layers, dropout)
        encoded_size = self.encoder.output_size
        combined_size = 4 * encoded_size
        self.attention = nn.Linear(encoded_size, 1)
        self.norm = nn.LayerNorm(combined_size)
        bottleneck_size = max(encoded_size, combined_size // 2)
        self.shared = nn.Sequential(
            nn.Linear(combined_size, bottleneck_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(bottleneck_size),
            nn.Linear(bottleneck_size, combined_size),
            nn.SiLU(),
        )
        head_widths = (
            list(target_head_hidden_sizes)
            if target_head_hidden_sizes is not None
            else [encoded_size for _ in range(targets)]
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(combined_size, int(head_width)),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(int(head_width)),
                    nn.Linear(int(head_width), max(4, int(head_width) // 2)),
                    nn.SiLU(),
                    nn.Linear(max(4, int(head_width) // 2), 1),
                )
                for head_width in head_widths
            ]
        )

    def _features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder.encode_sequence(x)
        weights = torch.softmax(self.attention(encoded).squeeze(-1), dim=1)
        context = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        center = encoded[:, encoded.shape[1] // 2, :]
        mean_context = encoded.mean(dim=1)
        max_context = encoded.max(dim=1).values
        combined = torch.cat([center, context, mean_context, max_context], dim=-1)
        features = self.shared(self.norm(combined)) + combined
        return features, weights

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        _, weights = self._features(x)
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, _ = self._features(x)
        return torch.cat([head(features) for head in self.heads], dim=-1)


class TargetAwareBiLSTMRegressor(nn.Module):
    """Bi-LSTM with independent attention pooling and bounded fraction heads."""

    def __init__(
        self,
        input_size: int,
        targets: tuple[str, ...] | list[str],
        hidden_size: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        target_head_hidden_sizes: list[int] | tuple[int, ...] | None = None,
        target_dropouts: list[float] | tuple[float, ...] | None = None,
        bounded_targets: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        super().__init__()
        if not targets:
            raise ValueError("targets must not be empty")
        self.targets = tuple(targets)
        self.bounded_targets = set(FRACTION_TARGETS if bounded_targets is None else bounded_targets)
        target_count = len(self.targets)
        if target_head_hidden_sizes is not None and len(target_head_hidden_sizes) != target_count:
            raise ValueError("target_head_hidden_sizes must match targets")
        if target_dropouts is not None and len(target_dropouts) != target_count:
            raise ValueError("target_dropouts must match targets")
        self.encoder = BiLSTMEncoder(input_size, hidden_size, layers, dropout)
        encoded_size = self.encoder.output_size
        feature_size = 3 * encoded_size
        widths = (
            list(target_head_hidden_sizes)
            if target_head_hidden_sizes is not None
            else [encoded_size for _ in self.targets]
        )
        dropouts = (
            list(target_dropouts)
            if target_dropouts is not None
            else [dropout for _ in self.targets]
        )
        self.attentions = nn.ModuleList([nn.Linear(encoded_size, 1) for _ in self.targets])
        self.norms = nn.ModuleList([nn.LayerNorm(feature_size) for _ in self.targets])
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_size, int(width)),
                    nn.SiLU(),
                    nn.Dropout(float(head_dropout)),
                    nn.LayerNorm(int(width)),
                    nn.Linear(int(width), max(4, int(width) // 2)),
                    nn.SiLU(),
                    nn.Linear(max(4, int(width) // 2), 1),
                )
                for width, head_dropout in zip(widths, dropouts)
            ]
        )

    def _target_features(self, encoded: torch.Tensor, target_index: int) -> torch.Tensor:
        weights = torch.softmax(self.attentions[target_index](encoded).squeeze(-1), dim=1)
        context = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        center = encoded[:, encoded.shape[1] // 2, :]
        mean_context = encoded.mean(dim=1)
        combined = torch.cat([center, context, mean_context], dim=-1)
        return self.norms[target_index](combined)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder.encode_sequence(x)
        outputs: list[torch.Tensor] = []
        for index, target in enumerate(self.targets):
            value = self.heads[index](self._target_features(encoded, index))
            if target in self.bounded_targets:
                value = torch.sigmoid(value)
            outputs.append(value)
        return torch.cat(outputs, dim=-1)


class ReservoirBiLSTMRegressor(nn.Module):
    """Reservoir-parameter Bi-LSTM with multi-scale pooling and residual target heads."""

    def __init__(
        self,
        input_size: int,
        targets: tuple[str, ...] | list[str],
        hidden_size: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
        target_head_hidden_sizes: list[int] | tuple[int, ...] | None = None,
        target_dropouts: list[float] | tuple[float, ...] | None = None,
        bounded_targets: tuple[str, ...] | list[str] | None = None,
    ) -> None:
        super().__init__()
        if not targets:
            raise ValueError("targets must not be empty")
        self.targets = tuple(targets)
        self.bounded_targets = set(FRACTION_TARGETS if bounded_targets is None else bounded_targets)
        target_count = len(self.targets)
        if target_head_hidden_sizes is not None and len(target_head_hidden_sizes) != target_count:
            raise ValueError("target_head_hidden_sizes must match targets")
        if target_dropouts is not None and len(target_dropouts) != target_count:
            raise ValueError("target_dropouts must match targets")
        self.encoder = BiLSTMEncoder(input_size, hidden_size, layers, dropout)
        encoded_size = self.encoder.output_size
        feature_size = 5 * encoded_size
        widths = (
            list(target_head_hidden_sizes)
            if target_head_hidden_sizes is not None
            else [max(encoded_size, 64) for _ in self.targets]
        )
        dropouts = (
            list(target_dropouts)
            if target_dropouts is not None
            else [dropout for _ in self.targets]
        )
        self.attentions = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(encoded_size, max(8, encoded_size // 2)),
                    nn.Tanh(),
                    nn.Linear(max(8, encoded_size // 2), 1),
                )
                for _ in self.targets
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(feature_size) for _ in self.targets])
        self.residuals = nn.ModuleList(
            [nn.Linear(feature_size, int(width)) for width in widths]
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_size, int(width)),
                    nn.SiLU(),
                    nn.Dropout(float(head_dropout)),
                    nn.LayerNorm(int(width)),
                    nn.Linear(int(width), int(width)),
                    nn.SiLU(),
                    nn.Dropout(float(head_dropout) * 0.5),
                    nn.Linear(int(width), 1),
                )
                for width, head_dropout in zip(widths, dropouts)
            ]
        )
        self.projections = nn.ModuleList([nn.Linear(int(width), 1) for width in widths])

    def _contexts(self, encoded: torch.Tensor, target_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.attentions[target_index](encoded).squeeze(-1), dim=1)
        context = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        center = encoded[:, encoded.shape[1] // 2, :]
        mean_context = encoded.mean(dim=1)
        max_context = encoded.max(dim=1).values
        edge_context = 0.5 * (encoded[:, 0, :] + encoded[:, -1, :])
        combined = torch.cat(
            [center, context, mean_context, max_context, edge_context], dim=-1
        )
        return self.norms[target_index](combined), weights

    def attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder.encode_sequence(x)
        weights = []
        for index in range(len(self.targets)):
            _, target_weights = self._contexts(encoded, index)
            weights.append(target_weights)
        return torch.stack(weights, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder.encode_sequence(x)
        outputs: list[torch.Tensor] = []
        for index, target in enumerate(self.targets):
            features, _ = self._contexts(encoded, index)
            residual = F.silu(self.residuals[index](features))
            value = self.heads[index](features) + self.projections[index](residual)
            if target in self.bounded_targets:
                value = torch.sigmoid(value)
            outputs.append(value)
        return torch.cat(outputs, dim=-1)

