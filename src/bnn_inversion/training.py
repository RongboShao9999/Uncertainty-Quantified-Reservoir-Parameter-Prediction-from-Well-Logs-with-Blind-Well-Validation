from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
from torch import nn


LossFunction = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]


@dataclass(frozen=True)
class TrainingHistory:
    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    best_epoch: int
    best_validation_loss: float
    learning_rates: tuple[float, ...] = ()


class Trainer:
    def __init__(
        self,
        *,
        epochs: int = 500,
        patience: int = 20,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        max_grad_norm: float = 5.0,
        device: str = "auto",
    ) -> None:
        if epochs < 1 or patience < 1 or learning_rate <= 0:
            raise ValueError("epochs, patience, and learning_rate must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if lr_patience < 1:
            raise ValueError("lr_patience must be positive")
        if not 0 < lr_factor < 1:
            raise ValueError("lr_factor must be between 0 and 1")
        self.epochs = epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.max_grad_norm = max_grad_norm
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

    def _batch_loss(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        loss_fn: LossFunction,
    ) -> torch.Tensor:
        x = batch["x"].to(self.device)
        target = batch["y"].to(self.device)
        mask = batch["mask"].to(self.device)
        output = model(x)
        if not isinstance(output, torch.Tensor):
            raise TypeError("Trainer requires a tensor-output model and three-argument loss")
        loss = loss_fn(output, target, mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss encountered")
        return loss

    def _evaluate(
        self,
        model: nn.Module,
        loader: Iterable[dict[str, torch.Tensor]],
        loss_fn: LossFunction,
    ) -> float:
        model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch in loader:
                losses.append(float(self._batch_loss(model, batch, loss_fn).item()))
        if not losses:
            raise ValueError("validation loader is empty")
        return sum(losses) / len(losses)

    def fit(
        self,
        model: nn.Module,
        train_loader: Iterable[dict[str, torch.Tensor]],
        validation_loader: Iterable[dict[str, torch.Tensor]],
        loss_fn: LossFunction,
    ) -> TrainingHistory:
        model.to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_factor,
            patience=self.lr_patience,
        )
        train_history: list[float] = []
        validation_history: list[float] = []
        learning_rates: list[float] = []
        best_loss = float("inf")
        best_epoch = -1
        best_state: dict[str, torch.Tensor] | None = None
        stale_epochs = 0
        for epoch in range(self.epochs):
            model.train()
            epoch_losses: list[float] = []
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = self._batch_loss(model, batch, loss_fn)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()
                epoch_losses.append(float(loss.detach().item()))
            if not epoch_losses:
                raise ValueError("training loader is empty")
            train_history.append(sum(epoch_losses) / len(epoch_losses))
            validation_loss = self._evaluate(model, validation_loader, loss_fn)
            validation_history.append(validation_loss)
            learning_rates.append(float(optimizer.param_groups[0]["lr"]))
            scheduler.step(validation_loss)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError("training did not produce a valid checkpoint")
        model.load_state_dict(best_state)
        return TrainingHistory(
            training_losses=tuple(train_history),
            validation_losses=tuple(validation_history),
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            learning_rates=tuple(learning_rates),
        )
