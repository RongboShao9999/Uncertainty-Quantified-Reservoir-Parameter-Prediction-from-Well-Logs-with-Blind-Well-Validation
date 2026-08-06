import pytest
import torch
from torch.utils.data import DataLoader

from bnn_inversion.models.common import masked_mse
from bnn_inversion.training import Trainer


def _loader() -> DataLoader:
    rows = [
        {
            "x": torch.tensor([value], dtype=torch.float32),
            "y": torch.tensor([2 * value], dtype=torch.float32),
            "mask": torch.tensor([True]),
        }
        for value in (1.0, 2.0, 3.0, 4.0)
    ]
    return DataLoader(rows, batch_size=2, shuffle=False)


def test_trainer_reduces_loss_and_restores_best_state() -> None:
    torch.manual_seed(2)
    model = torch.nn.Linear(1, 1)
    trainer = Trainer(epochs=80, patience=20, learning_rate=0.05, device="cpu")

    history = trainer.fit(model, _loader(), _loader(), masked_mse)

    assert history.best_validation_loss < history.validation_losses[0]
    prediction = model(torch.tensor([[5.0]])).item()
    assert prediction == pytest.approx(10.0, rel=0.15)


def test_trainer_rejects_nonfinite_loss() -> None:
    model = torch.nn.Linear(1, 1)
    trainer = Trainer(epochs=1, patience=1, device="cpu")

    def nan_loss(output, target, mask):
        return output.sum() * torch.tensor(float("nan"))

    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.fit(model, _loader(), _loader(), nan_loss)


def test_trainer_records_learning_rate_schedule() -> None:
    model = torch.nn.Linear(1, 1)
    trainer = Trainer(
        epochs=4,
        patience=3,
        learning_rate=0.01,
        weight_decay=0.001,
        lr_patience=1,
        lr_factor=0.5,
        device="cpu",
    )

    def constant_loss(output, target, mask):
        return output.sum() * 0.0 + torch.tensor(1.0, requires_grad=True)

    history = trainer.fit(model, _loader(), _loader(), constant_loss)

    assert min(history.learning_rates) < max(history.learning_rates)
    assert history.learning_rates[0] == pytest.approx(0.01)
