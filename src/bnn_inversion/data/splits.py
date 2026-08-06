from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_wells: tuple[object, ...] = ()
    validation_wells: tuple[object, ...] = ()
    test_wells: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        groups = [set(self.train), set(self.validation), set(self.test)]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise ValueError("train, validation, and test indices must be disjoint")


def make_splits(
    frame: pd.DataFrame,
    *,
    mode: Literal["group_well", "paper_random"],
    seed: int,
    test_fraction: float,
    validation_size: int = 200,
    well_column: str = "WELLNUM",
) -> SplitIndices:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if validation_size < 0:
        raise ValueError("validation_size must be non-negative")
    rng = np.random.default_rng(seed)
    positions = np.arange(len(frame), dtype=int)
    if mode == "paper_random":
        shuffled = rng.permutation(positions)
        n_test = max(1, int(np.ceil(len(frame) * test_fraction)))
        n_test = min(n_test, max(len(frame) - 1, 0))
        remaining = len(frame) - n_test
        n_validation = min(validation_size, max(remaining - 1, 0))
        return SplitIndices(
            train=np.sort(shuffled[n_test + n_validation :]),
            validation=np.sort(shuffled[n_test : n_test + n_validation]),
            test=np.sort(shuffled[:n_test]),
        )
    if mode != "group_well":
        raise ValueError(f"unknown split mode: {mode}")
    if well_column not in frame:
        raise ValueError(f"missing well column: {well_column}")
    wells = pd.unique(frame[well_column])
    if len(wells) < 2:
        raise ValueError("group_well split requires at least two wells")
    shuffled_wells = rng.permutation(wells)
    n_test_wells = max(1, int(np.ceil(len(wells) * test_fraction)))
    n_test_wells = min(n_test_wells, len(wells) - 1)
    test_wells = tuple(shuffled_wells[:n_test_wells].tolist())
    remaining_wells = shuffled_wells[n_test_wells:]
    if len(remaining_wells) >= 2 and validation_size > 0:
        validation_wells = (remaining_wells[0].item() if hasattr(remaining_wells[0], "item") else remaining_wells[0],)
        train_wells_array = remaining_wells[1:]
    else:
        validation_wells = ()
        train_wells_array = remaining_wells
    train_wells = tuple(
        value.item() if hasattr(value, "item") else value for value in train_wells_array
    )
    test_mask = frame[well_column].isin(test_wells).to_numpy()
    validation_mask = frame[well_column].isin(validation_wells).to_numpy()
    train_mask = ~(test_mask | validation_mask)
    return SplitIndices(
        train=positions[train_mask],
        validation=positions[validation_mask],
        test=positions[test_mask],
        train_wells=train_wells,
        validation_wells=validation_wells,
        test_wells=test_wells,
    )

