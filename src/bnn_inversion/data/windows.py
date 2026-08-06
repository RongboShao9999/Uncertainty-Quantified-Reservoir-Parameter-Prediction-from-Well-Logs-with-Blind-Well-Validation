from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset[dict[str, torch.Tensor | object]]):
    """Fixed-length, within-well windows whose rows all belong to one split."""

    def __init__(
        self,
        frame: pd.DataFrame,
        indices: Iterable[int],
        feature_columns: Iterable[str],
        target_columns: Iterable[str],
        window_size: int,
        *,
        well_column: str = "WELLNUM",
        depth_column: str = "DEPTH",
    ) -> None:
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        self.frame = frame.reset_index(drop=False).rename(columns={"index": "__source_index"})
        self.feature_columns = tuple(feature_columns)
        self.target_columns = tuple(target_columns)
        self.window_size = window_size
        self.well_column = well_column
        self.depth_column = depth_column
        required = {
            well_column,
            depth_column,
            *self.feature_columns,
            *self.target_columns,
        }
        missing = sorted(required - set(self.frame.columns))
        if missing:
            raise ValueError(f"window frame is missing columns: {missing}")
        self.features_array = self.frame.loc[:, self.feature_columns].to_numpy(
            dtype=np.float32, copy=True
        )
        targets = self.frame.loc[:, self.target_columns].to_numpy(
            dtype=np.float32, copy=True
        )
        self.target_mask_array = np.isfinite(targets)
        self.targets_array = np.nan_to_num(targets, nan=0.0).astype(
            np.float32, copy=False
        )
        selected = {int(index) for index in indices}
        if any(index < 0 or index >= len(self.frame) for index in selected):
            raise IndexError("window indices contain out-of-range positions")
        self.window_indices: list[tuple[int, ...]] = []
        self.window_wells: list[tuple[object, ...]] = []
        self.center_indices: list[int] = []
        ordered = self.frame.assign(__position=np.arange(len(self.frame))).sort_values(
            [well_column, depth_column], kind="stable"
        )
        for well, group in ordered.groupby(well_column, sort=False, dropna=False):
            positions = group["__position"].to_numpy(dtype=int)
            for start in range(0, len(positions) - window_size + 1):
                window = tuple(int(x) for x in positions[start : start + window_size])
                if all(position in selected for position in window):
                    self.window_indices.append(window)
                    self.window_wells.append(tuple([well] * window_size))
                    self.center_indices.append(window[window_size // 2])

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | object]:
        positions = list(self.window_indices[index])
        center = self.center_indices[index]
        return {
            "x": torch.from_numpy(self.features_array[positions]),
            "y": torch.from_numpy(self.targets_array[center]),
            "mask": torch.from_numpy(self.target_mask_array[center]),
            "index": torch.tensor(center, dtype=torch.long),
            "well": self.frame.iloc[center][self.well_column],
        }
