import numpy as np
import pandas as pd
import pytest

from bnn_inversion.data.splits import make_splits
from bnn_inversion.data.windows import WindowDataset


def _frame(wells: int = 4, rows_per_well: int = 5) -> pd.DataFrame:
    rows = []
    for well in range(wells):
        for depth in range(rows_per_well):
            rows.append(
                {
                    "WELLNUM": well,
                    "DEPTH": float(depth),
                    "GR": well * 10 + depth,
                    "SW": depth / max(rows_per_well - 1, 1),
                }
            )
    return pd.DataFrame(rows)


def test_group_split_has_no_well_overlap_and_is_reproducible() -> None:
    frame = _frame()
    first = make_splits(frame, mode="group_well", seed=7, test_fraction=0.25)
    second = make_splits(frame, mode="group_well", seed=7, test_fraction=0.25)

    assert set(first.train_wells).isdisjoint(first.test_wells)
    assert set(first.validation_wells).isdisjoint(first.test_wells)
    np.testing.assert_array_equal(first.test, second.test)


def test_paper_random_split_is_disjoint() -> None:
    split = make_splits(
        _frame(), mode="paper_random", seed=3, test_fraction=0.2, validation_size=3
    )
    all_indices = np.concatenate([split.train, split.validation, split.test])
    assert len(all_indices) == len(set(all_indices))


def test_windows_never_cross_wells_or_partition_boundaries() -> None:
    frame = _frame(wells=2, rows_per_well=5)
    selected = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    dataset = WindowDataset(
        frame,
        selected,
        feature_columns=("GR",),
        target_columns=("SW",),
        window_size=3,
    )

    assert len(dataset) == 4
    assert dataset.features_array.dtype == np.float32
    assert dataset.targets_array.dtype == np.float32
    assert dataset.target_mask_array.dtype == np.bool_
    assert all(len(set(wells)) == 1 for wells in dataset.window_wells)
    assert all(set(indices).issubset(set(selected)) for indices in dataset.window_indices)
    item = dataset[0]
    assert item["x"].shape == (3, 1)
    assert item["y"].shape == (1,)
    assert item["mask"].dtype.name == "bool" if hasattr(item["mask"].dtype, "name") else str(item["mask"].dtype) == "torch.bool"


def test_window_size_must_be_positive_and_odd() -> None:
    with pytest.raises(ValueError, match="odd"):
        WindowDataset(_frame(), np.arange(5), ("GR",), ("SW",), window_size=2)
