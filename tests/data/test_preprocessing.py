from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bnn_inversion.data.preprocessing import TabularPreprocessor


def test_scaler_and_imputer_fit_training_rows_only(tmp_path: Path) -> None:
    train = pd.DataFrame({"GR": [0.0, 10.0], "RT": [1.0, 100.0]})
    test = pd.DataFrame({"GR": [1000.0], "RT": [10.0]})
    preprocessor = TabularPreprocessor(
        feature_columns=("GR", "RT"), log_columns=("RT",)
    ).fit(train)

    transformed = preprocessor.transform(test)

    assert preprocessor.scaler.data_max_[0] == 10.0
    assert transformed["GR"].iloc[0] > 1.0
    assert transformed["RT"].iloc[0] == pytest.approx(0.5)
    path = tmp_path / "preprocessor.joblib"
    preprocessor.save(path)
    restored = TabularPreprocessor.load(path)
    pd.testing.assert_frame_equal(restored.transform(test), transformed)


def test_missing_values_use_training_median_and_have_indicators() -> None:
    train = pd.DataFrame({"GR": [1.0, 3.0], "RT": [1.0, 100.0]})
    test = pd.DataFrame({"GR": [np.nan], "RT": [np.nan]})
    preprocessor = TabularPreprocessor(
        feature_columns=("GR", "RT"), log_columns=("RT",)
    ).fit(train)

    transformed = preprocessor.transform(test)

    assert transformed["GR"].iloc[0] == pytest.approx(0.5)
    assert transformed["RT"].iloc[0] == pytest.approx(0.5)
    assert transformed["GR__missing"].iloc[0] == 1.0
    assert transformed["RT__missing"].iloc[0] == 1.0


def test_schema_drift_is_rejected() -> None:
    preprocessor = TabularPreprocessor(("GR",), ()).fit(pd.DataFrame({"GR": [1.0]}))
    with pytest.raises(ValueError, match="missing feature columns"):
        preprocessor.transform(pd.DataFrame({"DEN": [2.4]}))


def test_robust_scaler_and_winsorize_reduce_outlier_influence() -> None:
    train = pd.DataFrame({"GR": [0.0, 1.0, 2.0, 1000.0]})
    preprocessor = TabularPreprocessor(
        feature_columns=("GR",),
        log_columns=(),
        scaler="robust",
        winsorize_quantiles=(0.0, 0.75),
    ).fit(train)
    unbounded = TabularPreprocessor(
        feature_columns=("GR",),
        log_columns=(),
        scaler="robust",
    ).fit(train)

    transformed = preprocessor.transform(pd.DataFrame({"GR": [1000.0]}))
    raw_transformed = unbounded.transform(pd.DataFrame({"GR": [1000.0]}))

    assert transformed["GR"].iloc[0] < raw_transformed["GR"].iloc[0]

