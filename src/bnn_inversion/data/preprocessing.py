from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler


class TabularPreprocessor:
    """Train-only imputation, log transforms, scaling, and outlier scoring."""

    def __init__(
        self,
        feature_columns: Iterable[str],
        log_columns: Iterable[str],
        *,
        contamination: float | None = None,
        random_state: int = 42,
        scaler: str = "minmax",
        winsorize_quantiles: tuple[float, float] | None = None,
    ) -> None:
        self.feature_columns = tuple(feature_columns)
        self.log_columns = tuple(log_columns)
        unknown_logs = set(self.log_columns) - set(self.feature_columns)
        if unknown_logs:
            raise ValueError(f"log columns are not features: {sorted(unknown_logs)}")
        if contamination is not None and not 0 < contamination < 0.5:
            raise ValueError("contamination must be between 0 and 0.5")
        if scaler not in {"minmax", "robust", "quantile"}:
            raise ValueError("scaler must be 'minmax', 'robust', or 'quantile'")
        if winsorize_quantiles is not None:
            low, high = winsorize_quantiles
            if not 0 <= low < high <= 1:
                raise ValueError("winsorize_quantiles must satisfy 0 <= low < high <= 1")
        self.contamination = contamination
        self.random_state = random_state
        self.medians: pd.Series | None = None
        self.scaler_name = scaler
        self.scaler = self._make_scaler(scaler)
        self.outlier_detector: IsolationForest | None = None
        self.winsorize_quantiles = winsorize_quantiles
        self.winsorize_bounds: tuple[pd.Series, pd.Series] | None = None

    def _make_scaler(self, scaler: str):
        if scaler == "minmax":
            return MinMaxScaler()
        if scaler == "robust":
            return RobustScaler()
        return QuantileTransformer(
            output_distribution="normal",
            random_state=self.random_state,
        )

    def _validate_schema(self, frame: pd.DataFrame) -> None:
        missing = sorted(set(self.feature_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"missing feature columns: {missing}")

    def _numeric_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._validate_schema(frame)
        numeric = frame.loc[:, self.feature_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        for column in self.log_columns:
            positive = numeric[column].where(numeric[column] > 0)
            numeric[column] = np.log10(positive)
        return numeric

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        numeric = self._numeric_features(frame)
        if self.winsorize_quantiles is not None:
            low_q, high_q = self.winsorize_quantiles
            low = numeric.quantile(low_q)
            high = numeric.quantile(high_q)
            self.winsorize_bounds = (low, high)
            numeric = numeric.clip(lower=low, upper=high, axis=1)
        medians = numeric.median(axis=0, skipna=True)
        if medians.isna().any():
            columns = medians.index[medians.isna()].tolist()
            raise ValueError(f"training features contain no finite values: {columns}")
        self.medians = medians
        filled = numeric.fillna(medians)
        self.scaler.fit(filled)
        if self.contamination is not None:
            scaled = self.scaler.transform(filled)
            self.outlier_detector = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
            ).fit(scaled)
        return self

    def _require_fitted(self) -> pd.Series:
        if self.medians is None or not hasattr(self.scaler, "scale_"):
            raise RuntimeError("preprocessor must be fitted before use")
        return self.medians

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        medians = self._require_fitted()
        numeric = self._numeric_features(frame)
        if self.winsorize_bounds is not None:
            low, high = self.winsorize_bounds
            numeric = numeric.clip(lower=low, upper=high, axis=1)
        missing = numeric.isna().astype(float)
        filled = numeric.fillna(medians)
        scaled = pd.DataFrame(
            self.scaler.transform(filled),
            columns=self.feature_columns,
            index=frame.index,
        )
        for column in self.feature_columns:
            scaled[f"{column}__missing"] = missing[column]
        return scaled

    def inlier_mask(self, frame: pd.DataFrame) -> np.ndarray:
        if self.outlier_detector is None:
            return np.ones(len(frame), dtype=bool)
        transformed = self.transform(frame).loc[:, self.feature_columns]
        return self.outlier_detector.predict(transformed) == 1

    def save(self, path: Path | str) -> None:
        self._require_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: Path | str) -> "TabularPreprocessor":
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"expected {cls.__name__}, got {type(loaded).__name__}")
        return loaded

