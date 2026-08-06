from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UnitAudit:
    column: str
    source_unit: str
    unit: str
    conversion: str
    total_count: int
    missing_count: int
    invalid_count: int
    source_min: float | None
    source_max: float | None
    final_min: float | None
    final_max: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _finite_bounds(values: pd.Series) -> tuple[float | None, float | None]:
    finite = values[np.isfinite(values.to_numpy(dtype=float, na_value=np.nan))]
    if finite.empty:
        return None, None
    return float(finite.min()), float(finite.max())


def fraction_from_source(
    series: pd.Series,
    declared_unit: Literal["percent", "fraction"],
    column: str,
    *,
    invalid: Literal["raise", "mask"] = "raise",
) -> tuple[pd.Series, UnitAudit]:
    """Convert an explicitly declared proportion to a validated fraction."""

    source = pd.to_numeric(series, errors="coerce").astype(float)
    converted = source / 100.0 if declared_unit == "percent" else source.copy()
    if declared_unit not in {"percent", "fraction"}:
        raise ValueError(f"unsupported unit for {column}: {declared_unit}")
    invalid_mask = converted.notna() & ~converted.between(0.0, 1.0)
    invalid_count = int(invalid_mask.sum())
    if invalid_count and invalid == "raise":
        bad = converted.loc[invalid_mask].head(5).tolist()
        raise ValueError(
            f"{column} contains {invalid_count} values outside [0, 1]: {bad}"
        )
    if invalid == "mask":
        converted = converted.mask(invalid_mask)
    elif invalid != "raise":
        raise ValueError(f"unknown invalid policy: {invalid}")
    source_min, source_max = _finite_bounds(source)
    final_min, final_max = _finite_bounds(converted)
    audit = UnitAudit(
        column=column,
        source_unit=declared_unit,
        unit="fraction",
        conversion="divide_by_100" if declared_unit == "percent" else "identity",
        total_count=len(source),
        missing_count=int(source.isna().sum()),
        invalid_count=invalid_count,
        source_min=source_min,
        source_max=source_max,
        final_min=final_min,
        final_max=final_max,
    )
    return converted, audit


def positive_to_log10(
    series: pd.Series,
    *,
    column: str,
    source_unit: str,
    unit: str,
) -> tuple[pd.Series, UnitAudit]:
    """Transform positive finite values to base-10 logarithms."""
    source = pd.to_numeric(series, errors="coerce").astype(float)
    invalid_mask = source.notna() & ((source <= 0) | ~np.isfinite(source))
    if invalid_mask.any():
        raise ValueError(
            f"{column} must be finite and strictly positive before log10 transformation"
        )
    transformed = pd.Series(
        np.log10(source.to_numpy(dtype=float)), index=source.index, name=series.name
    )
    source_min, source_max = _finite_bounds(source)
    final_min, final_max = _finite_bounds(transformed)
    audit = UnitAudit(
        column=column,
        source_unit=source_unit,
        unit=unit,
        conversion="log10",
        total_count=len(source),
        missing_count=int(source.isna().sum()),
        invalid_count=int(invalid_mask.sum()),
        source_min=source_min,
        source_max=source_max,
        final_min=final_min,
        final_max=final_max,
    )
    return transformed, audit


def permeability_to_log10(series: pd.Series) -> tuple[pd.Series, UnitAudit]:
    """Transform positive permeability in mD to log10(mD)."""

    return positive_to_log10(
        series, column="PERM", source_unit="mD", unit="log10_mD"
    )


def permeability_from_log10(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(values, dtype=float))

