from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .audit import UnitAudit, fraction_from_source, permeability_to_log10, positive_to_log10


MAIN_7 = ("GR", "CAL", "SP", "AC", "CNL", "DEN", "RT")
TRANSFER_5 = ("GR", "CAL", "CNL", "DEN", "RT")
SPWLA_6 = ("CALI", "DEN", "GR", "NEU", "RDEP", "RMED")


@dataclass(frozen=True)
class CleaningAudit:
    reason: str
    removed_count: int
    remaining_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalFrame:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    well_column: str
    depth_column: str
    audit: tuple[UnitAudit, ...]
    dataset: str
    cleaning_audit: tuple[CleaningAudit, ...] = ()


def _require(frame: pd.DataFrame, columns: tuple[str, ...], dataset: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{dataset} is missing required columns: {missing}")


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8", encoding_errors="replace", **kwargs)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"empty CSV file: {path}") from exc


def _clean_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    target_columns: tuple[str, ...],
    dataset: str,
    apply_reservoir_thresholds: bool,
    extra_required_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, tuple[CleaningAudit, ...]]:
    working = frame.copy()
    audit: list[CleaningAudit] = []
    required = [*feature_columns, *target_columns, *extra_required_columns]
    before = len(working)
    working = working.dropna(subset=required)
    removed = before - len(working)
    if removed:
        audit.append(
            CleaningAudit(
                reason="missing_values",
                removed_count=int(removed),
                remaining_count=int(len(working)),
            )
        )
    if apply_reservoir_thresholds:
        before = len(working)
        mask = pd.Series(True, index=working.index)
        if "PHIF" in working:
            mask &= working["PHIF"] >= 0.05
        if "SW" in working:
            mask &= working["SW"] <= 0.99
        working = working.loc[mask].copy()
        removed = before - len(working)
        if removed:
            audit.append(
                CleaningAudit(
                    reason="reservoir_thresholds",
                    removed_count=int(removed),
                    remaining_count=int(len(working)),
                )
            )
    if working.empty:
        raise ValueError(f"{dataset} cleaning removed all rows")
    return working.reset_index(drop=True), tuple(audit)


def _load_field(root: Path, feature_profile: str) -> CanonicalFrame:
    path = root / "区块4(筛选).csv"
    preferred_path = root / "区块4(筛选).csv"
    if preferred_path.is_file():
        path = preferred_path
    else:
        matches = sorted(root.glob("区块4*.csv"))
        if matches:
            path = matches[0]
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = _read_csv(path)
    frame = frame.rename(columns={"POR": "PHIF"})
    features = MAIN_7 if feature_profile == "main_7" else TRANSFER_5
    targets = ("PHIF", "SW", "PERM")
    _require(frame, ("WELLNUM", "DEPTH", *features, *targets), "field")
    audit: list[UnitAudit] = []
    for column in ("PHIF", "SW"):
        frame[column], record = fraction_from_source(
            frame[column], "percent", column
        )
        audit.append(record)
    frame["PERM"], perm_audit = permeability_to_log10(frame["PERM"])
    audit.append(perm_audit)
    frame["RT"], rt_audit = positive_to_log10(
        frame["RT"], column="RT", source_unit="ohm_m", unit="log10_ohm_m"
    )
    audit.append(rt_audit)
    frame, cleaning_audit = _clean_frame(
        frame,
        feature_columns=features,
        target_columns=targets,
        dataset="field",
        apply_reservoir_thresholds=True,
    )
    return CanonicalFrame(
        frame=frame,
        feature_columns=features,
        target_columns=targets,
        well_column="WELLNUM",
        depth_column="DEPTH",
        audit=tuple(audit),
        dataset="field",
        cleaning_audit=cleaning_audit,
    )


def _load_field_jian(root: Path, feature_profile: str) -> CanonicalFrame:
    path = root / "涧字号.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = _read_csv(path)
    frame = frame.rename(columns={"well_name": "WELLNUM", "POR": "PHIF"})
    if "DEPTH" not in frame:
        frame["DEPTH"] = frame.groupby("WELLNUM", sort=False).cumcount()
    features = MAIN_7 if feature_profile == "main_7" else TRANSFER_5
    targets = ("PHIF", "SW", "PERM")
    _require(frame, ("WELLNUM", "DEPTH", *features, *targets), "field_jian")
    audit: list[UnitAudit] = []
    for column in ("PHIF", "SW"):
        frame[column], record = fraction_from_source(
            frame[column], "percent", column
        )
        audit.append(record)
    frame["PERM"], perm_audit = permeability_to_log10(frame["PERM"])
    audit.append(perm_audit)
    frame["RT"], rt_audit = positive_to_log10(
        frame["RT"], column="RT", source_unit="ohm_m", unit="log10_ohm_m"
    )
    audit.append(rt_audit)
    frame, cleaning_audit = _clean_frame(
        frame,
        feature_columns=features,
        target_columns=targets,
        dataset="field_jian",
        apply_reservoir_thresholds=True,
    )
    return CanonicalFrame(
        frame=frame,
        feature_columns=features,
        target_columns=targets,
        well_column="WELLNUM",
        depth_column="DEPTH",
        audit=tuple(audit),
        dataset="field_jian",
        cleaning_audit=cleaning_audit,
    )


def _positive_resistivity(frame: pd.DataFrame) -> pd.Series:
    rdep = pd.to_numeric(frame["RDEP"], errors="coerce").where(lambda x: x > 0)
    rmed = pd.to_numeric(frame["RMED"], errors="coerce").where(lambda x: x > 0)
    both = rdep.notna() & rmed.notna()
    result = rdep.combine_first(rmed)
    result.loc[both] = np.sqrt(rdep.loc[both] * rmed.loc[both])
    return result


def _load_spwla(root: Path, feature_profile: str) -> CanonicalFrame:
    path = root / "train.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = _read_csv(path, na_values=[-9999, -9999.0])
    _require(
        frame,
        ("WELLNUM", "DEPTH", "CALI", "DEN", "GR", "NEU", "RDEP", "RMED", "PHIF", "SW", "VSH"),
        "spwla",
    )
    features = SPWLA_6
    targets = ("PHIF", "SW", "VSH")
    audit: list[UnitAudit] = []
    for column in targets:
        frame[column], record = fraction_from_source(
            frame[column], "fraction", column, invalid="mask"
        )
        audit.append(record)
    frame["RDEP"], rdep_audit = positive_to_log10(
        frame["RDEP"], column="RDEP", source_unit="ohm_m", unit="log10_ohm_m"
    )
    audit.append(rdep_audit)
    frame["RMED"], rmed_audit = positive_to_log10(
        frame["RMED"], column="RMED", source_unit="ohm_m", unit="log10_ohm_m"
    )
    audit.append(rmed_audit)
    frame, cleaning_audit = _clean_frame(
        frame,
        feature_columns=features,
        target_columns=targets,
        dataset="spwla",
        apply_reservoir_thresholds=True,
    )
    return CanonicalFrame(
        frame=frame,
        feature_columns=features,
        target_columns=targets,
        well_column="WELLNUM",
        depth_column="DEPTH",
        audit=tuple(audit),
        dataset="spwla",
        cleaning_audit=cleaning_audit,
    )


def _load_forward(root: Path, feature_profile: str) -> CanonicalFrame:
    folder = root / "forward_dataset"
    paths = sorted(folder.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no forward CSV files found in {folder}")
    frames: list[pd.DataFrame] = []
    for path in paths:
        part = _read_csv(path)
        part["WELLNUM"] = path.stem
        frames.append(part)
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.rename(
        columns={"#DEPTH": "DEPTH", "POR": "PHIF", "Sw": "SW", "Vsh": "VSH"}
    )
    features = MAIN_7 if feature_profile == "main_7" else TRANSFER_5
    targets = ("PHIF", "SW", "VSH")
    _require(frame, ("WELLNUM", "DEPTH", *features, *targets), "forward")
    audit: list[UnitAudit] = []
    for column in targets:
        frame[column], record = fraction_from_source(
            frame[column], "percent", column
        )
        audit.append(record)
    frame["RT"], rt_audit = positive_to_log10(
        frame["RT"], column="RT", source_unit="ohm_m", unit="log10_ohm_m"
    )
    audit.append(rt_audit)
    frame, cleaning_audit = _clean_frame(
        frame,
        feature_columns=features,
        target_columns=targets,
        dataset="forward",
        apply_reservoir_thresholds=False,
    )
    return CanonicalFrame(
        frame=frame,
        feature_columns=features,
        target_columns=targets,
        well_column="WELLNUM",
        depth_column="DEPTH",
        audit=tuple(audit),
        dataset="forward",
        cleaning_audit=cleaning_audit,
    )


def load_dataset(
    dataset: Literal["field", "field_jian", "spwla", "forward"],
    root: Path | str,
    *,
    feature_profile: Literal["main_7", "transfer_5"] | None = None,
) -> CanonicalFrame:
    """Load one source into the canonical schema with audited units."""

    data_root = Path(root)
    profile = feature_profile or (
        "transfer_5" if dataset in {"spwla", "field_jian"} else "main_7"
    )
    if profile not in {"main_7", "transfer_5"}:
        raise ValueError(f"unknown feature profile: {profile}")
    loaders = {
        "field": _load_field,
        "field_jian": _load_field_jian,
        "spwla": _load_spwla,
        "forward": _load_forward,
    }
    try:
        loader = loaders[dataset]
    except KeyError as exc:
        raise ValueError(f"unknown dataset: {dataset}") from exc
    return loader(data_root, profile)


def export_cleaned_dataset(
    dataset: Literal["field", "field_jian", "spwla", "forward"],
    root: Path | str,
    *,
    feature_profile: Literal["main_7", "transfer_5"] | None = None,
    folder_name: str = "processed_cleaned",
) -> Path:
    canonical = load_dataset(dataset, root, feature_profile=feature_profile)
    output_dir = Path(root) / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_cleaned.csv"
    canonical.frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path

