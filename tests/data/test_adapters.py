from pathlib import Path

import numpy as np
import pandas as pd

from bnn_inversion.data.adapters import export_cleaned_dataset, load_dataset


FIELD_CSV = "鍖哄潡4(绛涢€?.csv"


FIELD_CSV = "区块4(筛选).csv"


def test_field_adapter_uses_fraction_targets_and_log_permeability(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "WELLNUM": [0, 0],
            "DEPTH": [100.0, 100.1],
            "AC": [220.0, 221.0],
            "CAL": [22.0, 22.1],
            "CNL": [20.0, 21.0],
            "DEN": [2.4, 2.5],
            "GR": [80.0, 81.0],
            "RT": [10.0, 11.0],
            "SP": [50.0, 51.0],
            "PERM": [0.1, 10.0],
            "POR": [10.0, 20.0],
            "SW": [40.0, 80.0],
        }
    ).to_csv(tmp_path / FIELD_CSV, index=False)

    result = load_dataset("field", tmp_path)

    assert result.feature_columns == ("GR", "CAL", "SP", "AC", "CNL", "DEN", "RT")
    assert result.target_columns == ("PHIF", "SW", "PERM")
    np.testing.assert_allclose(result.frame["RT"], [1.0, np.log10(11.0)])
    np.testing.assert_allclose(result.frame["PHIF"], [0.1, 0.2])
    np.testing.assert_allclose(result.frame["SW"], [0.4, 0.8])
    np.testing.assert_allclose(result.frame["PERM"], [-1.0, 1.0])


def test_spwla_adapter_logs_rdep_rmed_without_building_rt(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "WELLNUM": [0, 0, 0],
            "DEPTH": [1.0, 2.0, 3.0],
            "CALI": [8.5, 8.6, 8.7],
            "DEN": [2.4, 2.5, 2.6],
            "GR": [30.0, 40.0, 50.0],
            "NEU": [0.1, 0.2, 0.3],
            "RDEP": [4.0, -9999.0, 9.0],
            "RMED": [9.0, 16.0, -9999.0],
            "PHIF": [0.1, 0.2, -9999.0],
            "SW": [0.4, 0.5, -9999.0],
            "VSH": [0.2, 1.2, -9999.0],
        }
    ).to_csv(tmp_path / "train.csv", index=False)

    result = load_dataset("spwla", tmp_path)

    assert result.feature_columns == ("CALI", "DEN", "GR", "NEU", "RDEP", "RMED")
    assert "RT" not in result.frame
    np.testing.assert_allclose(result.frame["RDEP"], [np.log10(4.0)])
    np.testing.assert_allclose(result.frame["RMED"], [np.log10(9.0)])
    assert result.frame["VSH"].isna().tolist() == [False]
    assert result.frame["PHIF"].isna().tolist() == [False]


def test_forward_adapter_concatenates_files_and_uses_filename_as_well(tmp_path: Path) -> None:
    folder = tmp_path / "forward_dataset"
    folder.mkdir()
    row = {
        "#DEPTH": [1500.0],
        "CASE": ["case"],
        "POR": [12.0],
        "Vsh": [25.0],
        "Sw": [70.0],
        "DEN": [2.5],
        "AC": [220.0],
        "CNL": [20.0],
        "RT": [10.0],
        "GR": [80.0],
        "SP": [50.0],
        "CAL": [22.0],
    }
    pd.DataFrame(row).to_csv(folder / "data_random0.csv", index=False)
    pd.DataFrame(row).to_csv(folder / "data_random1.csv", index=False)

    result = load_dataset("forward", tmp_path)

    assert len(result.frame) == 2
    assert result.frame["WELLNUM"].nunique() == 2
    assert result.target_columns == ("PHIF", "SW", "VSH")
    np.testing.assert_allclose(result.frame["RT"], [1.0, 1.0])
    np.testing.assert_allclose(result.frame[["PHIF", "SW", "VSH"]], [[0.12, 0.7, 0.25]] * 2)


def test_field_adapter_drops_missing_and_reservoir_outliers(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "WELLNUM": [0, 0, 0, 0],
            "DEPTH": [100.0, 100.1, 100.2, 100.3],
            "AC": [220.0, 221.0, 222.0, np.nan],
            "CAL": [22.0, 22.1, 22.2, 22.3],
            "CNL": [20.0, 21.0, 22.0, 23.0],
            "DEN": [2.4, 2.5, 2.6, 2.7],
            "GR": [80.0, 81.0, 82.0, 83.0],
            "RT": [10.0, 11.0, 12.0, 13.0],
            "SP": [50.0, 51.0, 52.0, 53.0],
            "PERM": [0.1, 10.0, 1.0, 2.0],
            "POR": [10.0, 4.0, 15.0, 15.0],
            "SW": [40.0, 80.0, 99.5, 50.0],
        }
    ).to_csv(tmp_path / FIELD_CSV, index=False)

    result = load_dataset("field", tmp_path)

    assert len(result.frame) == 1
    np.testing.assert_allclose(result.frame["PHIF"], [0.1])
    np.testing.assert_allclose(result.frame["SW"], [0.4])
    reasons = {record.reason: record.removed_count for record in result.cleaning_audit}
    assert reasons["missing_values"] == 1
    assert reasons["reservoir_thresholds"] == 2


def test_spwla_adapter_drops_missing_and_reservoir_outliers(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "WELLNUM": [0, 0, 0, 0],
            "DEPTH": [1.0, 2.0, 3.0, 4.0],
            "CALI": [8.5, 8.6, 8.7, 8.8],
            "DEN": [2.4, 2.5, 2.6, 2.7],
            "GR": [30.0, 40.0, 50.0, 60.0],
            "NEU": [0.1, 0.2, 0.3, 0.4],
            "RDEP": [4.0, 16.0, 9.0, np.nan],
            "RMED": [9.0, 16.0, 9.0, 4.0],
            "PHIF": [0.1, 0.04, 0.2, 0.2],
            "SW": [0.4, 0.5, 0.995, 0.5],
            "VSH": [0.2, 0.3, 0.4, 0.5],
        }
    ).to_csv(tmp_path / "train.csv", index=False)

    result = load_dataset("spwla", tmp_path)

    assert len(result.frame) == 1
    assert result.frame["PHIF"].tolist() == [0.1]
    reasons = {record.reason: record.removed_count for record in result.cleaning_audit}
    assert reasons["missing_values"] == 1
    assert reasons["reservoir_thresholds"] == 2


def test_export_cleaned_dataset_writes_csv_under_processed_folder(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "WELLNUM": [0, 0],
            "DEPTH": [100.0, 100.1],
            "AC": [220.0, 221.0],
            "CAL": [22.0, 22.1],
            "CNL": [20.0, 21.0],
            "DEN": [2.4, 2.5],
            "GR": [80.0, 81.0],
            "RT": [10.0, 11.0],
            "SP": [50.0, 51.0],
            "PERM": [0.1, 10.0],
            "POR": [10.0, 4.0],
            "SW": [40.0, 80.0],
        }
    ).to_csv(tmp_path / FIELD_CSV, index=False)

    path = export_cleaned_dataset("field", tmp_path)

    assert path == tmp_path / "processed_cleaned" / "field_cleaned.csv"
    exported = pd.read_csv(path)
    assert exported["PHIF"].tolist() == [0.1]


def test_field_jian_adapter_uses_well_name_and_synthetic_depth(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "well_name": ["jian-a", "jian-a", "jian-b"],
            "AC": [220.0, 221.0, 222.0],
            "CAL": [22.0, 22.1, 22.2],
            "CNL": [20.0, 21.0, 22.0],
            "DEN": [2.4, 2.5, 2.6],
            "GR": [80.0, 81.0, 82.0],
            "RT": [10.0, 100.0, 1000.0],
            "PERM": [0.1, 10.0, 1.0],
            "POR": [10.0, 20.0, 15.0],
            "SW": [40.0, 80.0, 50.0],
        }
    ).to_csv(tmp_path / "涧字号.csv", index=False)

    result = load_dataset("field_jian", tmp_path)

    assert result.well_column == "WELLNUM"
    assert result.depth_column == "DEPTH"
    assert result.frame["WELLNUM"].tolist() == ["jian-a", "jian-a", "jian-b"]
    assert result.frame["DEPTH"].tolist() == [0, 1, 0]
    np.testing.assert_allclose(result.frame["RT"], [1.0, 2.0, 3.0])
