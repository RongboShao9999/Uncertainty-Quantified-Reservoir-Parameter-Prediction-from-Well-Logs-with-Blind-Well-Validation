from pathlib import Path

import numpy as np
import pytest

from bnn_inversion.data.adapters import load_dataset


DATA_ROOT = Path("D:/coding/BNN/DATASET")


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="external dataset unavailable")
def test_real_data_adapters_match_audited_contract() -> None:
    field = load_dataset("field", DATA_ROOT)
    spwla = load_dataset("spwla", DATA_ROOT)
    forward = load_dataset("forward", DATA_ROOT)

    assert len(field.frame) == 731_787
    assert len(spwla.frame) == 24_036
    assert len(forward.frame) == 41_740
    field_cleaning = {record.reason: record.removed_count for record in field.cleaning_audit}
    spwla_cleaning = {record.reason: record.removed_count for record in spwla.cleaning_audit}
    assert field_cleaning == {"reservoir_thresholds": 109_846}
    assert spwla_cleaning == {
        "missing_values": 276_985,
        "reservoir_thresholds": 17_946,
    }
    assert field.target_columns == ("PHIF", "SW", "PERM")
    assert spwla.feature_columns == ("CALI", "DEN", "GR", "NEU", "RDEP", "RMED")
    assert spwla.target_columns == ("PHIF", "SW", "VSH")
    assert forward.target_columns == ("PHIF", "SW", "VSH")
    assert field.frame["PHIF"].between(0, 1).all()
    assert field.frame["SW"].between(0, 1).all()
    assert field.frame["PHIF"].min() >= 0.05
    assert field.frame["SW"].max() <= 0.99
    assert np.isfinite(field.frame["PERM"]).all()
    assert spwla.frame["VSH"].dropna().between(0, 1).all()
    assert "RT" not in spwla.frame
    assert np.isfinite(spwla.frame[["RDEP", "RMED"]]).all().all()
    assert spwla.frame["PHIF"].min() >= 0.05
    assert spwla.frame["SW"].max() <= 0.99
    assert forward.frame[["PHIF", "SW", "VSH"]].apply(
        lambda values: values.between(0, 1).all()
    ).all()
    vsh_audit = next(record for record in spwla.audit if record.column == "VSH")
    assert vsh_audit.invalid_count == 371
