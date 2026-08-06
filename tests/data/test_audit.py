import numpy as np
import pandas as pd
import pytest

from bnn_inversion.data.audit import (
    fraction_from_source,
    permeability_from_log10,
    permeability_to_log10,
)


def test_percent_and_fraction_units_are_explicit() -> None:
    percent, percent_audit = fraction_from_source(
        pd.Series([0.0, 50.0, 100.0]), "percent", "SW"
    )
    fraction, fraction_audit = fraction_from_source(
        pd.Series([0.0, 0.5, 1.0]), "fraction", "SW"
    )

    assert percent.tolist() == [0.0, 0.5, 1.0]
    assert fraction.tolist() == [0.0, 0.5, 1.0]
    assert percent_audit.conversion == "divide_by_100"
    assert fraction_audit.conversion == "identity"


def test_fraction_rejects_or_masks_out_of_range_values() -> None:
    values = pd.Series([-0.1, 0.5, 1.2])
    with pytest.raises(ValueError, match="VSH"):
        fraction_from_source(values, "fraction", "VSH", invalid="raise")

    masked, audit = fraction_from_source(
        values, "fraction", "VSH", invalid="mask"
    )
    assert masked.isna().tolist() == [True, False, True]
    assert audit.invalid_count == 2


def test_permeability_log10_is_reversible_and_rejects_nonpositive() -> None:
    physical = pd.Series([0.01, 1.0, 100.0])
    transformed, audit = permeability_to_log10(physical)
    np.testing.assert_allclose(transformed, [-2.0, 0.0, 2.0])
    np.testing.assert_allclose(permeability_from_log10(transformed), physical)
    assert audit.unit == "log10_mD"

    with pytest.raises(ValueError, match="strictly positive"):
        permeability_to_log10(pd.Series([1.0, 0.0]))

