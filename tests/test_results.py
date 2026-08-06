from pathlib import Path

import pandas as pd
import pytest

from bnn_inversion.results import summarize_matrix_results


def _write_metrics(path: Path, mae: float, picp: float | None = None) -> None:
    path.parent.mkdir(parents=True)
    row = {
        "target": "PHIF",
        "metric_space": "linear",
        "rmse": mae * 2,
        "mae": mae,
        "epsilon_mape": mae * 10,
        "r2": 0.5,
    }
    if picp is not None:
        row.update({"picp": picp, "mpiw": 0.2, "uce": 0.03})
    pd.DataFrame([row]).to_csv(path, index=False)


def test_summarize_matrix_results_uses_directory_budget_and_skips_seed0(
    tmp_path: Path,
) -> None:
    root = tmp_path / "matrix"
    _write_metrics(root / "M1" / "N50" / "seed0" / "metrics.csv", 0.50)
    _write_metrics(root / "M1" / "N50" / "seed1" / "metrics.csv", 0.10)
    _write_metrics(root / "M1" / "N50" / "seed2" / "metrics.csv", 0.30)
    _write_metrics(root / "M9" / "N500" / "seed1" / "metrics.csv", 0.20, picp=0.95)
    _write_metrics(root / "M9" / "N500" / "seed2" / "metrics.csv", 0.40, picp=0.99)

    result = summarize_matrix_results(root, tmp_path / "summary", exclude_seeds=(0,))

    assert (tmp_path / "summary" / "matrix_metrics_long.csv").exists()
    assert (tmp_path / "summary" / "matrix_metrics_summary.csv").exists()
    assert set(result.long["seed"]) == {1, 2}
    m1 = result.summary[
        (result.summary["method"] == "M1") & (result.summary["budget_N"] == 50)
    ].iloc[0]
    assert m1["mae_mean"] == 0.20
    assert m1["mae_std"] == pytest.approx(0.1414213562373095)
    assert m1["run_count"] == 2
    m9 = result.summary[
        (result.summary["method"] == "M9") & (result.summary["budget_N"] == 500)
    ].iloc[0]
    assert m9["picp_mean"] == 0.97
