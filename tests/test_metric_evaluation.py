from pathlib import Path

import pandas as pd

from tools.evaluate_new_metrics import evaluate_experiment_tree, write_metric_figures


def _write_run(root: Path, method: str, budget: int, seed: int, *, rmse: float, picp: float) -> None:
    run = root / method / f"N{budget}" / f"seed{seed}"
    run.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "target": "PHIF",
                "metric_space": "linear",
                "rmse": rmse,
                "mae": rmse / 2,
                "r2": 0.5,
                "picp": picp,
                "mpiw": 0.2,
                "nmpiw": 0.4,
                "interval_score": 0.3,
                "uce": 0.02,
                "nll": 0.1,
                "uncertainty_error_spearman": 0.8,
            }
        ]
    ).to_csv(run / "metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "target": "PHIF",
                "trusted_rate": 0.7,
                "suspect_rate": 0.3,
                "trusted_mae": 0.02,
                "suspect_mae": 0.08,
                "mae_gap": 0.06,
            }
        ]
    ).to_csv(run / "trust_metrics.csv", index=False)


def test_evaluate_experiment_tree_collects_new_metric_outputs(tmp_path: Path) -> None:
    root = tmp_path / "matrix"
    _write_run(root, "M5", 100, 0, rmse=0.40, picp=0.93)
    _write_run(root, "M9", 100, 0, rmse=0.30, picp=0.95)

    result = evaluate_experiment_tree(root)

    assert {"metrics", "trust", "active_efficiency"}.issubset(result)
    assert set(result["metrics"]["method"]) == {"M5", "M9"}
    assert "nmpiw" in result["metrics"].columns
    assert "nll" in result["metrics"].columns
    assert result["trust"].iloc[0]["mae_gap"] == 0.06


def test_write_metric_figures_creates_pdf_and_svg(tmp_path: Path) -> None:
    root = tmp_path / "matrix"
    _write_run(root, "M5", 100, 0, rmse=0.40, picp=0.93)
    _write_run(root, "M9", 100, 0, rmse=0.30, picp=0.95)
    tables = evaluate_experiment_tree(root)

    paths = write_metric_figures(tables, tmp_path / "figures")

    names = {path.name for path in paths}
    assert "metric_point_accuracy.pdf" in names
    assert "metric_point_accuracy.svg" in names
    assert "metric_uncertainty_quality.pdf" in names
    assert "metric_trust_mae_gap.svg" in names
    assert all(path.stat().st_size > 0 for path in paths)
