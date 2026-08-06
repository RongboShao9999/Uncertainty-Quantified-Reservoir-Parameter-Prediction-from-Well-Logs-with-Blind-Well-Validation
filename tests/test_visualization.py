from pathlib import Path

import pandas as pd

from bnn_inversion.visualization import write_visualizations


def test_write_visualizations_creates_prediction_and_diagnostic_figures(
    tmp_path: Path,
) -> None:
    predictions = pd.DataFrame(
        {
            "target": ["PHIF", "PHIF", "PHIF", "SW", "SW", "SW"],
            "y_true": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
            "point_prediction": [0.11, 0.18, 0.35, 0.42, 0.45, 0.62],
            "interval_mean": [0.12, 0.19, 0.32, 0.41, 0.48, 0.61],
            "lower": [0.05, 0.10, 0.20, 0.30, 0.35, 0.50],
            "upper": [0.20, 0.30, 0.45, 0.50, 0.60, 0.70],
            "total_variance": [0.01, 0.02, 0.04, 0.01, 0.03, 0.05],
            "status": ["可信", "可信", "存疑", "可信", "存疑", "存疑"],
        }
    )
    active = pd.DataFrame(
        {
            "round": [1, 1, 2, 2],
            "score": [0.5, 0.7, 0.4, 0.6],
            "epistemic_component": [0.2, 0.3, 0.2, 0.2],
            "inconsistency_component": [0.3, 0.4, 0.2, 0.4],
            "selection_mode": ["high_score", "random_exploration", "high_score", "high_score"],
        }
    )

    paths = write_visualizations(predictions, tmp_path, active_learning=active)

    names = {path.name for path in paths}
    assert "prediction_scatter_PHIF.pdf" in names
    assert "interval_coverage_SW.pdf" in names
    assert "uncertainty_error_PHIF.pdf" in names
    assert "calibration_PHIF.pdf" in names
    assert "trust_status.pdf" in names
    assert "active_learning_scores.pdf" in names
    assert all(path.stat().st_size > 0 for path in paths)


def test_write_visualizations_can_emit_svg(tmp_path: Path) -> None:
    predictions = pd.DataFrame(
        {
            "target": ["PHIF", "PHIF"],
            "y_true": [0.10, 0.20],
            "point_prediction": [0.11, 0.19],
            "status": ["可信", "存疑"],
        }
    )

    paths = write_visualizations(predictions, tmp_path, figure_format="svg")

    assert {path.suffix for path in paths} == {".svg"}
    assert (tmp_path / "prediction_scatter_PHIF.svg").exists()
