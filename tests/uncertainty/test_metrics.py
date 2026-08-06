import math

import numpy as np
import pytest

from bnn_inversion.uncertainty.metrics import (
    active_learning_efficiency_metrics,
    calibration_bin_metrics,
    conformal_scale,
    interval_metrics,
    point_metrics,
    risk_metrics,
    trust_metrics,
    uncertainty_metrics,
)


def test_risk_metrics_identify_top_error_samples_from_top_uncertainty() -> None:
    metrics = risk_metrics(
        target=np.zeros(10),
        prediction=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 4.0]),
        total=np.arange(10, dtype=float),
        fraction=0.2,
    )

    assert metrics["risk_count"] == 2.0
    assert metrics["high_error_count"] == 2.0
    assert metrics["risk_precision"] == 1.0
    assert metrics["risk_recall"] == 1.0
    assert metrics["risk_f1"] == 1.0
    assert metrics["risk_error_enrichment"] > 1.0


def test_conformal_scale_uses_finite_sample_corrected_quantile() -> None:
    scale, count = conformal_scale(
        target=np.array([0.0, 1.0, 2.0, 3.0]),
        mean=np.zeros(4),
        variance=np.ones(4),
        confidence=0.75,
    )

    assert count == 4
    assert scale == 3.0


def test_point_metrics_are_perfect_for_exact_prediction() -> None:
    metrics = point_metrics(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    assert metrics["rmse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["mape"] == 0.0
    assert metrics["smape"] == 0.0
    assert metrics["epsilon_mape"] == 0.0
    assert metrics["r2"] == 1.0


def test_point_metrics_report_stable_percentage_errors() -> None:
    metrics = point_metrics(
        np.array([0.0, 0.2, 1.0]),
        np.array([0.02, 0.1, 1.2]),
        epsilon=0.1,
    )

    assert metrics["mape"] == pytest.approx(0.35)
    assert metrics["epsilon_mape"] == pytest.approx(0.3)
    assert metrics["smape"] == pytest.approx(
        np.mean(
            [
                2 * 0.02 / (0.0 + 0.02 + 0.1),
                2 * 0.1 / (0.2 + 0.1 + 0.1),
                2 * 0.2 / (1.0 + 1.2 + 0.1),
            ]
        )
    )


def test_picp_mpiw_and_interval_score() -> None:
    metrics = interval_metrics(
        np.array([0.5, 2.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        confidence=0.95,
        bins=2,
    )
    assert metrics["picp"] == 0.5
    assert metrics["mpiw"] == 1.0
    assert metrics["nmpiw"] == pytest.approx(1.0 / 1.5)
    assert metrics["interval_score"] > 1.0
    assert 0.0 <= metrics["uce"] <= 1.0


def test_uncertainty_metrics_report_nll_and_correlate_with_error() -> None:
    metrics = uncertainty_metrics(
        target=np.array([0.0, 0.0, 0.0]),
        prediction=np.array([0.1, 0.2, 0.3]),
        epistemic=np.array([0.01, 0.02, 0.03]),
        total=np.array([0.02, 0.04, 0.06]),
    )
    assert metrics["epistemic_ratio"] == 0.5
    assert metrics["uncertainty_error_spearman"] == 1.0
    assert metrics["nll"] == pytest.approx(
        np.mean(
            0.5 * np.log(2 * np.pi * np.array([0.02, 0.04, 0.06]))
            + np.array([0.1**2 / (2 * 0.02), 0.2**2 / (2 * 0.04), 0.3**2 / (2 * 0.06)])
        )
    )


def test_metrics_return_nan_reason_when_undefined() -> None:
    result = point_metrics(np.array([1.0]), np.array([1.0]))
    assert math.isnan(result["r2"])
    assert "r2_reason" in result


def test_trust_metrics_compare_trusted_and_suspect_rows() -> None:
    rows = [
        {"target": "PHIF", "y_true": 0.10, "point_prediction": 0.11, "interval_width": 0.10, "status": "可信"},
        {"target": "PHIF", "y_true": 0.20, "point_prediction": 0.30, "interval_width": 0.30, "status": "存疑"},
        {"target": "PHIF", "y_true": 0.30, "point_prediction": 0.35, "interval_width": 0.50, "status": "存疑"},
    ]

    metrics = trust_metrics(rows)

    assert metrics["trusted_rate"] == 1 / 3
    assert metrics["suspect_rate"] == 2 / 3
    assert metrics["trusted_rmse"] == pytest.approx(0.01)
    assert metrics["trusted_mae"] == pytest.approx(0.01)
    assert metrics["suspect_mae"] == pytest.approx((0.10 + 0.05) / 2)
    assert metrics["mae_gap"] == pytest.approx((0.10 + 0.05) / 2 - 0.01)
    assert metrics["suspect_mean_interval_width"] == pytest.approx(0.40)


def test_active_learning_efficiency_metrics_compare_learning_curves() -> None:
    metrics = active_learning_efficiency_metrics(
        candidate_budgets=np.array([100, 150, 200]),
        candidate_rmse=np.array([0.40, 0.32, 0.28]),
        baseline_budgets=np.array([100, 150, 200]),
        baseline_rmse=np.array([0.40, 0.36, 0.34]),
        threshold=0.34,
    )

    assert metrics["delta_rmse_pct"] == pytest.approx(30.0)
    assert metrics["relative_improvement_pct"] == pytest.approx((0.34 - 0.28) / 0.34 * 100)
    assert metrics["aulc"] == pytest.approx((0.40 + 0.32) / 2 * 50 + (0.32 + 0.28) / 2 * 50)
    assert metrics["baseline_aulc"] == pytest.approx((0.40 + 0.36) / 2 * 50 + (0.36 + 0.34) / 2 * 50)
    assert metrics["label_saving_rate"] == pytest.approx((200 - 150) / 200)


def test_calibration_bin_metrics_report_coverage_by_uncertainty() -> None:
    bins = calibration_bin_metrics(
        target=np.array([0.1, 0.2, 0.3, 1.5]),
        lower=np.array([0.0, 0.0, 0.0, 0.0]),
        upper=np.array([1.0, 1.0, 1.0, 1.0]),
        total=np.array([0.1, 0.2, 0.9, 1.0]),
        bins=2,
    )

    assert [row["bin"] for row in bins] == [1, 2]
    assert bins[0]["coverage"] == 1.0
    assert bins[1]["coverage"] == 0.5
