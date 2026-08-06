from pathlib import Path

import pandas as pd
import pytest

from bnn_inversion.results import (
    collect_supplementary_metrics,
    summarize_supplementary_results,
    validate_supplementary_budgets,
)


def _write_metrics(path: Path, *, rmse: float = 0.1, picp: float = 0.9) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
                "interval_score": 0.3,
                "uce": 0.04,
            }
        ]
    ).to_csv(path, index=False)


def test_collect_supplementary_metrics_parses_all_sections(tmp_path: Path) -> None:
    _write_metrics(tmp_path / "active/field/M9/N200/seed0/metrics.csv", picp=0.91)
    _write_metrics(
        tmp_path / "calibration/spwla/nll/M5/N100/seed1/metrics.csv"
    )
    _write_metrics(
        tmp_path / "backbone/forward/bilstm/N500/seed2/metrics.csv"
    )
    _write_metrics(
        tmp_path / "transfer/field_to_forward/M5/seed3/metrics.csv"
    )

    result = collect_supplementary_metrics(tmp_path)

    assert set(result["section"]) == {
        "active",
        "calibration",
        "backbone",
        "transfer",
    }
    active = result[result["section"] == "active"].iloc[0]
    assert active["dataset"] == "field"
    assert active["method"] == "M9"
    assert active["budget_N"] == 200
    assert active["seed"] == 0
    assert active["coverage_error"] == pytest.approx(0.04)
    transfer = result[result["section"] == "transfer"].iloc[0]
    assert transfer["protocol"] == "field_to_forward"
    assert transfer["dataset"] == "forward"


def test_collect_supplementary_metrics_rejects_duplicate_identity(
    tmp_path: Path,
) -> None:
    first = tmp_path / "active/field/M9/N200/seed0/metrics.csv"
    _write_metrics(first)
    rows = pd.read_csv(first)
    pd.concat([rows, rows], ignore_index=True).to_csv(first, index=False)

    with pytest.raises(ValueError, match="duplicate supplementary result"):
        collect_supplementary_metrics(tmp_path)


def test_collect_supplementary_metrics_requires_perm_metric_space(
    tmp_path: Path,
) -> None:
    path = tmp_path / "active/field/M5/N100/seed0/metrics.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame([{"target": "PERM", "rmse": 0.3, "metric_space": None}]).to_csv(
        path, index=False
    )

    with pytest.raises(ValueError, match="PERM metric_space is required"):
        collect_supplementary_metrics(tmp_path)


def test_validate_active_budget_reads_acquired_label_count(tmp_path: Path) -> None:
    metrics = tmp_path / "active/field/M9/N200/seed0/metrics.csv"
    _write_metrics(metrics)
    pd.DataFrame(
        [
            {"round": 1, "cumulative_labeled": 20},
            {"round": 5, "cumulative_labeled": 100},
        ]
    ).to_csv(metrics.with_name("active_learning_metrics.csv"), index=False)
    long = collect_supplementary_metrics(tmp_path)

    valid = validate_supplementary_budgets(tmp_path, long)
    assert valid.iloc[0]["initial_labels"] == 100
    assert valid.iloc[0]["expected_added"] == 100
    assert valid.iloc[0]["observed_added"] == 100
    assert valid.iloc[0]["status"] == "valid"

    audit = pd.read_csv(metrics.with_name("active_learning_metrics.csv"))
    audit.loc[audit.index[-1], "cumulative_labeled"] = 95
    audit.to_csv(metrics.with_name("active_learning_metrics.csv"), index=False)
    invalid = validate_supplementary_budgets(tmp_path, long)
    assert invalid.iloc[0]["status"] == "invalid"


def test_validate_static_m5_budget_without_acquisition_audit(tmp_path: Path) -> None:
    _write_metrics(tmp_path / "active/field/M5/N200/seed0/metrics.csv")
    long = collect_supplementary_metrics(tmp_path)

    result = validate_supplementary_budgets(tmp_path, long)

    assert result.iloc[0]["initial_labels"] == 200
    assert result.iloc[0]["expected_added"] == 0
    assert result.iloc[0]["status"] == "valid"


def test_supplementary_summary_has_deterministic_bootstrap_and_single_seed_nan(
    tmp_path: Path,
) -> None:
    for seed, baseline, candidate in (
        (0, 0.30, 0.20),
        (1, 0.40, 0.25),
        (2, 0.50, 0.35),
    ):
        _write_metrics(
            tmp_path / f"backbone/field/bilstm/N100/seed{seed}/metrics.csv",
            rmse=baseline,
        )
        _write_metrics(
            tmp_path
            / f"backbone/field/target_aware_bilstm/N100/seed{seed}/metrics.csv",
            rmse=candidate,
        )
    _write_metrics(
        tmp_path / "calibration/spwla/nll/M5/N100/seed0/metrics.csv",
        rmse=0.2,
    )

    first = summarize_supplementary_results(
        tmp_path, tmp_path / "summary-a", bootstrap_samples=100, bootstrap_seed=7
    )
    second = summarize_supplementary_results(
        tmp_path, tmp_path / "summary-b", bootstrap_samples=100, bootstrap_seed=7
    )

    target_aware = first.summary[
        (first.summary["section"] == "backbone")
        & (first.summary["variant"] == "target_aware_bilstm")
        & (first.summary["metric"] == "rmse")
    ].iloc[0]
    repeated = second.summary[
        (second.summary["section"] == "backbone")
        & (second.summary["variant"] == "target_aware_bilstm")
        & (second.summary["metric"] == "rmse")
    ].iloc[0]
    assert target_aware["run_count"] == 3
    assert target_aware["ci95_low"] == repeated["ci95_low"]
    assert target_aware["ci95_high"] == repeated["ci95_high"]
    single = first.summary[
        (first.summary["section"] == "calibration")
        & (first.summary["metric"] == "rmse")
    ].iloc[0]
    assert pd.isna(single["ci95_low"])
    assert pd.isna(single["ci95_high"])


def test_supplementary_summary_pairs_only_matching_seeds_and_metric_space(
    tmp_path: Path,
) -> None:
    for seed, baseline, candidate in ((0, 0.3, 0.2), (1, 0.5, 0.4)):
        _write_metrics(
            tmp_path / f"backbone/field/bilstm/N100/seed{seed}/metrics.csv",
            rmse=baseline,
        )
        _write_metrics(
            tmp_path
            / f"backbone/field/target_aware_bilstm/N100/seed{seed}/metrics.csv",
            rmse=candidate,
        )
    _write_metrics(
        tmp_path / "backbone/field/target_aware_bilstm/N100/seed2/metrics.csv",
        rmse=0.1,
    )

    result = summarize_supplementary_results(
        tmp_path, tmp_path / "summary", bootstrap_samples=50
    )
    comparison = result.paired[
        (result.paired["section"] == "backbone")
        & (result.paired["metric"] == "rmse")
    ].iloc[0]

    assert comparison["pair_count"] == 2
    assert comparison["win_count"] == 2
    assert comparison["improvement_fraction"] > 0
    for filename in (
        "supplementary_metrics_long.csv",
        "supplementary_metrics_summary.csv",
        "paired_comparisons.csv",
        "budget_validation.csv",
        "missing_experiments.csv",
    ):
        assert (tmp_path / "summary" / filename).exists()
