from pathlib import Path

from tools.run_all_experiments import build_plan


def test_build_plan_contains_full_experiment_steps() -> None:
    plan = build_plan(
        python="python",
        output_root=Path("outputs/all"),
        methods="M1,M5,M9",
        seeds="0",
        optimization_seeds="0",
        optimization_trials=2,
        overrides=[],
    )
    names = [step.name for step in plan]

    assert names == [
        "audit-field",
        "audit-spwla",
        "audit-forward",
        "export-cleaned",
        "optimize-m5",
        "main-matrix",
        "transfer-spwla-M5",
        "transfer-spwla-M9",
        "transfer-field-M5",
        "transfer-field-M9",
        "transfer-forward-M5",
        "transfer-forward-M9",
    ]
    assert plan[4].command[:4] == ["python", "-m", "bnn_inversion.cli", "optimize"]
    assert "--max-trials" in plan[4].command
    assert plan[5].command[-4:] == ["--methods", "M1,M5,M9", "--seeds", "0"]


def test_build_plan_can_skip_expensive_sections() -> None:
    plan = build_plan(
        python="python",
        output_root=Path("outputs/all"),
        methods="M1",
        seeds="0",
        optimization_seeds="",
        optimization_trials=None,
        overrides=["runtime.device=cpu"],
        skip_export=True,
        skip_optimization=True,
        skip_matrix=True,
    )
    names = [step.name for step in plan]

    assert "export-cleaned" not in names
    assert "optimize-m5" not in names
    assert "main-matrix" not in names
    assert names[-6:] == [
        "transfer-spwla-M5",
        "transfer-spwla-M9",
        "transfer-field-M5",
        "transfer-field-M9",
        "transfer-forward-M5",
        "transfer-forward-M9",
    ]
    for step in plan:
        assert "runtime.device=cpu" in step.command
