import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import tools.run_supplementary_experiments as supplementary_runner
from tools.run_supplementary_experiments import (
    SupplementaryStep,
    build_supplementary_plan,
    parse_args,
    run_plan,
)


def test_active_plan_uses_equal_final_budgets(tmp_path: Path) -> None:
    plan = build_supplementary_plan(
        python="python",
        output_root=tmp_path,
        sections=("active",),
        datasets=("field",),
        seeds=(0,),
        overrides=(),
    )

    assert len(plan) == 15
    m5 = next(step for step in plan if "/M5/N200/" in step.artifact.as_posix())
    m9 = next(step for step in plan if "/M9/N200/" in step.artifact.as_posix())
    assert "data.initial_labels=200" in m5.command
    assert "data.initial_labels=100" in m9.command
    assert "active_learning.rounds=5" in m9.command
    assert "active_learning.batch_budget=20" in m9.command


def test_other_sections_have_complete_factorial_plans(tmp_path: Path) -> None:
    kwargs = {
        "python": "python",
        "output_root": tmp_path,
        "datasets": ("field", "spwla", "forward"),
        "seeds": (2,),
        "overrides": (),
    }
    calibration = build_supplementary_plan(
        sections=("calibration",), **kwargs
    )
    backbone = build_supplementary_plan(sections=("backbone",), **kwargs)
    transfer = build_supplementary_plan(sections=("transfer",), **kwargs)

    assert len(calibration) == 12
    assert len(backbone) == 12
    assert len(transfer) == 6
    assert {
        value
        for step in calibration
        for value in step.command
        if value.startswith("uncertainty.calibration_objective=")
    } == {
        "uncertainty.calibration_objective=nll",
        "uncertainty.calibration_objective=interval_score",
    }
    assert {
        value
        for step in backbone
        for value in step.command
        if value.startswith("model.point_architecture=")
    } == {
        "model.point_architecture=bilstm",
        "model.point_architecture=target_aware_bilstm",
    }
    assert all("seed=2" in step.command for step in transfer)
    assert {
        step.command[step.command.index("--target-dataset") + 1]
        for step in transfer
    } == {"field", "spwla", "forward"}


def test_run_plan_skips_existing_artifact(tmp_path: Path, monkeypatch) -> None:
    artifact = tmp_path / "metrics.csv"
    artifact.write_text("target,rmse\nPHIF,0.1\n", encoding="utf-8")
    (tmp_path / "summary.json").write_text('{"method": "M5"}', encoding="utf-8")
    called: list[object] = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: called.append(args))

    result = run_plan(
        [SupplementaryStep("active", "existing", ("python",), artifact)],
        dry_run=False,
    )

    assert result == 0
    assert called == []


def test_run_plan_does_not_skip_partial_metrics(tmp_path: Path, monkeypatch) -> None:
    artifact = tmp_path / "metrics.csv"
    artifact.write_text("target,rmse\nPHIF,0.1\n", encoding="utf-8")
    called: list[object] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: called.append(args)
        or SimpleNamespace(returncode=0),
    )

    assert run_plan(
        [
            SupplementaryStep(
                "active",
                "partial",
                ("python", "-m", "bnn_inversion.cli", "train"),
                artifact,
            )
        ],
        dry_run=False,
    ) == 0
    assert len(called) == 1


def test_run_plan_dry_run_does_not_start_subprocess(tmp_path: Path, monkeypatch) -> None:
    called: list[object] = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: called.append(args))

    result = run_plan(
        [
            SupplementaryStep(
                "active", "preview", ("python", "train"), tmp_path / "metrics.csv"
            )
        ],
        dry_run=True,
    )

    assert result == 0
    assert called == []


def test_run_plan_propagates_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=7),
    )

    with pytest.raises(subprocess.CalledProcessError) as error:
        run_plan(
            [
                SupplementaryStep(
                    "active", "broken", ("python",), tmp_path / "metrics.csv"
                )
            ],
            dry_run=False,
        )
    assert error.value.returncode == 7


def test_run_plan_accepts_native_exit_failure_after_complete_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = tmp_path / "metrics.csv"

    def finish_then_fail(*args, **kwargs):
        artifact.write_text("target,rmse\nPHIF,0.1\n", encoding="utf-8")
        (tmp_path / "summary.json").write_text(
            '{"method": "M5"}', encoding="utf-8"
        )
        return SimpleNamespace(returncode=3221226505)

    monkeypatch.setattr(
        subprocess,
        "run",
        finish_then_fail,
    )

    result = run_plan(
        [
            SupplementaryStep(
                "active",
                "native-exit-after-success",
                ("python", "-m", "bnn_inversion.cli", "train"),
                artifact,
            )
        ],
        dry_run=False,
    )

    assert result == 0


def test_parse_args_validates_comma_separated_choices() -> None:
    args = parse_args(
        [
            "--sections",
            "active,summary",
            "--datasets",
            "field,spwla",
            "--seeds",
            "0,4",
        ]
    )
    assert args.sections == ("active", "summary")
    assert args.datasets == ("field", "spwla")
    assert args.seeds == (0, 4)

    with pytest.raises(SystemExit):
        parse_args(["--sections", "unknown"])


def test_summary_section_runs_in_process_and_dry_run_only_previews(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    calls: list[tuple[object, object]] = []
    monkeypatch.setattr(
        supplementary_runner,
        "summarize_supplementary_results",
        lambda root, output_dir, **kwargs: calls.append((root, output_dir)),
    )

    assert supplementary_runner.main(
        ["--sections", "summary", "--output-root", str(tmp_path), "--dry-run"]
    ) == 0
    assert calls == []
    assert str(tmp_path / "summary") in capsys.readouterr().out

    assert supplementary_runner.main(
        ["--sections", "summary", "--output-root", str(tmp_path)]
    ) == 0
    assert calls == [(tmp_path, tmp_path / "summary")]
