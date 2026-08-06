from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class PlanStep:
    name: str
    command: list[str]


def _cli(python: str, command: str, config: str, output: Path, overrides: Sequence[str]) -> list[str]:
    result = [
        python,
        "-m",
        "bnn_inversion.cli",
        command,
        "--config",
        config,
        "--override",
        f"runtime.output_dir={output.as_posix()}",
    ]
    for item in overrides:
        result.extend(["--override", item])
    return result


def _with_overrides(command: list[str], overrides: Sequence[str]) -> list[str]:
    result = list(command)
    for item in overrides:
        result.extend(["--override", item])
    return result


def _output_dir_from_command(command: Sequence[str]) -> Path | None:
    for index, item in enumerate(command):
        if item == "--override" and index + 1 < len(command):
            value = command[index + 1]
            if value.startswith("runtime.output_dir="):
                return Path(value.split("=", 1)[1])
    return None


def _completion_artifact(step: PlanStep) -> Path | None:
    output = _output_dir_from_command(step.command)
    if output is None:
        return None
    if step.name == "optimize-m5":
        return output / "optimization_results.csv"
    if step.name == "main-matrix":
        return None
    return output / "metrics.csv"


def build_plan(
    *,
    python: str,
    output_root: Path,
    methods: str,
    seeds: str,
    optimization_seeds: str,
    optimization_trials: int | None,
    overrides: Sequence[str],
    skip_audit: bool = False,
    skip_export: bool = False,
    skip_optimization: bool = False,
    skip_matrix: bool = False,
    skip_transfer: bool = False,
) -> list[PlanStep]:
    plan: list[PlanStep] = []
    audit_configs = [
        ("audit-field", "configs/field_main.yaml", output_root / "audit" / "field.json"),
        ("audit-spwla", "configs/transfer.yaml", output_root / "audit" / "spwla.json"),
        ("audit-forward", "configs/forward.yaml", output_root / "audit" / "forward.json"),
    ]
    if not skip_audit:
        for name, config, output in audit_configs:
            command = [
                python,
                "-m",
                "bnn_inversion.cli",
                "audit",
                "--config",
                config,
                "--output",
                output.as_posix(),
            ]
            plan.append(PlanStep(name, _with_overrides(command, overrides)))
    if not skip_export:
        command = [
            python,
            "-m",
            "bnn_inversion.cli",
            "export-cleaned",
            "--config",
            "configs/field_main.yaml",
            "--dataset",
            "all",
        ]
        plan.append(PlanStep("export-cleaned", _with_overrides(command, overrides)))
    if not skip_optimization:
        command = _cli(
            python,
            "optimize",
            "configs/field_main.yaml",
            output_root / "optimization",
            overrides,
        )
        if optimization_trials is not None:
            command.extend(["--max-trials", str(optimization_trials)])
        if optimization_seeds:
            command.extend(["--seeds", optimization_seeds])
        plan.append(PlanStep("optimize-m5", command))
    if not skip_matrix:
        command = _cli(
            python,
            "run-matrix",
            "configs/field_main.yaml",
            output_root / "matrix",
            overrides,
        )
        command.extend(["--methods", methods, "--seeds", seeds])
        plan.append(PlanStep("main-matrix", command))
    if not skip_transfer:
        transfers = [
            (
                "spwla",
                "configs/transfer.yaml",
                "spwla",
                output_root / "transfer" / "spwla",
            ),
            (
                "field",
                "configs/field_transfer.yaml",
                "field",
                output_root / "transfer" / "field",
            ),
            (
                "forward",
                "configs/forward_transfer.yaml",
                "forward",
                output_root / "transfer" / "forward",
            ),
        ]
        for label, config, target, output in transfers:
            for method in ("M5", "M9"):
                method_output = output / method
                method_overrides = list(overrides)
                if method == "M9":
                    method_overrides.extend(["method=M9", "data.initial_labels=100"])
                command = _cli(python, "evaluate", config, method_output, [])
                command.extend(["--target-dataset", target])
                command = _with_overrides(command, method_overrides)
                plan.append(PlanStep(f"transfer-{label}-{method}", command))
    return plan


def _fast_overrides(existing: Sequence[str]) -> list[str]:
    values = list(existing)
    values.extend(
        [
            "training.epochs=2",
            "training.early_stopping_patience=1",
            "training.batch_size=16",
            "uncertainty.mc_samples=3",
            "uncertainty.bnn_samples=3",
            "data.validation_size=20",
            "active_learning.rounds=1",
            "active_learning.batch_budget=2",
            "runtime.device=cpu",
        ]
    )
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete BNN inversion experiment suite."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/all_experiments"))
    parser.add_argument("--methods", default="M1,M2,M3,M4,M5,M6,M7,M8,M9")
    parser.add_argument("--seeds", default="1,2,3,4")
    parser.add_argument("--optimization-seeds", default="1,2,3,4")
    parser.add_argument("--optimization-trials", type=int, default=12)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra override appended to every experiment command.",
    )
    parser.add_argument("--fast", action="store_true", help="Run a short CPU smoke suite.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-optimization", action="store_true")
    parser.add_argument("--skip-matrix", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    methods = "M1,M5,M9" if args.fast else args.methods
    seeds = "1" if args.fast else args.seeds
    optimization_trials = 2 if args.fast else args.optimization_trials
    optimization_seeds = "1" if args.fast else args.optimization_seeds
    overrides = _fast_overrides(args.override) if args.fast else args.override
    plan = build_plan(
        python=args.python,
        output_root=args.output_root,
        methods=methods,
        seeds=seeds,
        optimization_seeds=optimization_seeds,
        optimization_trials=optimization_trials,
        overrides=overrides,
        skip_audit=args.skip_audit,
        skip_export=args.skip_export,
        skip_optimization=args.skip_optimization,
        skip_matrix=args.skip_matrix,
        skip_transfer=args.skip_transfer,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    for index, step in enumerate(plan, start=1):
        rendered = subprocess.list2cmdline(step.command)
        print(f"[{index}/{len(plan)}] {step.name}")
        print(f"  {rendered}")
        artifact = _completion_artifact(step)
        if artifact is not None and artifact.exists():
            print(f"  skip existing artifact: {artifact}")
            continue
        if not args.dry_run:
            completed = subprocess.run(step.command, check=False)
            if completed.returncode != 0:
                if artifact is not None and artifact.exists():
                    print(
                        f"  command returned {completed.returncode}, "
                        f"but artifact exists: {artifact}"
                    )
                    continue
                raise subprocess.CalledProcessError(
                    completed.returncode,
                    step.command,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
