from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from bnn_inversion.results import summarize_supplementary_results


DATASET_CONFIGS = {
    "field": "configs/field_main.yaml",
    "spwla": "configs/spwla.yaml",
    "forward": "configs/forward.yaml",
}
ACTIVE_BUDGETS = {125: (5, 5), 200: (5, 20), 500: (5, 80)}
TRANSFER_PROTOCOLS = {
    "spwla_holdout": ("spwla", "configs/transfer.yaml", "spwla"),
    "field_internal": ("field", "configs/field_transfer.yaml", "field"),
    "field_to_forward": ("forward", "configs/forward_transfer.yaml", "forward"),
}
SECTION_CHOICES = ("active", "uncertainty", "calibration", "backbone", "transfer", "summary")
UNCERTAINTY_BUDGETS = (50, 100, 200, 500)


@dataclass(frozen=True)
class SupplementaryStep:
    section: str
    identity: str
    command: tuple[str, ...]
    artifact: Path


def _command(
    python: str,
    action: str,
    config: str,
    output: Path,
    overrides: Sequence[str],
) -> tuple[str, ...]:
    command = [
        python,
        "-m",
        "bnn_inversion.cli",
        action,
        "--config",
        config,
        "--override",
        f"runtime.output_dir={output.as_posix()}",
    ]
    for override in overrides:
        command.extend(("--override", override))
    return tuple(command)


def _evaluate_command(
    python: str,
    config: str,
    target_dataset: str,
    output: Path,
    overrides: Sequence[str],
) -> tuple[str, ...]:
    command = list(_command(python, "evaluate", config, output, ()))
    command.extend(("--target-dataset", target_dataset))
    for override in overrides:
        command.extend(("--override", override))
    return tuple(command)


def build_supplementary_plan(
    *,
    python: str,
    output_root: Path,
    sections: Sequence[str],
    datasets: Sequence[str],
    seeds: Sequence[int],
    overrides: Sequence[str],
) -> list[SupplementaryStep]:
    plan: list[SupplementaryStep] = []
    if "uncertainty" in sections:
        for dataset in datasets:
            config = DATASET_CONFIGS[dataset]
            for method in ("M2", "M3", "M4", "M5"):
                for budget in UNCERTAINTY_BUDGETS:
                    for seed in seeds:
                        output = output_root / "uncertainty" / dataset / method / f"N{budget}" / f"seed{seed}"
                        step_overrides = (f"seed={seed}", f"method={method}", f"data.initial_labels={budget}", *overrides)
                        plan.append(SupplementaryStep(
                            section="uncertainty",
                            identity=f"uncertainty:{dataset}:{method}:N{budget}:seed{seed}",
                            command=_command(python, "train", config, output, step_overrides),
                            artifact=output / "metrics.csv",
                        ))
    if "active" in sections:
        for dataset in datasets:
            config = DATASET_CONFIGS[dataset]
            for method in ("M5", "M6", "M7", "M8", "M9"):
                for final_budget, (rounds, batch_budget) in ACTIVE_BUDGETS.items():
                    for seed in seeds:
                        output = (
                            output_root
                            / "active"
                            / dataset
                            / method
                            / f"N{final_budget}"
                            / f"seed{seed}"
                        )
                        step_overrides = [f"seed={seed}", f"method={method}"]
                        action = "train"
                        if method == "M5":
                            step_overrides.append(f"data.initial_labels={final_budget}")
                        else:
                            action = "active-learn"
                            step_overrides.extend(
                                (
                                    "data.initial_labels=100",
                                    f"active_learning.rounds={rounds}",
                                    f"active_learning.batch_budget={batch_budget}",
                                )
                            )
                        step_overrides.extend(overrides)
                        plan.append(
                            SupplementaryStep(
                                section="active",
                                identity=(
                                    f"active:{dataset}:{method}:N{final_budget}:seed{seed}"
                                ),
                                command=_command(
                                    python, action, config, output, step_overrides
                                ),
                                artifact=output / "metrics.csv",
                            )
                        )

    if "calibration" in sections:
        for dataset in datasets:
            config = DATASET_CONFIGS[dataset]
            for objective in ("nll", "interval_score"):
                for method in ("M4", "M5"):
                    for seed in seeds:
                        output = (
                            output_root
                            / "calibration"
                            / dataset
                            / objective
                            / method
                            / "N100"
                            / f"seed{seed}"
                        )
                        step_overrides = (
                            f"seed={seed}",
                            f"method={method}",
                            "data.initial_labels=100",
                            f"uncertainty.calibration_objective={objective}",
                            *overrides,
                        )
                        plan.append(
                            SupplementaryStep(
                                section="calibration",
                                identity=(
                                    f"calibration:{dataset}:{objective}:{method}:seed{seed}"
                                ),
                                command=_command(
                                    python, "train", config, output, step_overrides
                                ),
                                artifact=output / "metrics.csv",
                            )
                        )

    if "backbone" in sections:
        for dataset in datasets:
            config = DATASET_CONFIGS[dataset]
            for architecture in ("bilstm", "target_aware_bilstm"):
                for budget in (100, 500):
                    for seed in seeds:
                        output = (
                            output_root
                            / "backbone"
                            / dataset
                            / architecture
                            / f"N{budget}"
                            / f"seed{seed}"
                        )
                        step_overrides = (
                            f"seed={seed}",
                            "method=M5",
                            f"data.initial_labels={budget}",
                            f"model.point_architecture={architecture}",
                            *overrides,
                        )
                        plan.append(
                            SupplementaryStep(
                                section="backbone",
                                identity=(
                                    f"backbone:{dataset}:{architecture}:N{budget}:seed{seed}"
                                ),
                                command=_command(
                                    python, "train", config, output, step_overrides
                                ),
                                artifact=output / "metrics.csv",
                            )
                        )

    if "transfer" in sections:
        selected = set(datasets)
        for protocol, (dataset, config, target) in TRANSFER_PROTOCOLS.items():
            if dataset not in selected:
                continue
            for method in ("M5", "M9"):
                for seed in seeds:
                    output = (
                        output_root
                        / "transfer"
                        / protocol
                        / method
                        / f"seed{seed}"
                    )
                    step_overrides = [f"seed={seed}", f"method={method}"]
                    if method == "M9":
                        step_overrides.append("data.initial_labels=100")
                    step_overrides.extend(overrides)
                    plan.append(
                        SupplementaryStep(
                            section="transfer",
                            identity=f"transfer:{protocol}:{method}:seed{seed}",
                            command=_evaluate_command(
                                python,
                                config,
                                target,
                                output,
                                step_overrides,
                            ),
                            artifact=output / "metrics.csv",
                        )
                    )
    return plan


def _csv_strings(value: str, allowed: Sequence[str], label: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(items) - set(allowed))
    if not items or invalid:
        raise argparse.ArgumentTypeError(
            f"invalid {label}: {','.join(invalid) if invalid else value}"
        )
    return items


def _csv_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be non-negative integers")
    return seeds


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run equal-budget and ablation experiments for BNN inversion."
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/supplementary")
    )
    parser.add_argument(
        "--sections",
        type=lambda value: _csv_strings(value, SECTION_CHOICES, "section"),
        default=SECTION_CHOICES,
    )
    parser.add_argument(
        "--datasets",
        type=lambda value: _csv_strings(value, tuple(DATASET_CONFIGS), "dataset"),
        default=tuple(DATASET_CONFIGS),
    )
    parser.add_argument("--seeds", type=_csv_seeds, default=(0, 1, 2, 3, 4))
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _artifacts_complete(step: SupplementaryStep) -> bool:
    if not step.artifact.is_file() or step.artifact.stat().st_size == 0:
        return False
    try:
        with step.artifact.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"target", "rmse"}.issubset(reader.fieldnames or ()):
                return False
            if next(reader, None) is None:
                return False
        action = ""
        if "bnn_inversion.cli" in step.command:
            action_index = step.command.index("bnn_inversion.cli") + 1
            if action_index < len(step.command):
                action = step.command[action_index]
        marker_name = "domain_summary.json" if action == "evaluate" else "summary.json"
        marker = step.artifact.with_name(marker_name)
        if not marker.is_file() or marker.stat().st_size == 0:
            return False
        with marker.open(encoding="utf-8") as handle:
            return isinstance(json.load(handle), dict)
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError):
        return False


def run_plan(plan: Sequence[SupplementaryStep], *, dry_run: bool) -> int:
    for index, step in enumerate(plan, start=1):
        print(f"[{index}/{len(plan)}] {step.identity}")
        print(f"  {subprocess.list2cmdline(step.command)}")
        if _artifacts_complete(step):
            print(f"  skip complete artifacts: {step.artifact.parent}")
            continue
        if dry_run:
            continue
        completed = subprocess.run(step.command, check=False)
        if completed.returncode != 0:
            if _artifacts_complete(step):
                print(
                    f"  process exited with {completed.returncode} after complete "
                    f"artifacts: {step.artifact.parent}"
                )
                continue
            raise subprocess.CalledProcessError(completed.returncode, step.command)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    training_sections = tuple(
        section for section in args.sections if section != "summary"
    )
    plan = build_supplementary_plan(
        python=args.python,
        output_root=args.output_root,
        sections=training_sections,
        datasets=args.datasets,
        seeds=args.seeds,
        overrides=args.override,
    )
    result = run_plan(plan, dry_run=args.dry_run)
    if "summary" in args.sections:
        summary_dir = args.output_root / "summary"
        if args.dry_run:
            print(f"[summary] {summary_dir}")
        else:
            expected_sections = training_sections or SECTION_CHOICES[:-1]
            expected_plan = build_supplementary_plan(
                python=args.python,
                output_root=args.output_root,
                sections=expected_sections,
                datasets=args.datasets,
                seeds=args.seeds,
                overrides=args.override,
            )
            summarize_supplementary_results(
                args.output_root,
                summary_dir,
                expected_artifacts=[step.artifact for step in expected_plan],
            )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
