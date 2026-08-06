from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

from bnn_inversion.config import ExperimentConfig, load_config
from bnn_inversion.data.adapters import export_cleaned_dataset, load_dataset
from bnn_inversion.experiments import (
    METHOD_REGISTRY,
    run_cross_domain_experiment,
    run_experiment,
)
from bnn_inversion.optimization import optimize_config
from bnn_inversion.results import summarize_matrix_results


def parse_overrides(items: Sequence[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"override must use key=value syntax: {item}")
        dotted_key, raw_value = item.split("=", 1)
        keys = [part for part in dotted_key.split(".") if part]
        if not keys:
            raise ValueError(f"override has an empty key: {item}")
        value = yaml.safe_load(raw_value)
        cursor = result
        for key in keys[:-1]:
            existing = cursor.setdefault(key, {})
            if not isinstance(existing, dict):
                raise ValueError(f"override path conflicts at {key}: {item}")
            cursor = existing
        cursor[keys[-1]] = value
    return result


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable dotted configuration override.",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bnn-inversion")
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit", help="Audit schema, units, and ranges")
    _common_config(audit)
    audit.add_argument("--output", type=Path)
    export_cleaned = commands.add_parser(
        "export-cleaned", help="Export cleaned canonical CSV data under the data root"
    )
    _common_config(export_cleaned)
    export_cleaned.add_argument(
        "--dataset",
        choices=("field", "spwla", "forward", "all"),
        help="Dataset to export. Defaults to data.dataset from config.",
    )
    export_cleaned.add_argument("--folder-name", default="processed_cleaned")
    for name in ("train", "active-learn"):
        command = commands.add_parser(name)
        _common_config(command)
    evaluate = commands.add_parser("evaluate")
    _common_config(evaluate)
    evaluate.add_argument(
        "--target-dataset", choices=("spwla", "field", "forward"), required=True
    )
    matrix = commands.add_parser("run-matrix")
    _common_config(matrix)
    matrix.add_argument(
        "--methods", default="M1,M2,M3,M4,M5,M6,M7,M8,M9"
    )
    matrix.add_argument("--seeds", default="1,2,3,4")
    optimize = commands.add_parser("optimize")
    _common_config(optimize)
    optimize.add_argument("--max-trials", type=int)
    optimize.add_argument("--seeds", default="")
    summarize = commands.add_parser("summarize-matrix")
    summarize.add_argument("--root", type=Path, required=True)
    summarize.add_argument("--output", type=Path)
    summarize.add_argument("--exclude-seeds", default="")
    return parser


def _load(args: argparse.Namespace) -> ExperimentConfig:
    return load_config(args.config, parse_overrides(args.override))


def _audit(config: ExperimentConfig, output: Path | None) -> int:
    dataset = load_dataset(
        config.data.dataset,
        config.data.root,
        feature_profile=config.data.feature_profile,
    )
    payload = {
        "dataset": dataset.dataset,
        "rows": len(dataset.frame),
        "features": list(dataset.feature_columns),
        "targets": list(dataset.target_columns),
        "audit": [record.to_dict() for record in dataset.audit],
        "cleaning_audit": [record.to_dict() for record in dataset.cleaning_audit],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if output is None:
        print(rendered)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 0


def _export_cleaned(
    config: ExperimentConfig, dataset: str | None, folder_name: str
) -> int:
    datasets = (
        ("field", "spwla", "forward")
        if dataset == "all"
        else (dataset or config.data.dataset,)
    )
    for name in datasets:
        path = export_cleaned_dataset(
            name,
            config.data.root,
            feature_profile=config.data.feature_profile,
            folder_name=folder_name,
        )
        print(path)
    return 0


def _run_matrix(config: ExperimentConfig, methods: str, seeds: str) -> int:
    requested = [value.strip() for value in methods.split(",") if value.strip()]
    unknown = sorted(set(requested) - set(METHOD_REGISTRY))
    if unknown:
        raise ValueError(f"unknown matrix methods: {unknown}")
    seed_values = [int(value.strip()) for value in seeds.split(",") if value.strip()]
    if not seed_values:
        raise ValueError("matrix requires at least one seed")
    output_root = config.runtime.output_dir
    for method in requested:
        if method in {"M1", "M2", "M3", "M4", "M5", "M9"}:
            label_counts = (50, 100, 200, 500)
        else:
            label_counts = (100,)
        for labels in label_counts:
            for seed in seed_values:
                output = output_root / method / f"N{labels}" / f"seed{seed}"
                if (output / "metrics.csv").exists():
                    continue
                initial_labels = labels
                active_learning = config.active_learning
                if method in {"M6", "M7", "M8"}:
                    initial_labels = 100
                if method == "M9":
                    initial_labels = min(100, labels)
                    acquisition_budget = max(labels - initial_labels, 0)
                    batch_budget = 100 if acquisition_budget else 0
                    rounds = (
                        int(np.ceil(acquisition_budget / batch_budget))
                        if batch_budget
                        else 0
                    )
                    active_learning = replace(
                        config.active_learning,
                        rounds=rounds,
                        batch_budget=batch_budget,
                    )
                run_config = replace(
                    config,
                    seed=seed,
                    method=method,
                    data=replace(config.data, initial_labels=initial_labels),
                    active_learning=active_learning,
                    runtime=replace(config.runtime, output_dir=output),
                )
                run_experiment(run_config)
    return 0


def _seed_list(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "summarize-matrix":
        excluded = _seed_list(args.exclude_seeds)
        result = summarize_matrix_results(
            args.root,
            args.output or args.root,
            exclude_seeds=excluded,
        )
        print(f"long metrics: {result.output_dir / 'matrix_metrics_long.csv'}")
        print(f"summary: {result.output_dir / 'matrix_metrics_summary.csv'}")
        return 0
    config = _load(args)
    if args.command == "audit":
        return _audit(config, args.output)
    if args.command == "export-cleaned":
        return _export_cleaned(config, args.dataset, args.folder_name)
    if args.command == "active-learn" and config.method not in {"M6", "M7", "M8", "M9"}:
        raise ValueError("active-learn requires method M6, M7, M8, or M9")
    if args.command == "run-matrix":
        return _run_matrix(config, args.methods, args.seeds)
    if args.command == "optimize":
        seeds = (
            _seed_list(args.seeds)
            if args.seeds
            else None
        )
        result = optimize_config(
            config,
            config.runtime.output_dir,
            max_trials=args.max_trials,
            seeds=seeds,
        )
        print(result.results.head(10).to_string(index=False))
        print(f"best config: {result.output_dir / 'best_config.yaml'}")
        return 0
    if args.command == "evaluate":
        result = run_cross_domain_experiment(
            config, target_dataset=args.target_dataset
        )
    else:
        result = run_experiment(config)
    print(result.metrics.to_string(index=False))
    print(f"artifacts: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
