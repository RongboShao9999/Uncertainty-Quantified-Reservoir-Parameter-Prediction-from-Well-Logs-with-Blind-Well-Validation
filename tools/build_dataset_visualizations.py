from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from bnn_inversion.data.adapters import MAIN_7, SPWLA_6, load_dataset
from bnn_inversion.publication_style import FIGURE_WIDTH_IN, PALETTE, apply_sci_style, audit_layout


SEED = 20260707
DATASETS = ("field", "spwla", "forward")
TARGET_COLORS = {"PHIF": PALETTE["primary"], "SW": PALETTE["negative"], "PERM": PALETTE["text"], "VSH": PALETTE["text"]}
LINE_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1))]
FEATURE_COLORS = [PALETTE["primary"], PALETTE["negative"], PALETTE["text"], PALETTE["secondary"], PALETTE["trusted"], PALETTE["light"], "#75685C"]
CORR_CMAP = LinearSegmentedColormap.from_list("corr", ["#4D6F93", "#F7F7F5", "#B85C4A"])


def select_representative_group(frame: pd.DataFrame, group_column: str) -> tuple[object, str, int]:
    counts = frame.groupby(group_column, sort=False).size().rename("length").reset_index()
    counts["stable"] = counts[group_column].astype(str)
    counts = counts.sort_values("stable", kind="stable").reset_index(drop=True)
    counts["anonymous_id"] = [f"Group-{index + 1:03d}" for index in range(len(counts))]
    median = float(counts["length"].median())
    row = counts.assign(distance=(counts.length - median).abs()).sort_values(["distance", "stable"], kind="stable").iloc[0]
    return row[group_column], str(row.anonymous_id), int(row.length)


def robust_zscore(frame: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> pd.DataFrame:
    result = frame.loc[:, list(columns)].astype(float).copy()
    median = result.median()
    iqr = result.quantile(0.75) - result.quantile(0.25)
    iqr = iqr.mask(iqr.abs() < 1e-12, 1.0)
    return (result - median) / iqr


def fixed_sample(frame: pd.DataFrame, max_rows: int, *, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy().reset_index(drop=True)
    return frame.sample(n=max_rows, random_state=seed).sort_index().reset_index(drop=True)


def ordered_track_sample(frame: pd.DataFrame, depth_column: str, max_rows: int) -> pd.DataFrame:
    ordered = frame.sort_values(depth_column, kind="stable").reset_index(drop=True)
    if len(ordered) <= max_rows:
        return ordered
    positions = np.linspace(0, len(ordered) - 1, max_rows, dtype=int)
    return ordered.iloc[positions].reset_index(drop=True)


def stratified_sample(frame: pd.DataFrame, group_column: str, max_rows: int, *, seed: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    fractions = frame.groupby(group_column).size() / len(frame) * max_rows
    allocation = np.floor(fractions).astype(int).clip(lower=1)
    while allocation.sum() > max_rows:
        key = allocation.idxmax(); allocation.loc[key] -= 1
    remainder = max_rows - int(allocation.sum())
    for key in (fractions - np.floor(fractions)).sort_values(ascending=False).index[:remainder]:
        allocation.loc[key] += 1
    parts = []
    for key, count in allocation.items():
        group = frame[frame[group_column] == key]
        take = min(int(count), len(group))
        parts.append(group.sample(n=take, random_state=int(rng.integers(0, 2**31 - 1))))
    return pd.concat(parts).sort_index().reset_index(drop=True)


def _raw_frame(root: Path, dataset: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    if dataset == "field":
        features, targets = list(MAIN_7), ["PHIF", "SW", "PERM"]
        frame = pd.read_csv(root / "区块4(筛选).csv", encoding="utf-8", encoding_errors="replace").rename(columns={"POR": "PHIF"})
    elif dataset == "spwla":
        features, targets = list(SPWLA_6), ["PHIF", "SW", "VSH"]
        frame = pd.read_csv(root / "train.csv", encoding="utf-8", encoding_errors="replace", na_values=[-9999, -9999.0])
    else:
        features, targets = list(MAIN_7), ["PHIF", "SW", "VSH"]
        frames = [pd.read_csv(path, encoding="utf-8", encoding_errors="replace") for path in sorted((root / "forward_dataset").glob("*.csv"))]
        frame = pd.concat(frames, ignore_index=True).rename(columns={"#DEPTH": "DEPTH", "POR": "PHIF", "Sw": "SW", "Vsh": "VSH"})
    return frame, features, targets


def _panel_label(fig, ax, label: str) -> None:
    ax.text(-0.13, 1.02, label, transform=ax.transAxes, fontsize=9, fontweight="bold", fontfamily="Times New Roman", ha="left", va="bottom", clip_on=False)


def missingness_axis_limit(values: pd.Series) -> tuple[float, float]:
    maximum = float(pd.to_numeric(values, errors="coerce").fillna(0).max())
    return 0.0, max(5.0, maximum * 1.08)


def crossplot_specs(dataset: str) -> list[tuple[str, str, str, str]]:
    common = [("DEN", "PHIF", "DEN", "PHIF")]
    if dataset == "field":
        return common + [("RT", "SW", "log10(RT)", "SW")]
    if dataset == "spwla":
        return common + [("GR", "VSH", "GR", "VSH"), ("RDEP", "SW", "log10(RDEP)", "SW")]
    if dataset == "forward":
        return common + [("GR", "VSH", "GR", "VSH"), ("RT", "SW", "log10(RT)", "SW")]
    raise ValueError(f"unknown dataset: {dataset}")


def individual_target_specs() -> dict[str, list[str]]:
    return {
        "field": ["PHIF", "SW", "PERM"],
        "spwla": ["PHIF", "SW", "VSH"],
        "forward": ["PHIF", "SW", "VSH"],
    }


def _export_snapshot(root: Path, dataset: str, output: Path) -> dict[str, object]:
    canonical = load_dataset(dataset, root, feature_profile="main_7")
    frame = canonical.frame.copy()
    representative, anonymous_id, representative_length = select_representative_group(frame, canonical.well_column)
    track = ordered_track_sample(frame[frame[canonical.well_column] == representative], canonical.depth_column, 600)
    sample = stratified_sample(frame, canonical.well_column, 50000, seed=SEED)
    raw, raw_features, raw_targets = _raw_frame(root, dataset)
    relevant = [column for column in [*raw_features, *raw_targets] if column in raw]
    missing = raw[relevant].isna().mean().mul(100).rename("missing_rate_pct").reset_index(name="missing_rate_pct").rename(columns={"index": "variable"})
    missing["variable_type"] = np.where(missing.variable.isin(raw_features), "feature", "target")

    export_track = track[[canonical.depth_column, *canonical.feature_columns, *canonical.target_columns]].copy()
    export_track.insert(0, "anonymous_group", anonymous_id)
    export_sample = sample[[*canonical.feature_columns, *canonical.target_columns]].copy()
    crossplot_sample = fixed_sample(export_sample, 20000, seed=SEED)
    correlation = export_sample[[*canonical.feature_columns, *canonical.target_columns]].corr(method="spearman").loc[list(canonical.feature_columns), list(canonical.target_columns)]

    data_dir = output / "data"; data_dir.mkdir(parents=True, exist_ok=True)
    export_sample.to_csv(data_dir / f"{dataset}_distribution_sample.csv", index=False, encoding="utf-8-sig")
    crossplot_sample.to_csv(data_dir / f"{dataset}_crossplot_sample.csv", index=False, encoding="utf-8-sig")
    correlation.to_csv(data_dir / f"{dataset}_feature_target_spearman.csv", encoding="utf-8-sig")
    missing.to_csv(data_dir / f"{dataset}_missingness.csv", index=False, encoding="utf-8-sig")
    return {
        "dataset": dataset,
        "canonical": canonical,
        "track": export_track,
        "sample": export_sample,
        "crossplot": crossplot_sample,
        "correlation": correlation,
        "missing": missing,
        "anonymous_id": anonymous_id,
        "representative_length": representative_length,
        "raw_rows": len(raw),
        "clean_rows": len(frame),
        "sample_rows": len(export_sample),
        "group_count": frame[canonical.well_column].nunique(),
    }


def _render(snapshot: dict[str, object], output: Path) -> tuple[Path, Path, dict[str, object]]:
    apply_sci_style()
    canonical = snapshot["canonical"]; track = snapshot["track"]; sample = snapshot["sample"]
    features = list(canonical.feature_columns); targets = list(canonical.target_columns); depth = canonical.depth_column
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 220 / 25.4), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.0, 1.0])
    ax_track_x = fig.add_subplot(grid[0, 0]); ax_track_y = fig.add_subplot(grid[0, 1])
    ax_features = fig.add_subplot(grid[1, 0]); ax_targets = fig.add_subplot(grid[1, 1])
    ax_corr = fig.add_subplot(grid[2, 0]); ax_complete = fig.add_subplot(grid[2, 1])

    x = track[depth].to_numpy(); z = robust_zscore(track, features)
    for index, feature in enumerate(features):
        ax_track_x.plot(x, z[feature], color=FEATURE_COLORS[index % len(FEATURE_COLORS)], linestyle=LINE_STYLES[index % len(LINE_STYLES)], label=feature, linewidth=0.9)
    ax_track_x.set_xlabel("Depth / sequence index"); ax_track_x.set_ylabel("Robust z-score"); ax_track_x.legend(ncol=4, frameon=False, fontsize=6.8, loc="upper center"); ax_track_x.grid(axis="y", color="#D9D9D9", linewidth=.45)

    for target in targets:
        label = f"{target} (log10)" if target == "PERM" else target
        ax_track_y.plot(x, track[target], color=TARGET_COLORS[target], linestyle={"PHIF":"-", "SW":"--", "PERM":"-.", "VSH":"-."}[target], label=label, linewidth=1.0)
    ax_track_y.set_xlabel("Depth / sequence index"); ax_track_y.set_ylabel("Target value"); ax_track_y.legend(frameon=False, ncol=3, loc="upper center"); ax_track_y.grid(axis="y", color="#D9D9D9", linewidth=.45)

    distribution = robust_zscore(sample, features)
    bp = ax_features.boxplot([distribution[column].clip(-5, 5) for column in features], tick_labels=features, showfliers=False, patch_artist=True, medianprops={"color": PALETTE["negative"]})
    for box in bp["boxes"]: box.set(facecolor=PALETTE["secondary"], edgecolor=PALETTE["primary"], linewidth=.6)
    ax_features.tick_params(axis="x", rotation=35); ax_features.set_ylabel("Robust z-score"); ax_features.grid(axis="y", color="#D9D9D9", linewidth=.45)

    for target in targets:
        values = sample[target].dropna().to_numpy(); hist, edges = np.histogram(values, bins=35, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax_targets.plot(centers, hist, color=TARGET_COLORS[target], linestyle={"PHIF":"-", "SW":"--", "PERM":"-.", "VSH":"-."}[target], label=f"{target}{' (log10)' if target == 'PERM' else ''}")
    ax_targets.set_xlabel("Target value"); ax_targets.set_ylabel("Density"); ax_targets.legend(frameon=False); ax_targets.grid(axis="y", color="#D9D9D9", linewidth=.45)

    corr = snapshot["correlation"]
    image = ax_corr.imshow(corr.to_numpy(), cmap=CORR_CMAP, vmin=-1, vmax=1, aspect="auto")
    ax_corr.set_xticks(range(len(targets)), targets); ax_corr.set_yticks(range(len(features)), features)
    for i in range(len(features)):
        for j in range(len(targets)):
            ax_corr.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=6.8)
    fig.colorbar(image, ax=ax_corr, fraction=.05, pad=.03, label="Spearman ρ")

    missing = snapshot["missing"].sort_values("missing_rate_pct", ascending=True)
    ax_complete.barh(missing.variable, missing.missing_rate_pct, color=np.where(missing.variable_type.eq("target"), PALETTE["negative"], PALETTE["secondary"]), edgecolor=PALETTE["text"], linewidth=.35)
    ax_complete.set_xlabel("Raw-data missing rate (%)"); ax_complete.set_xlim(*missingness_axis_limit(missing.missing_rate_pct)); ax_complete.grid(axis="x", color="#D9D9D9", linewidth=.45)
    summary = f"Rows: {snapshot['raw_rows']:,} raw / {snapshot['clean_rows']:,} valid\nGroups: {snapshot['group_count']:,}\nDistribution sample: {snapshot['sample_rows']:,}\nRepresentative: {snapshot['anonymous_id']} ({snapshot['representative_length']:,} rows)"
    ax_complete.text(.98, .03, summary, transform=ax_complete.transAxes, ha="right", va="bottom", fontsize=7.5, bbox={"facecolor":"white", "edgecolor":PALETTE["light"], "linewidth":.5, "alpha":.92})

    for ax in [ax_track_x, ax_track_y, ax_features, ax_targets, ax_corr, ax_complete]:
        ax.spines[["top", "right"]].set_visible(False)
    _panel_label(fig, ax_track_x, "(a)"); _panel_label(fig, ax_features, "(b)"); _panel_label(fig, ax_targets, "(c)"); _panel_label(fig, ax_corr, "(d)"); _panel_label(fig, ax_complete, "(e)")
    fig.canvas.draw()
    audit = audit_layout(fig, str(snapshot["dataset"]))
    pdf = output / f"{snapshot['dataset']}_dataset.pdf"; svg = output / f"{snapshot['dataset']}_dataset.svg"
    fig.savefig(pdf, facecolor="white"); fig.savefig(svg, facecolor="white")
    preview = output.parents[1] / ".docx_work" / "dataset_previews"; preview.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview / f"{snapshot['dataset']}_dataset.png", dpi=140, facecolor="white")
    plt.close(fig)
    return pdf, svg, audit


def _draw_crossplot(ax, frame: pd.DataFrame, spec: tuple[str, str, str, str]) -> None:
    x_column, y_column, x_label, y_label = spec
    part = frame[[x_column, y_column]].replace([np.inf, -np.inf], np.nan).dropna()
    ax.scatter(part[x_column], part[y_column], s=4, alpha=0.12, color=PALETTE["primary"], edgecolors="none", rasterized=False)
    if len(part) >= 20:
        try:
            part = part.assign(bin=pd.qcut(part[x_column], 20, duplicates="drop"))
            trend = part.groupby("bin", observed=True)[[x_column, y_column]].median()
            ax.plot(trend[x_column], trend[y_column], color=PALETTE["negative"], linewidth=1.2)
        except ValueError:
            pass
    ax.set_xlabel(x_label); ax.set_ylabel(y_label)
    ax.grid(True, color="#D9D9D9", linewidth=.45, alpha=.65)


def _draw_target_density(ax, sample: pd.DataFrame, targets: list[str]) -> None:
    styles = {"PHIF": "-", "SW": "--", "PERM": "-.", "VSH": "-."}
    for target in targets:
        values = sample[target].dropna().to_numpy(); hist, edges = np.histogram(values, bins=35, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, hist, color=TARGET_COLORS[target], linestyle=styles[target], label=f"{target}{' (log10)' if target == 'PERM' else ''}")
    ax.set_xlabel("Target value"); ax.set_ylabel("Density"); ax.legend(frameon=False); ax.grid(axis="y", color="#D9D9D9", linewidth=.45)


def _draw_single_target_density(ax, sample: pd.DataFrame, target: str) -> None:
    values = sample[target].dropna().to_numpy()
    hist, edges = np.histogram(values, bins=35, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    color = TARGET_COLORS[target]
    ax.fill_between(centers, hist, color=color, alpha=0.18, linewidth=0)
    ax.plot(centers, hist, color=color, linewidth=1.4)
    ax.set_xlabel(f"{target} (log10)" if target == "PERM" else target)
    ax.set_ylabel("Density")
    ax.grid(axis="y", color="#D9D9D9", linewidth=.45)


def _draw_correlation(fig, ax, correlation: pd.DataFrame) -> None:
    image = ax.imshow(correlation.to_numpy(), cmap=CORR_CMAP, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(correlation.columns)), correlation.columns); ax.set_yticks(range(len(correlation.index)), correlation.index)
    for i in range(len(correlation.index)):
        for j in range(len(correlation.columns)):
            ax.text(j, i, f"{correlation.iloc[i,j]:.2f}", ha="center", va="center", fontsize=6.8)
    fig.colorbar(image, ax=ax, fraction=.045, pad=.025, label="Spearman ρ")


def _draw_completeness(ax, snapshot: dict[str, object]) -> None:
    missing = snapshot["missing"].sort_values("missing_rate_pct", ascending=True)
    ax.barh(missing.variable, missing.missing_rate_pct, color=np.where(missing.variable_type.eq("target"), PALETTE["negative"], PALETTE["secondary"]), edgecolor=PALETTE["text"], linewidth=.35)
    ax.set_xlabel("Raw-data missing rate (%)"); ax.set_xlim(*missingness_axis_limit(missing.missing_rate_pct)); ax.grid(axis="x", color="#D9D9D9", linewidth=.45)
    summary = f"Rows: {snapshot['raw_rows']:,} raw / {snapshot['clean_rows']:,} valid\nGroups: {snapshot['group_count']:,}\nDistribution sample: {snapshot['sample_rows']:,}\nCrossplot sample: {len(snapshot['crossplot']):,}"
    ax.text(.98, .03, summary, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, bbox={"facecolor":"white", "edgecolor":PALETTE["light"], "linewidth":.5, "alpha":.92})


def _render_compact(snapshot: dict[str, object], output: Path) -> tuple[Path, Path, dict[str, object]]:
    apply_sci_style()
    canonical = snapshot["canonical"]; dataset = str(snapshot["dataset"])
    targets = individual_target_specs()[dataset]; specs = crossplot_specs(dataset)
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 185 / 25.4), constrained_layout=True)
    grid = fig.add_gridspec(3, 3)
    axes: list[plt.Axes] = []
    for index, spec in enumerate(specs):
        ax = fig.add_subplot(grid[0, index]); _draw_crossplot(ax, snapshot["crossplot"], spec); axes.append(ax)
    if len(specs) == 2:
        target_axes = [fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
        axes.extend(target_axes)
        complete_ax = fig.add_subplot(grid[1, 2]); axes.append(complete_ax)
        corr_ax = fig.add_subplot(grid[2, :]); axes.append(corr_ax)
    else:
        target_axes = [fig.add_subplot(grid[1, index]) for index in range(3)]
        axes.extend(target_axes)
        corr_ax = fig.add_subplot(grid[2, 0:2]); axes.append(corr_ax)
        complete_ax = fig.add_subplot(grid[2, 2]); axes.append(complete_ax)
    for ax, target in zip(target_axes, targets):
        _draw_single_target_density(ax, snapshot["sample"], target)
    _draw_completeness(complete_ax, snapshot)
    _draw_correlation(fig, corr_ax, snapshot["correlation"])
    for index, ax in enumerate(axes):
        ax.spines[["top", "right"]].set_visible(False); _panel_label(fig, ax, f"({chr(97 + index)})")
    fig.canvas.draw(); audit = audit_layout(fig, dataset)
    pdf = output / f"{dataset}_dataset.pdf"; svg = output / f"{dataset}_dataset.svg"
    fig.savefig(pdf, facecolor="white"); fig.savefig(svg, facecolor="white")
    preview = output.parents[1] / ".docx_work" / "dataset_previews"; preview.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview / f"{dataset}_dataset.png", dpi=150, facecolor="white")
    plt.close(fig)
    return pdf, svg, audit


def _render_individual_target_distributions(
    snapshot: dict[str, object], output: Path
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    apply_sci_style()
    dataset = str(snapshot["dataset"])
    individual = output / "individual"
    individual.mkdir(parents=True, exist_ok=True)
    preview = output.parents[1] / ".docx_work" / "dataset_previews" / "individual"
    preview.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    audits: list[dict[str, object]] = []
    for target in individual_target_specs()[dataset]:
        fig, ax = plt.subplots(figsize=(FIGURE_WIDTH_IN, 85 / 25.4), constrained_layout=True)
        _draw_single_target_density(ax, snapshot["sample"], target)
        ax.spines[["top", "right"]].set_visible(False)
        figure_id = f"{dataset}_{target}_distribution"
        fig.canvas.draw()
        audits.append(audit_layout(fig, figure_id))
        pdf = individual / f"{figure_id}.pdf"
        svg = individual / f"{figure_id}.svg"
        fig.savefig(pdf, facecolor="white")
        fig.savefig(svg, facecolor="white")
        fig.savefig(preview / f"{figure_id}.png", dpi=150, facecolor="white")
        plt.close(fig)
        manifest.append({"dataset": dataset, "target": target, "pdf_path": str(pdf), "svg_path": str(svg)})
    return manifest, audits


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path("D:/coding/BNN/DATASET")); parser.add_argument("--output", type=Path, default=Path("figure/datasets")); args = parser.parse_args()
    root = args.root.resolve(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    audits=[]; layouts=[]; individual_manifest=[]; individual_layouts=[]
    for dataset in DATASETS:
        snapshot = _export_snapshot(root, dataset, output); pdf, svg, layout = _render_compact(snapshot, output)
        manifest_rows, target_layouts = _render_individual_target_distributions(snapshot, output)
        individual_manifest.extend(manifest_rows); individual_layouts.extend(target_layouts)
        audits.append({"dataset":dataset, "source_root":str(root), "raw_rows":snapshot["raw_rows"], "clean_rows":snapshot["clean_rows"], "sample_rows":snapshot["sample_rows"], "group_count":snapshot["group_count"], "anonymous_representative":snapshot["anonymous_id"], "representative_length":snapshot["representative_length"], "features":";".join(snapshot["canonical"].feature_columns), "targets":";".join(snapshot["canonical"].target_columns), "sample_seed":SEED, "pdf_path":str(pdf), "svg_path":str(svg)})
        layouts.append(layout)
    pd.DataFrame(audits).to_csv(output / "dataset_visualization_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(layouts).to_csv(output / "layout_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(individual_manifest).to_csv(output / "individual" / "individual_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(individual_layouts).to_csv(output / "individual" / "individual_layout_audit.csv", index=False, encoding="utf-8-sig")
    print("generated 3 comprehensive dataset figures and 9 individual target figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
