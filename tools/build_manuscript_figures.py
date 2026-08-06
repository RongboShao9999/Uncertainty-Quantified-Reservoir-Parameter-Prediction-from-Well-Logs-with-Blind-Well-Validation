from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


SEEDS = {0, 1, 2, 3, 4}
DATASET_NAMES = {"field": "油田", "spwla": "SPWLA", "forward": "正演"}
TARGET_ORDER = ["PHIF", "SW", "PERM", "VSH"]
METHOD_COLORS = {
    "M1": "#4C78A8", "M2": "#F58518", "M3": "#54A24B", "M4": "#E45756",
    "M5": "#7A5195", "M6": "#4C78A8", "M7": "#F58518", "M8": "#54A24B", "M9": "#E45756",
}


class FigureArtifact(NamedTuple):
    figure_id: str
    stem: str
    title: str
    anchor: str
    caption: str
    pdf_path: Path
    svg_path: Path


class ResultTables(NamedTuple):
    root: Path
    repo: Path
    metrics: pd.DataFrame
    paired: pd.DataFrame
    budgets: pd.DataFrame


FIGURE_SPECS = [
    ("1", "fig01_learning_curves", "三类数据域的M5学习曲线", "4.1", "三类数据域中静态M5随标注预算变化的RMSE（五个随机种子的均值±标准差）。"),
    ("2", "fig02_field_backbone", "油田数据的目标适配结构对比", "4.2", "油田数据上普通Bi-LSTM与Target-aware Bi-LSTM的RMSE对比；点为种子结果，柱为均值。"),
    ("3", "fig03_field_prediction_interval", "油田代表性结果的预测区间", "4.3", "油田M5、终态预算125的代表性中位种子预测结果；阴影为预测区间。"),
    ("4", "fig04_coverage_width", "覆盖率与区间宽度权衡", "4.3", "油田M5不同目标和预算的PICP—MPIW权衡（五种子均值）。"),
    ("5", "fig05_uncertainty_error", "预测误差与不确定性关系", "4.3", "油田代表性中位种子中总方差与绝对误差的关系；曲线为分箱中位数。"),
    ("6", "fig06_active_improvement", "主动学习等预算改善热力图", "4.4", "M6—M9相对静态M5的五种子配对RMSE改善率；正值表示改善，负值表示退化。"),
    ("7", "fig07_active_selection_process", "主动学习选样过程", "4.4", "终态预算125时不同主动策略的平均选样得分随轮次变化；阴影为五种子标准差。"),
    ("8", "fig08_domain_transfer", "数据域与迁移结果对比", "4.6", "Target-aware结构及M9迁移相对各自基线的五种子配对RMSE改善率。"),
    ("S1", "figS01_seed_stability", "五种子稳定性", "附录", "主要比较中各随机种子的配对RMSE改善率分布。"),
    ("S2", "figS02_method_ranking", "M1—M9任务内排名", "附录", "各数据域内按目标、预算和种子计算的RMSE任务内排名；数值越低越好。"),
    ("S3", "figS03_calibration_weights", "校准目标与融合权重", "附录", "NLL与区间得分校准选择的MC融合权重；相同结果表示当前候选网格上的边界重合。"),
    ("S4", "figS04_trust_status", "可信状态构成", "附录", "油田M5、终态预算125下可信与需复核样本比例（五种子均值）。"),
]


def expected_artifacts(output_dir: Path) -> list[FigureArtifact]:
    return [
        FigureArtifact(fid, stem, title, anchor, caption, output_dir / f"{stem}.pdf", output_dir / f"{stem}.svg")
        for fid, stem, title, anchor, caption in FIGURE_SPECS
    ]


def paired_rmse_improvement(frame: pd.DataFrame, *, baseline: str, candidate: str) -> pd.Series:
    base = pd.to_numeric(frame[baseline], errors="coerce")
    cand = pd.to_numeric(frame[candidate], errors="coerce")
    return (base - cand) / base * 100.0


def validate_seed_coverage(frame: pd.DataFrame, *, group_columns: Sequence[str]) -> None:
    missing: list[str] = []
    for keys, group in frame.groupby(list(group_columns), dropna=False):
        found = set(pd.to_numeric(group["seed"], errors="coerce").dropna().astype(int))
        if found != SEEDS:
            missing.append(f"{keys}: missing seeds {sorted(SEEDS - found)}")
    if missing:
        raise ValueError("; ".join(missing))


def select_median_seed(frame: pd.DataFrame, *, seed_column: str = "seed", score_column: str = "rmse") -> int:
    scores = frame.groupby(seed_column, as_index=False)[score_column].mean().sort_values([score_column, seed_column])
    if scores.empty:
        raise ValueError("cannot select a median seed from an empty frame")
    median = float(scores[score_column].median())
    scores["distance"] = (scores[score_column] - median).abs()
    return int(scores.sort_values(["distance", seed_column]).iloc[0][seed_column])


def load_results(results_dir: Path, repo: Path) -> ResultTables:
    metrics = pd.read_csv(results_dir / "supplementary_metrics_long.csv")
    paired = pd.read_csv(results_dir / "paired_comparisons.csv")
    budgets = pd.read_csv(results_dir / "budget_validation.csv")
    for frame in (metrics,):
        frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").astype("Int64")
        frame["budget_N"] = pd.to_numeric(frame["budget_N"], errors="coerce")
    paired["budget_N"] = pd.to_numeric(paired["budget_N"], errors="coerce")
    return ResultTables(results_dir, repo, metrics, paired, budgets)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def _save(fig: plt.Figure, artifact: FigureArtifact) -> None:
    fig.tight_layout()
    fig.savefig(artifact.pdf_path, bbox_inches="tight")
    fig.savefig(artifact.svg_path, bbox_inches="tight")
    preview_dir = artifact.pdf_path.parent / "_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview_dir / f"{artifact.stem}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def _ordered_targets(values: Sequence[str]) -> list[str]:
    present = set(values)
    return [target for target in TARGET_ORDER if target in present]


def _plot_learning_curves(results: ResultTables, artifact: FigureArtifact) -> None:
    frame = results.metrics.query("section == 'active' and method == 'M5'").copy()
    datasets = ["field", "spwla", "forward"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), sharex=True)
    for ax, dataset in zip(axes, datasets):
        part = frame[frame.dataset == dataset]
        for target in _ordered_targets(part.target.unique()):
            grouped = part[part.target == target].groupby("budget_N").rmse.agg(["mean", "std"]).reset_index()
            ax.errorbar(grouped.budget_N, grouped["mean"], yerr=grouped["std"], marker="o", capsize=3, label=target)
        ax.set_title(DATASET_NAMES[dataset])
        ax.set_xlabel("终态标注预算")
        ax.set_ylabel("RMSE")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("静态M5在三类数据域中的学习曲线", y=1.02)
    _save(fig, artifact)


def _plot_field_backbone(results: ResultTables, artifact: FigureArtifact) -> None:
    frame = results.metrics.query("section == 'backbone' and dataset == 'field'").copy()
    targets = _ordered_targets(frame.target.unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(11.5, 3.8))
    rng = np.random.default_rng(20260703)
    for ax, target in zip(np.atleast_1d(axes), targets):
        part = frame[frame.target == target]
        for xi, budget in enumerate(sorted(part.budget_N.unique())):
            for offset, variant, color in [(-0.17, "bilstm", "#9ECAE1"), (0.17, "target_aware_bilstm", "#3182BD")]:
                values = part[(part.budget_N == budget) & (part.variant == variant)].rmse.astype(float).to_numpy()
                ax.bar(xi + offset, values.mean(), width=0.3, color=color, alpha=0.85)
                ax.scatter(xi + offset + rng.normal(0, 0.018, len(values)), values, s=20, color="#252525", zorder=3)
        ax.set_xticks(range(len(sorted(part.budget_N.unique()))), [f"N={int(v)}" for v in sorted(part.budget_N.unique())])
        ax.set_title(target + ("（log10）" if target == "PERM" else ""))
        ax.set_ylabel("RMSE")
        ax.grid(axis="y", alpha=0.2)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#9ECAE1"), plt.Rectangle((0, 0), 1, 1, color="#3182BD")]
    fig.legend(handles, ["普通Bi-LSTM", "Target-aware"], loc="upper center", ncol=2, frameon=False)
    fig.suptitle("油田数据目标适配结构的五种子比较", y=1.05)
    _save(fig, artifact)


def _representative_prediction_path(results: ResultTables) -> tuple[Path, int]:
    frame = results.metrics.query("section == 'active' and dataset == 'field' and method == 'M5' and budget_N == 125")
    validate_seed_coverage(frame, group_columns=["dataset", "target", "budget_N", "method"])
    seed_scores = frame.groupby("seed", as_index=False).rmse.mean()
    seed = select_median_seed(seed_scores)
    row = frame[frame.seed == seed].iloc[0]
    return results.repo / Path(str(row.metrics_path)).parent / "predictions.csv", seed


def _load_predictions(path: Path) -> pd.DataFrame:
    usecols = ["source_index", "target", "y_true", "interval_mean", "lower", "upper", "total_variance", "y_true_log10", "interval_mean_log10", "lower_log10", "upper_log10"]
    return pd.read_csv(path, usecols=lambda col: col in usecols)


def _target_prediction_columns(frame: pd.DataFrame, target: str) -> tuple[str, str, str, str]:
    if target == "PERM" and "y_true_log10" in frame and frame["y_true_log10"].notna().any():
        return "y_true_log10", "interval_mean_log10", "lower_log10", "upper_log10"
    return "y_true", "interval_mean", "lower", "upper"


def _plot_prediction_interval(predictions: pd.DataFrame, seed: int, artifact: FigureArtifact) -> None:
    targets = _ordered_targets(predictions.target.unique())
    fig, axes = plt.subplots(len(targets), 1, figsize=(10.0, 7.3), sharex=False)
    for ax, target in zip(np.atleast_1d(axes), targets):
        part = predictions[predictions.target == target].sort_values("source_index", kind="stable").reset_index(drop=True)
        if len(part) > 350:
            idx = np.linspace(0, len(part) - 1, 350, dtype=int)
            part = part.iloc[idx].reset_index(drop=True)
        y, mean, lower, upper = _target_prediction_columns(part, target)
        x = np.arange(len(part))
        ax.plot(x, part[y], color="#222222", lw=1.0, label="真实值")
        ax.plot(x, part[mean], color="#2C7FB8", lw=1.0, label="预测均值")
        ax.fill_between(x, part[lower].astype(float), part[upper].astype(float), color="#7FCDBB", alpha=0.35, label="预测区间")
        ax.set_ylabel(target + (" (log10)" if target == "PERM" else ""))
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel("按源索引排序的抽样点")
    axes[0].legend(loc="upper right", ncol=3, frameon=False)
    fig.suptitle(f"油田M5、N=125代表性中位种子（seed={seed}）预测区间")
    _save(fig, artifact)


def _plot_coverage_width(results: ResultTables, artifact: FigureArtifact) -> None:
    frame = results.metrics.query("section == 'active' and dataset == 'field' and method == 'M5'").copy()
    grouped = frame.groupby(["target", "budget_N"])[["picp", "mpiw"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    budget_values = sorted(int(value) for value in grouped.budget_N.unique())
    marker_cycle = ["o", "s", "^", "D", "P", "X"]
    markers = {budget: marker_cycle[index % len(marker_cycle)] for index, budget in enumerate(budget_values)}
    colors = {"PHIF": "#4C78A8", "SW": "#F58518", "PERM": "#54A24B"}
    for _, row in grouped.iterrows():
        ax.scatter(row.mpiw, row.picp, s=75, marker=markers[int(row.budget_N)], color=colors[row.target], edgecolor="white", linewidth=0.7)
        ax.annotate(f"{row.target}, N={int(row.budget_N)}", (row.mpiw, row.picp), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0.95, color="#777777", ls="--", lw=1, label="目标覆盖率0.95")
    ax.set_xlabel("平均区间宽度（MPIW）")
    ax.set_ylabel("预测区间覆盖率（PICP）")
    ax.set_ylim(0.85, 1.01)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    ax.set_title("油田M5覆盖率—区间宽度权衡（五种子均值）")
    _save(fig, artifact)


def _plot_uncertainty_error(predictions: pd.DataFrame, seed: int, artifact: FigureArtifact) -> None:
    targets = _ordered_targets(predictions.target.unique())
    fig, axes = plt.subplots(1, len(targets), figsize=(12.0, 3.8))
    for ax, target in zip(np.atleast_1d(axes), targets):
        part = predictions[predictions.target == target].copy()
        y, mean, _, _ = _target_prediction_columns(part, target)
        part["absolute_error"] = (part[mean] - part[y]).abs()
        part = part.replace([np.inf, -np.inf], np.nan).dropna(subset=["total_variance", "absolute_error"])
        if len(part) > 4000:
            part = part.sample(4000, random_state=20260703)
        ax.scatter(part.total_variance, part.absolute_error, s=5, alpha=0.12, color="#4C78A8")
        try:
            part["bin"] = pd.qcut(part.total_variance, 12, duplicates="drop")
            trend = part.groupby("bin", observed=True)[["total_variance", "absolute_error"]].median()
            ax.plot(trend.total_variance, trend.absolute_error, color="#E45756", marker="o", lw=1.5)
        except ValueError:
            pass
        rho = part[["total_variance", "absolute_error"]].corr(method="spearman").iloc[0, 1]
        ax.set_title(f"{target}  Spearman ρ={rho:.2f}")
        ax.set_xlabel("总方差")
        ax.set_ylabel("绝对误差" + ("（log10）" if target == "PERM" else ""))
        ax.grid(alpha=0.15)
    fig.suptitle(f"油田M5、N=125代表性中位种子（seed={seed}）误差—不确定性关系", y=1.03)
    _save(fig, artifact)


def _heatmap(ax: plt.Axes, data: pd.DataFrame, title: str, *, cmap: str = "RdBu_r", center: float | None = 0.0, fmt: str = ".1f") -> None:
    values = data.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if center is not None and finite.size:
        bound = max(abs(float(finite.min())), abs(float(finite.max())), 1.0)
        image = ax.imshow(values, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(vmin=-bound, vcenter=center, vmax=bound))
    else:
        image = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(data.columns)), [str(int(v)) if isinstance(v, (float, np.floating)) and float(v).is_integer() else str(v) for v in data.columns])
    ax.set_yticks(range(len(data.index)), [str(v) for v in data.index])
    ax.set_title(title)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isfinite(values[i, j]):
                ax.text(j, i, format(values[i, j], fmt), ha="center", va="center", fontsize=7)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _plot_active_improvement(results: ResultTables, artifact: FigureArtifact) -> None:
    frame = results.paired.query("section == 'active' and metric == 'rmse' and variant != 'M5'").copy()
    frame["improvement_pct"] = pd.to_numeric(frame.improvement_fraction, errors="coerce") * 100.0
    fig, axes = plt.subplots(3, 1, figsize=(7.6, 11.0))
    for ax, dataset in zip(axes, ["field", "spwla", "forward"]):
        part = frame[frame.dataset == dataset].copy()
        part["row"] = part.variant.astype(str) + "—" + part.target.astype(str)
        pivot = part.pivot(index="row", columns="budget_N", values="improvement_pct")
        order = [f"{m}—{t}" for m in ["M6", "M7", "M8", "M9"] for t in TARGET_ORDER if f"{m}—{t}" in pivot.index]
        pivot = pivot.reindex(order)
        _heatmap(ax, pivot, DATASET_NAMES[dataset])
        ax.set_xlabel("终态预算")
    fig.suptitle("主动学习相对静态M5的五种子配对RMSE改善率（%）", y=1.01)
    _save(fig, artifact)


def _active_round_frame(results: ResultTables) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    selected = results.metrics.query("section == 'active' and budget_N == 125 and method in ['M6','M7','M8','M9']")
    for path_text in selected.metrics_path.drop_duplicates():
        path = results.repo / Path(str(path_text)).parent / "active_learning_metrics.csv"
        if not path.exists():
            continue
        context = selected[selected.metrics_path == path_text].iloc[0]
        frame = pd.read_csv(path)
        frame["dataset"] = context.dataset
        frame["method"] = context.method
        frame["seed"] = int(context.seed)
        records.append(frame)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _plot_active_process(results: ResultTables, artifact: FigureArtifact) -> None:
    frame = _active_round_frame(results)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True)
    for ax, dataset in zip(axes, ["field", "spwla", "forward"]):
        part = frame[frame.dataset == dataset]
        for method in ["M6", "M7", "M8", "M9"]:
            grouped = part[part.method == method].groupby("round").mean_score.agg(["mean", "std"]).reset_index()
            if grouped.empty:
                continue
            ax.plot(grouped["round"], grouped["mean"], marker="o", color=METHOD_COLORS[method], label=method)
            ax.fill_between(grouped["round"], grouped["mean"] - grouped["std"], grouped["mean"] + grouped["std"], color=METHOD_COLORS[method], alpha=0.14)
        ax.set_title(DATASET_NAMES[dataset])
        ax.set_xlabel("主动学习轮次")
        ax.set_ylabel("平均选样得分")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, ncol=2)
    fig.suptitle("终态预算125条件下的主动学习选样过程（五种子均值±标准差）", y=1.03)
    _save(fig, artifact)


def _plot_domain_transfer(results: ResultTables, artifact: FigureArtifact) -> None:
    frame = results.paired.query("metric == 'rmse' and section in ['backbone','transfer']").copy()
    frame["improvement_pct"] = pd.to_numeric(frame.improvement_fraction, errors="coerce") * 100.0
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    backbone = frame[frame.section == "backbone"].copy()
    backbone["row"] = backbone.dataset.map(DATASET_NAMES) + "—" + backbone.target
    pivot = backbone.pivot(index="row", columns="budget_N", values="improvement_pct")
    _heatmap(axes[0], pivot, "Target-aware相对普通Bi-LSTM")
    axes[0].set_xlabel("初始标注预算")
    transfer = frame[frame.section == "transfer"].copy()
    transfer["row"] = transfer.protocol.fillna(transfer.dataset) + "—" + transfer.target
    series = transfer.set_index("row")[["improvement_pct"]].rename(columns={"improvement_pct": "M9 vs M5"})
    _heatmap(axes[1], series, "迁移协议中M9相对M5")
    axes[1].set_xlabel("配对改善率（%）")
    fig.suptitle("数据生成机制与迁移协议下的配对RMSE改善率（%）", y=1.03)
    _save(fig, artifact)


def _plot_seed_stability(results: ResultTables, artifact: FigureArtifact) -> None:
    frame = results.metrics.query("section == 'backbone'").copy()
    base = frame[frame.variant == "bilstm"][["dataset", "target", "budget_N", "seed", "rmse"]].rename(columns={"rmse": "baseline"})
    cand = frame[frame.variant == "target_aware_bilstm"][["dataset", "target", "budget_N", "seed", "rmse"]].rename(columns={"rmse": "candidate"})
    paired = base.merge(cand, on=["dataset", "target", "budget_N", "seed"])
    paired["improvement"] = paired_rmse_improvement(paired, baseline="baseline", candidate="candidate")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), sharey=True)
    for ax, dataset in zip(axes, ["field", "spwla", "forward"]):
        part = paired[paired.dataset == dataset]
        labels = [f"{t}\nN={int(n)}" for t, n in part[["target", "budget_N"]].drop_duplicates().itertuples(index=False, name=None)]
        groups = [part[(part.target == t) & (part.budget_N == n)].improvement.to_numpy() for t, n in part[["target", "budget_N"]].drop_duplicates().itertuples(index=False, name=None)]
        ax.boxplot(groups, tick_labels=labels, showmeans=True)
        ax.axhline(0, color="#777777", lw=1)
        ax.set_title(DATASET_NAMES[dataset])
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("配对RMSE改善率（%）")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Target-aware结构的五种子稳定性")
    _save(fig, artifact)


def _plot_method_ranking(results: ResultTables, artifact: FigureArtifact) -> None:
    path = results.repo / "outputs" / "dataset_completion_check" / "matrix_all_metrics_long.csv"
    frame = pd.read_csv(path)
    frame = frame[frame.method.isin([f"M{i}" for i in range(1, 10)])].copy()
    frame["rank"] = frame.groupby(["dataset", "target", "N", "seed"]).rmse.rank(method="average")
    ranking = frame.groupby(["dataset", "method"])["rank"].agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    for ax, dataset in zip(axes, ["field", "spwla", "forward"]):
        part = ranking[ranking.dataset == dataset].set_index("method").reindex([f"M{i}" for i in range(1, 10)]).reset_index()
        ax.bar(part.method, part["mean"], yerr=part["std"], color=[METHOD_COLORS.get(m, "#777777") for m in part.method], capsize=2)
        ax.set_title(DATASET_NAMES[dataset])
        ax.set_xlabel("方法")
        ax.set_ylabel("平均任务内排名（越低越好）")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("M1—M9跨目标、预算和种子的任务内排名")
    _save(fig, artifact)


def _calibration_weight_frame(results: ResultTables) -> pd.DataFrame:
    selected = results.metrics.query("section == 'calibration'")
    rows: list[dict[str, object]] = []
    for path_text in selected.metrics_path.drop_duplicates():
        context = selected[selected.metrics_path == path_text].iloc[0]
        path = results.repo / Path(str(path_text)).parent / "calibration.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"dataset": context.dataset, "variant": context.variant, "budget_N": context.budget_N, "seed": int(context.seed), "lambda_mc": payload.get("lambda_mc", np.nan)})
    return pd.DataFrame(rows).drop_duplicates()


def _plot_calibration_weights(results: ResultTables, artifact: FigureArtifact) -> None:
    frame = _calibration_weight_frame(results)
    grouped = frame.groupby(["dataset", "variant", "budget_N"]).lambda_mc.mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7), sharey=True)
    for ax, dataset in zip(axes, ["field", "spwla", "forward"]):
        part = grouped[grouped.dataset == dataset]
        for variant, color in [("nll", "#4C78A8"), ("interval_score", "#E45756")]:
            p = part[part.variant == variant].sort_values("budget_N")
            ax.plot(p.budget_N, p.lambda_mc, marker="o", color=color, label=variant)
        ax.set_title(DATASET_NAMES[dataset])
        ax.set_xlabel("标注预算")
        ax.set_ylabel("λmc")
        ax.set_ylim(-0.05, 1.15)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    fig.suptitle("两种校准目标选择的MC融合权重")
    _save(fig, artifact)


def _trust_frame(results: ResultTables) -> pd.DataFrame:
    selected = results.metrics.query("section == 'active' and dataset == 'field' and method == 'M5' and budget_N == 125")
    rows: list[pd.DataFrame] = []
    for path_text in selected.metrics_path.drop_duplicates():
        context = selected[selected.metrics_path == path_text].iloc[0]
        path = results.repo / Path(str(path_text)).parent / "trust_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["seed"] = int(context.seed)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _plot_trust_status(results: ResultTables, artifact: FigureArtifact) -> None:
    frame = _trust_frame(results)
    grouped = frame.groupby("target")[["trusted_rate", "suspect_rate"]].mean().reindex(["PHIF", "SW", "PERM"]).fillna(0)
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.bar(grouped.index, grouped.trusted_rate * 100, label="可信", color="#54A24B")
    ax.bar(grouped.index, grouped.suspect_rate * 100, bottom=grouped.trusted_rate * 100, label="需复核", color="#E45756")
    for i, row in enumerate(grouped.itertuples()):
        ax.text(i, row.trusted_rate * 50, f"{row.trusted_rate*100:.1f}%", ha="center", va="center", color="white")
        if row.suspect_rate > 0.005:
            ax.text(i, row.trusted_rate * 100 + row.suspect_rate * 50, f"{row.suspect_rate*100:.1f}%", ha="center", va="center", color="white")
    ax.set_ylim(0, 100)
    ax.set_ylabel("样本比例（%）")
    ax.set_title("油田M5、N=125可信状态构成（五种子均值）")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, artifact)


def build_all_figures(results: ResultTables, output_dir: Path) -> list[FigureArtifact]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    artifacts = expected_artifacts(output_dir)
    by_id = {item.figure_id: item for item in artifacts}
    prediction_path, seed = _representative_prediction_path(results)
    predictions = _load_predictions(prediction_path)
    _plot_learning_curves(results, by_id["1"])
    _plot_field_backbone(results, by_id["2"])
    _plot_prediction_interval(predictions, seed, by_id["3"])
    _plot_coverage_width(results, by_id["4"])
    _plot_uncertainty_error(predictions, seed, by_id["5"])
    _plot_active_improvement(results, by_id["6"])
    _plot_active_process(results, by_id["7"])
    _plot_domain_transfer(results, by_id["8"])
    _plot_seed_stability(results, by_id["S1"])
    _plot_method_ranking(results, by_id["S2"])
    _plot_calibration_weights(results, by_id["S3"])
    _plot_trust_status(results, by_id["S4"])
    return artifacts


def _write_manifest(artifacts: list[FigureArtifact], output_dir: Path) -> None:
    rows = []
    for item in artifacts:
        rows.append({
            "figure_id": item.figure_id,
            "title": item.title,
            "anchor": item.anchor,
            "caption": item.caption,
            "pdf_path": str(item.pdf_path),
            "svg_path": str(item.svg_path),
            "pdf_bytes": item.pdf_path.stat().st_size,
            "svg_bytes": item.svg_path.stat().st_size,
        })
    pd.DataFrame(rows).to_csv(output_dir / "figure_manifest.csv", index=False, encoding="utf-8-sig")


def _write_audit(results: ResultTables, artifacts: list[FigureArtifact], output_dir: Path) -> None:
    records = []
    for item in artifacts:
        records.append({
            "figure_id": item.figure_id,
            "source": "supplementary_metrics_long.csv; paired_comparisons.csv; per-run artifacts",
            "seed_set": "0,1,2,3,4",
            "exclusions": "smoke tests; incomplete runs; best-seed selection",
        })
    pd.DataFrame(records).to_csv(output_dir / "figure_data_audit.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build manuscript figures from completed BNN outputs.")
    parser.add_argument("--results", type=Path, default=Path("outputs/supplementary/summary"))
    parser.add_argument("--output", type=Path, default=Path("figure"))
    parser.add_argument("--repo", type=Path, default=Path("."))
    args = parser.parse_args()
    results = load_results(args.results.resolve(), args.repo.resolve())
    artifacts = build_all_figures(results, args.output.resolve())
    _write_manifest(artifacts, args.output.resolve())
    _write_audit(results, artifacts, args.output.resolve())
    print(f"generated {len(artifacts)} figures ({len(artifacts) * 2} vector files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
