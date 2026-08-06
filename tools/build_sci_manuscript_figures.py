from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from bnn_inversion.publication_style import PALETTE, add_panel_labels, audit_layout, new_figure


class Artifact(NamedTuple):
    figure_id: str
    stem: str
    pdf: Path
    svg: Path


SPECS = [
    ("1", "fig01_learning_curves"), ("2", "fig02_field_backbone"),
    ("3", "fig03_field_prediction_interval"), ("4", "fig04_coverage_width"),
    ("5", "fig05_uncertainty_error"), ("6", "fig06_active_improvement"),
    ("7", "fig07_active_selection_process"), ("8", "fig08_domain_transfer"),
    ("S1", "figS01_seed_stability"), ("S2", "figS02_method_ranking"),
    ("S3", "figS03_calibration_weights"), ("S4", "figS04_trust_status"),
]
DATASET_NAMES = {"field": "Field", "spwla": "SPWLA", "forward": "Forward"}
TARGETS = ["PHIF", "SW", "PERM", "VSH"]
TARGET_STYLE = {
    "PHIF": (PALETTE["primary"], "o", "-"),
    "SW": (PALETTE["negative"], "s", "--"),
    "PERM": (PALETTE["text"], "^", "-."),
    "VSH": (PALETTE["text"], "D", ":"),
}
METHOD_STYLE = {
    "M6": (PALETTE["secondary"], "-"), "M7": (PALETTE["primary"], "--"),
    "M8": (PALETTE["text"], "-."), "M9": (PALETTE["negative"], ":"),
}
CMAP = LinearSegmentedColormap.from_list("sci_diverging", ["#4D6F93", "#F7F7F5", "#B85C4A"])


def artifacts(output: Path) -> list[Artifact]:
    return [Artifact(fid, stem, output / f"{stem}.pdf", output / f"{stem}.svg") for fid, stem in SPECS]


def _grid(ax) -> None:
    ax.grid(True, axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.65, zorder=0)


def _save(fig, artifact: Artifact, audits: list[dict]) -> None:
    fig.canvas.draw()
    audits.append(audit_layout(fig, artifact.figure_id))
    fig.savefig(artifact.pdf, facecolor="white")
    fig.savefig(artifact.svg, facecolor="white")
    preview = artifact.pdf.parent.parent / ".docx_work" / "sci_figure_previews"
    preview.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview / f"{artifact.stem}.png", dpi=140, facecolor="white")
    plt.close(fig)


def _panel_title(ax, title: str) -> None:
    ax.text(0.5, 1.03, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5)


def _figure1(metrics, artifact, audits):
    frame = metrics.query("section == 'active' and method == 'M5'")
    fig, axes = new_figure(1, 3, height_mm=63, sharex=True)
    for ax, dataset in zip(axes, ["field", "spwla", "forward"]):
        part = frame[frame.dataset == dataset]
        for target in [t for t in TARGETS if t in set(part.target)]:
            summary = part[part.target == target].groupby("final_budget").rmse.agg(["mean", "std"]).reset_index()
            color, marker, line = TARGET_STYLE[target]
            ax.errorbar(summary.final_budget, summary["mean"], yerr=summary["std"], color=color, marker=marker, linestyle=line, capsize=2, label=target)
        ax.set_yscale("log"); ax.set_xlabel("Final label budget"); ax.set_ylabel("RMSE")
        _panel_title(ax, DATASET_NAMES[dataset]); _grid(ax)
    add_panel_labels(axes)
    present = [t for t in TARGETS if t in set(frame.target)]
    handles = [Line2D([0],[0], color=TARGET_STYLE[t][0], marker=TARGET_STYLE[t][1], linestyle=TARGET_STYLE[t][2], label=t) for t in present]
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), frameon=False)
    _save(fig, artifact, audits)


def _figure2(metrics, artifact, audits):
    frame = metrics.query("section == 'backbone' and dataset == 'field'")
    targets = [t for t in TARGETS if t in set(frame.target)]
    fig, axes = new_figure(1, len(targets), height_mm=66)
    rng = np.random.default_rng(20260706)
    for ax, target in zip(axes, targets):
        part = frame[frame.target == target]; budgets = sorted(part.final_budget.unique())
        for i, budget in enumerate(budgets):
            for off, variant, color in [(-0.17, "bilstm", PALETTE["secondary"]), (0.17, "target_aware_bilstm", PALETTE["primary"])]:
                values = part[(part.final_budget == budget) & (part.variant == variant)].rmse.to_numpy()
                ax.bar(i + off, values.mean(), width=0.3, facecolor=color, edgecolor=PALETTE["text"], linewidth=0.45, zorder=2)
                ax.scatter(i + off + rng.normal(0, 0.015, len(values)), values, s=10, facecolors="white", edgecolors=PALETTE["text"], linewidths=0.45, zorder=3)
        ax.set_xticks(range(len(budgets)), [f"N={int(v)}" for v in budgets]); ax.set_ylabel("RMSE")
        _panel_title(ax, target + (" (log10)" if target == "PERM" else "")); _grid(ax)
    add_panel_labels(axes)
    fig.legend([plt.Rectangle((0,0),1,1,color=PALETTE["secondary"]), plt.Rectangle((0,0),1,1,color=PALETTE["primary"])], ["Bi-LSTM", "Target-aware"], loc="outside upper center", ncol=2, frameon=False)
    _save(fig, artifact, audits)


def _pred_cols(part, target):
    if target == "PERM" and part.y_true_log10.notna().any():
        return "y_true_log10", "interval_mean_log10", "lower_log10", "upper_log10"
    return "y_true", "interval_mean", "lower", "upper"


def _figure3(pred, artifact, audits):
    targets = [t for t in TARGETS if t in set(pred.target)]
    fig, axes = new_figure(len(targets), 1, height_mm=150)
    for ax, target in zip(axes, targets):
        part = pred[pred.target == target].sort_values("source_index").reset_index(drop=True)
        y, mean, lo, hi = _pred_cols(part, target); x = np.arange(len(part))
        ax.fill_between(x, part[lo], part[hi], color=PALETTE["secondary"], alpha=0.28, linewidth=0)
        ax.plot(x, part[y], color=PALETTE["text"], linewidth=0.9, label="Reference")
        ax.plot(x, part[mean], color=PALETTE["primary"], linewidth=1.0, label="Prediction")
        ax.set_ylabel(target + (" (log10)" if target == "PERM" else "")); ax.set_xlabel("Samples ordered by source index")
        _grid(ax)
    add_panel_labels(axes)
    handles = [Line2D([0],[0],color=PALETTE["text"],label="Reference"), Line2D([0],[0],color=PALETTE["primary"],label="Prediction"), plt.Rectangle((0,0),1,1,color=PALETTE["secondary"],alpha=.28,label="Prediction interval")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
    _save(fig, artifact, audits)


def _figure4(metrics, artifact, audits):
    frame = metrics.query("section == 'active' and dataset == 'field' and method == 'M5'")
    grouped = frame.groupby(["target", "final_budget"])[["picp", "mpiw"]].mean().reset_index()
    fig, ax = new_figure(1, 1, height_mm=105)
    marker_map = {125:"o", 200:"s", 500:"^"}
    for target in [t for t in TARGETS if t in set(grouped.target)]:
        color = TARGET_STYLE[target][0]
        for _, row in grouped[grouped.target == target].iterrows():
            ax.scatter(row.mpiw, row.picp, s=34, marker=marker_map.get(int(row.final_budget), "o"), color=color, edgecolor="white", linewidth=.5)
    ax.axhline(.95, color=PALETTE["light"], linestyle="--", linewidth=.8)
    ax.set_xlabel("Mean prediction interval width (MPIW)"); ax.set_ylabel("Prediction interval coverage probability (PICP)"); ax.set_ylim(.85, 1.01); _grid(ax)
    target_handles = [Line2D([0],[0],marker=TARGET_STYLE[t][1],color="none",markerfacecolor=TARGET_STYLE[t][0],markeredgecolor="white",label=t) for t in ["PHIF","SW","PERM"]]
    budget_handles = [Line2D([0],[0],marker=m,color="none",markerfacecolor=PALETTE["light"],label=f"N={n}") for n,m in marker_map.items()]
    fig.legend(handles=target_handles+budget_handles, loc="outside upper center", ncol=6, frameon=False)
    _save(fig, artifact, audits)


def _figure5(pred, artifact, audits):
    targets = [t for t in TARGETS if t in set(pred.target)]
    fig, axes = new_figure(1, len(targets), height_mm=67)
    for ax, target in zip(axes, targets):
        part = pred[pred.target == target].copy(); y, mean, _, _ = _pred_cols(part, target)
        part["error"] = (part[mean] - part[y]).abs(); part = part.dropna(subset=["total_variance", "error"])
        ax.scatter(part.total_variance, part.error, s=5, alpha=.22, color=PALETTE["light"], edgecolors="none")
        part["bin"] = pd.qcut(part.total_variance, 10, duplicates="drop")
        trend = part.groupby("bin", observed=True)[["total_variance","error"]].median()
        ax.plot(trend.total_variance, trend.error, color=PALETTE["negative"], marker="o", markersize=3)
        rho = part[["total_variance","error"]].corr(method="spearman").iloc[0,1]
        _panel_title(ax, f"{target}, ρ={rho:.2f}"); ax.set_xlabel("Total variance"); ax.set_ylabel("Absolute error"); _grid(ax)
    add_panel_labels(axes); _save(fig, artifact, audits)


def _heatmap(ax, pivot, norm, *, annotate=True):
    im = ax.imshow(pivot.to_numpy(float), aspect="auto", cmap=CMAP, norm=norm)
    ax.set_xticks(range(len(pivot.columns)), [str(int(v)) if isinstance(v,(int,float,np.integer,np.floating)) else str(v) for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    if annotate:
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                value = pivot.iloc[i,j]
                if pd.notna(value): ax.text(j,i,f"{value:.1f}",ha="center",va="center",fontsize=7.5)
    return im


def _figure6(paired, artifact, audits):
    frame = paired[paired.comparison.str.contains("_vs_M5") & ~paired.comparison.str.contains("transfer")].copy()
    frame["method"] = frame.comparison.str.extract(r"(M\d)")
    pivots=[]; datasets=["field","spwla","forward"]
    for dataset in datasets:
        part=frame[frame.dataset==dataset].copy(); part["row"]=part.method+"—"+part.target
        pivots.append(part.groupby(["row","final_budget"]).improvement_pct.mean().unstack())
    bound=max(abs(p.to_numpy(float)[np.isfinite(p.to_numpy(float))]).max() for p in pivots); norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound)
    fig, axes = new_figure(3,1,height_mm=235)
    for ax,dataset,pivot in zip(axes,datasets,pivots): _heatmap(ax,pivot,norm); _panel_title(ax,DATASET_NAMES[dataset]); ax.set_xlabel("Final label budget")
    add_panel_labels(axes); fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=CMAP),ax=axes,location="right",label="RMSE improvement (%)",shrink=.85)
    _save(fig,artifact,audits)


def _figure7(active, artifact, audits):
    frame=active[active.final_budget==125]
    fig,axes=new_figure(1,3,height_mm=65,sharex=True)
    for ax,dataset in zip(axes,["field","spwla","forward"]):
        part=frame[frame.dataset==dataset]
        for method,(color,line) in METHOD_STYLE.items():
            summary=part[part.method==method].groupby("round").mean_score.agg(["mean","std"]).reset_index()
            ax.plot(summary["round"],summary["mean"],color=color,linestyle=line,marker="o",label=method)
            ax.fill_between(summary["round"],summary["mean"]-summary["std"],summary["mean"]+summary["std"],color=color,alpha=.10,linewidth=0)
        _panel_title(ax,DATASET_NAMES[dataset]); ax.set_xlabel("Active-learning round"); ax.set_ylabel("Mean acquisition score"); _grid(ax)
    add_panel_labels(axes)
    handles = [Line2D([0],[0], color=METHOD_STYLE[m][0], linestyle=METHOD_STYLE[m][1], marker="o", label=m) for m in ["M6","M7","M8","M9"]]
    fig.legend(handles=handles, loc="outside upper center",ncol=4,frameon=False); _save(fig,artifact,audits)


def _figure8(paired, artifact, audits):
    backbone=paired[paired.comparison=="target_aware_vs_bilstm"].copy(); backbone["row"]=backbone.dataset.map(DATASET_NAMES)+"—"+backbone.target
    p1=backbone.groupby(["row","final_budget"]).improvement_pct.mean().unstack()
    transfer=paired[paired.comparison=="M9_vs_M5_transfer"].copy(); transfer["row"]=transfer.protocol+"—"+transfer.target
    p2=transfer.groupby("row").improvement_pct.mean().to_frame("M9 vs M5")
    vals=np.r_[p1.to_numpy().ravel(),p2.to_numpy().ravel()]; vals=vals[np.isfinite(vals)]; bound=max(abs(vals)); norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound)
    fig,axes=new_figure(1,2,height_mm=105); _heatmap(axes[0],p1,norm); _heatmap(axes[1],p2,norm)
    _panel_title(axes[0],"Target-aware vs Bi-LSTM"); _panel_title(axes[1],"Transfer: M9 vs M5"); add_panel_labels(axes)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=CMAP),ax=axes,location="right",label="RMSE improvement (%)",shrink=.75); _save(fig,artifact,audits)


def _figure_s1(paired, artifact, audits):
    frame=paired[paired.comparison=="target_aware_vs_bilstm"]
    fig,axes=new_figure(1,3,height_mm=75,sharey=True)
    for ax,dataset in zip(axes,["field","spwla","forward"]):
        part=frame[frame.dataset==dataset]; groups=[]; labels=[]
        for (target,budget),group in part.groupby(["target","final_budget"]): groups.append(group.improvement_pct); labels.append(f"{target}\nN={int(budget)}")
        bp=ax.boxplot(groups,tick_labels=labels,patch_artist=True,showmeans=True,medianprops={"color":PALETTE["negative"]})
        for box in bp["boxes"]: box.set(facecolor=PALETTE["secondary"],edgecolor=PALETTE["primary"],linewidth=.7)
        ax.axhline(0,color=PALETTE["light"],linewidth=.8); ax.tick_params(axis="x",rotation=35); ax.set_ylabel("RMSE improvement (%)"); _panel_title(ax,DATASET_NAMES[dataset]); _grid(ax)
    add_panel_labels(axes); _save(fig,artifact,audits)


def _figure_s2(repo, artifact, audits):
    frame=pd.read_csv(repo/"outputs/dataset_completion_check/matrix_all_metrics_long.csv"); frame=frame[frame.method.str.match(r"M[1-9]$")].copy()
    frame["rank"]=frame.groupby(["dataset","target","N","seed"]).rmse.rank(); summary=frame.groupby(["dataset","method"])["rank"].agg(["mean","std"]).reset_index()
    fig,axes=new_figure(1,3,height_mm=65,sharey=True)
    for ax,dataset in zip(axes,["field","spwla","forward"]):
        part=summary[summary.dataset==dataset].set_index("method").reindex([f"M{i}" for i in range(1,10)]).reset_index()
        ax.bar(part.method,part["mean"],yerr=part["std"],color=PALETTE["primary"],edgecolor=PALETTE["text"],linewidth=.4,capsize=2)
        ax.set_ylabel("Mean within-task rank"); _panel_title(ax,DATASET_NAMES[dataset]); _grid(ax)
    add_panel_labels(axes); _save(fig,artifact,audits)


def _figure_s3(calibration, artifact, audits):
    frame=calibration[calibration.method=="M5"]
    summary=frame.groupby(["dataset","variant"]).lambda_mc.agg(["mean","std"]).reset_index()
    fig,ax=new_figure(1,1,height_mm=78)
    datasets=["field","spwla","forward"]; x=np.arange(len(datasets)); width=.28
    for offset,variant,color,label in [(-width/2,"nll",PALETTE["primary"],"NLL"),(width/2,"interval_score",PALETTE["negative"],"Interval score")]:
        part=summary[summary.variant==variant].set_index("dataset").reindex(datasets)
        ax.errorbar(x+offset,part["mean"],yerr=part["std"].fillna(0),fmt="o",color=color,capsize=3,label=label)
    ax.set_xticks(x,[DATASET_NAMES[d] for d in datasets]); ax.set_ylabel("λmc"); ax.set_ylim(-.05,1.1); _grid(ax)
    fig.legend(loc="outside upper center",ncol=2,frameon=False); _save(fig,artifact,audits)


def _figure_s4(trust, artifact, audits):
    summary=trust.groupby("target")[["trusted_rate","suspect_rate"]].mean().reindex(["PHIF","SW","PERM"]).fillna(0)
    fig,ax=new_figure(1,1,height_mm=92)
    ax.bar(summary.index,summary.trusted_rate*100,color=PALETTE["trusted"],label="Trusted",edgecolor=PALETTE["text"],linewidth=.4)
    ax.bar(summary.index,summary.suspect_rate*100,bottom=summary.trusted_rate*100,color=PALETTE["negative"],label="Review required",edgecolor=PALETTE["text"],linewidth=.4)
    for i,row in enumerate(summary.itertuples()):
        ax.text(i,row.trusted_rate*50,f"{row.trusted_rate*100:.1f}%",ha="center",va="center",color="white",fontsize=8)
    ax.set_ylabel("Sample proportion (%)"); ax.set_ylim(0,100); _grid(ax); fig.legend(loc="outside upper center",ncol=2,frameon=False); _save(fig,artifact,audits)


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--repo",type=Path,default=Path(".")); parser.add_argument("--data",type=Path,default=Path("figure/data")); parser.add_argument("--output",type=Path,default=Path("figure")); args=parser.parse_args()
    repo=args.repo.resolve(); data=args.data.resolve(); output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    metrics=pd.read_csv(data/"metrics_seed_long.csv"); paired=pd.read_csv(data/"paired_improvement_seed_long.csv"); active=pd.read_csv(data/"active_round_long.csv"); pred=pd.read_csv(data/"representative_predictions.csv"); calibration=pd.read_csv(data/"calibration_weights_long.csv"); trust=pd.read_csv(data/"trust_status_long.csv")
    arts={a.figure_id:a for a in artifacts(output)}; audits=[]
    _figure1(metrics,arts["1"],audits); _figure2(metrics,arts["2"],audits); _figure3(pred,arts["3"],audits); _figure4(metrics,arts["4"],audits); _figure5(pred,arts["5"],audits); _figure6(paired,arts["6"],audits); _figure7(active,arts["7"],audits); _figure8(paired,arts["8"],audits); _figure_s1(paired,arts["S1"],audits); _figure_s2(repo,arts["S2"],audits); _figure_s3(calibration,arts["S3"],audits); _figure_s4(trust,arts["S4"],audits)
    pd.DataFrame(audits).to_csv(output/"layout_audit.csv",index=False,encoding="utf-8-sig")
    manifest=pd.DataFrame([{"figure_id":a.figure_id,"pdf_path":str(a.pdf),"svg_path":str(a.svg),"pdf_bytes":a.pdf.stat().st_size,"svg_bytes":a.svg.stat().st_size} for a in arts.values()]); manifest.to_csv(output/"figure_manifest.csv",index=False,encoding="utf-8-sig")
    print(f"generated {len(arts)} SCI figures")
    return 0


if __name__=="__main__": raise SystemExit(main())
