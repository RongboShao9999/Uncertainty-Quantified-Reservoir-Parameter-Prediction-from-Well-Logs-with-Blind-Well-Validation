from __future__ import annotations

import html
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "outputs" / "metric_evaluation_exact"
WORK = REPO / ".docx_work" / "results_section"
FIGURE = REPO / "figure"
OUTPUT = REPO / "实验结果_重新组织_含表图.docx"

DATASET_LABELS = {
    "field": "油田实测数据",
    "spwla": "SPWLA竞赛数据",
    "forward": "正演生成数据",
}
PLOT_DATASET_LABELS = {"field": "Field", "spwla": "SPWLA", "forward": "Forward"}
TARGET_LABELS = {"PHIF": "孔隙度", "SW": "含水饱和度", "PERM": "渗透率", "VSH": "泥质含量"}


def _fmt(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.{digits}f}"


def _pct(value: object, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def _load_tables() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_csv(EVAL / f"{name}.csv")
        for name in ("metrics", "summary", "trust", "active_efficiency")
    }


def dataset_coverage_table(tables: dict[str, pd.DataFrame]) -> list[list[str]]:
    metrics = tables["metrics"]
    trust = tables["trust"]
    rows = [["数据集", "数据属性", "目标参数", "预算点", "指标记录", "可信记录", "结果用途"]]
    notes = {
        "field": ("实际生产测井资料", "生产适用性核心验证"),
        "spwla": ("筛选后的优质公开实测数据", "受控同域对照"),
        "forward": ("纯机理正演生成数据", "理想条件与泛化边界"),
    }
    for dataset in ("field", "spwla", "forward"):
        frame = metrics[metrics["dataset"] == dataset]
        targets = "、".join(sorted(frame["target"].dropna().unique()))
        budgets = "、".join(str(int(v)) for v in sorted(frame["budget_N"].dropna().unique()))
        rows.append(
            [
                DATASET_LABELS[dataset],
                notes[dataset][0],
                targets,
                budgets,
                str(len(frame)),
                str(len(trust[trust["dataset"] == dataset])),
                notes[dataset][1],
            ]
        )
    return rows


def point_accuracy_table(summary: pd.DataFrame) -> list[list[str]]:
    frame = summary[(summary["budget_N"] == 500) & summary["method"].isin(["M1", "M5", "M9"])]
    rows = [["数据集", "目标", "M1 RMSE/R²", "M5 RMSE/R²", "M9 RMSE/R²", "主要观察"]]
    for dataset in ("field", "spwla", "forward"):
        for target in sorted(frame[frame["dataset"] == dataset]["target"].unique()):
            part = frame[(frame["dataset"] == dataset) & (frame["target"] == target)]
            cells = []
            for method in ("M1", "M5", "M9"):
                row = part[part["method"] == method]
                if row.empty:
                    cells.append("—")
                else:
                    item = row.iloc[0]
                    cells.append(f"{_fmt(item.rmse)}/{_fmt(item.r2)}")
            best = part.loc[part["rmse"].idxmin(), "method"] if not part.empty else "—"
            rows.append(
                [
                    DATASET_LABELS[dataset],
                    f"{target}（{TARGET_LABELS.get(target, target)}）",
                    *cells,
                    f"RMSE最优：{best}",
                ]
            )
    return rows


def uncertainty_table(summary: pd.DataFrame) -> list[list[str]]:
    frame = summary[(summary["budget_N"] == 500) & summary["method"].isin(["M5", "M9"])]
    rows = [["数据集", "目标", "方法", "PICP", "NMPIW", "UCE", "NLL", "Spearman ρ"]]
    for dataset in ("field", "spwla", "forward"):
        for target in sorted(frame[frame["dataset"] == dataset]["target"].unique()):
            for method in ("M5", "M9"):
                row = frame[
                    (frame["dataset"] == dataset)
                    & (frame["target"] == target)
                    & (frame["method"] == method)
                ]
                if row.empty:
                    continue
                item = row.iloc[0]
                rows.append(
                    [
                        DATASET_LABELS[dataset],
                        target,
                        method,
                        _fmt(item.picp),
                        _fmt(item.nmpiw),
                        _fmt(item.uce),
                        _fmt(item.nll),
                        _fmt(item.uncertainty_error_spearman),
                    ]
                )
    return rows


def trust_table(trust: pd.DataFrame) -> list[list[str]]:
    frame = trust[(trust["budget_N"] == 500) & trust["method"].isin(["M5", "M9"])]
    rows = [["数据集", "目标", "方法", "可信比例", "存疑比例", "可信MAE", "存疑MAE", "MAE差值"]]
    for dataset in ("field", "spwla", "forward"):
        for target in sorted(frame[frame["dataset"] == dataset]["target"].unique()):
            for method in ("M5", "M9"):
                part = frame[
                    (frame["dataset"] == dataset)
                    & (frame["target"] == target)
                    & (frame["method"] == method)
                ]
                if part.empty:
                    continue
                mean = part[["trusted_rate", "suspect_rate", "trusted_mae", "suspect_mae", "mae_gap"]].mean(numeric_only=True)
                rows.append(
                    [
                        DATASET_LABELS[dataset],
                        target,
                        method,
                        _pct(mean.trusted_rate),
                        _pct(mean.suspect_rate),
                        _fmt(mean.trusted_mae),
                        _fmt(mean.suspect_mae),
                        _fmt(mean.mae_gap),
                    ]
                )
    return rows


def active_table(active: pd.DataFrame) -> list[list[str]]:
    frame = active[active["method"] == "M9"]
    rows = [["数据集", "目标", "ΔRMSE", "相对M6改善", "AULC", "标注节省率", "解释"]]
    for dataset in ("field", "spwla", "forward"):
        for target in sorted(frame[frame["dataset"] == dataset]["target"].unique()):
            row = frame[(frame["dataset"] == dataset) & (frame["target"] == target)].iloc[0]
            rows.append(
                [
                    DATASET_LABELS[dataset],
                    target,
                    f"{_fmt(row.delta_rmse_pct, 1)}%",
                    f"{_fmt(row.relative_improvement_pct, 1)}%",
                    _fmt(row.aulc),
                    _pct(row.label_saving_rate),
                    "学习曲线有效" if pd.notna(row.aulc) else "仅终态预算对比",
                ]
            )
    return rows


def make_plots(tables: dict[str, pd.DataFrame]) -> dict[str, Path]:
    WORK.mkdir(parents=True, exist_ok=True)
    FIGURE.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    previews = [
        REPO / ".docx_work" / "dataset_previews" / f"{name}_dataset.png"
        for name in ("field", "spwla", "forward")
    ]
    images = [Image.open(path).convert("RGB") for path in previews if path.exists()]
    if images:
        thumb_width = 1300
        thumbs = []
        for image in images:
            ratio = thumb_width / image.width
            resized = image.resize((thumb_width, int(image.height * ratio)))
            thumbs.append(ImageOps.expand(resized, border=18, fill="white"))
        total_width = sum(image.width for image in thumbs)
        max_height = max(image.height for image in thumbs)
        canvas = Image.new("RGB", (total_width, max_height), "white")
        x = 0
        for image in thumbs:
            canvas.paste(image, (x, 0))
            x += image.width
        paths["dataset"] = WORK / "fig_dataset_overview.png"
        canvas.save(paths["dataset"], dpi=(220, 220))

    summary = tables["summary"]
    final = summary[summary["budget_N"] == 500].copy()
    method_order = ["M1", "M5", "M9"]
    colors = {"M1": "#6F6F6F", "M5": "#3B5B92", "M9": "#C65D4B"}
    plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 9, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    for ax, dataset in zip(axes, ("field", "spwla", "forward")):
        part = final[(final["dataset"] == dataset) & final["method"].isin(method_order)]
        xlabels = sorted(part["target"].unique())
        x = np.arange(len(xlabels))
        width = 0.24
        for i, method in enumerate(method_order):
            vals = [
                part[(part["target"] == target) & (part["method"] == method)]["rmse"].mean()
                for target in xlabels
            ]
            ax.bar(x + (i - 1) * width, vals, width=width, label=method, color=colors[method], alpha=0.9)
        ax.set_title(PLOT_DATASET_LABELS[dataset])
        ax.set_xticks(x, xlabels)
        ax.set_ylabel("RMSE")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncol=3)
    paths["accuracy"] = WORK / "fig_point_accuracy.png"
    fig.savefig(paths["accuracy"], dpi=260)
    fig.savefig(FIGURE / "fig_point_accuracy.pdf")
    fig.savefig(FIGURE / "fig_point_accuracy.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    unc = final[final["method"].isin(["M5", "M9"])]
    for ax, metric, ylabel in zip(
        axes,
        ("picp", "nmpiw", "uncertainty_error_spearman"),
        ("PICP", "NMPIW", "Spearman ρ"),
    ):
        plot = unc.groupby(["dataset", "method"], as_index=False)[metric].mean(numeric_only=True)
        labels = [PLOT_DATASET_LABELS[d] for d in ("field", "spwla", "forward")]
        x = np.arange(len(labels))
        for i, method in enumerate(["M5", "M9"]):
            vals = [
                plot[(plot["dataset"] == ds) & (plot["method"] == method)][metric].mean()
                for ds in ("field", "spwla", "forward")
            ]
            ax.bar(x + (i - 0.5) * 0.32, vals, width=0.32, label=method, color=colors[method], alpha=0.9)
        if metric == "picp":
            ax.axhline(0.95, color="#4D4D4D", linestyle="--", linewidth=0.9)
        ax.set_xticks(x, labels, rotation=15)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    paths["uncertainty"] = WORK / "fig_uncertainty_quality.png"
    fig.savefig(paths["uncertainty"], dpi=260)
    fig.savefig(FIGURE / "fig_uncertainty_quality.pdf")
    fig.savefig(FIGURE / "fig_uncertainty_quality.svg")
    plt.close(fig)

    trust = tables["trust"]
    trust_final = trust[(trust["budget_N"] == 500) & trust["method"].isin(["M5", "M9"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), constrained_layout=True)
    for ax, metric, ylabel in zip(axes, ("trusted_rate", "mae_gap"), ("Trusted rate", "Suspect MAE - Trusted MAE")):
        plot = trust_final.groupby(["dataset", "method"], as_index=False)[metric].mean(numeric_only=True)
        x = np.arange(3)
        for i, method in enumerate(["M5", "M9"]):
            vals = [plot[(plot["dataset"] == ds) & (plot["method"] == method)][metric].mean() for ds in ("field", "spwla", "forward")]
            ax.bar(x + (i - 0.5) * 0.32, vals, width=0.32, label=method, color=colors[method], alpha=0.9)
        ax.set_xticks(x, [PLOT_DATASET_LABELS[d] for d in ("field", "spwla", "forward")], rotation=15)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    paths["trust"] = WORK / "fig_trust_output.png"
    fig.savefig(paths["trust"], dpi=260)
    fig.savefig(FIGURE / "fig_trust_output.pdf")
    fig.savefig(FIGURE / "fig_trust_output.svg")
    plt.close(fig)

    active = tables["active_efficiency"]
    m9 = active[active["method"] == "M9"].groupby("dataset", as_index=False)[
        ["delta_rmse_pct", "relative_improvement_pct"]
    ].mean(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.2), constrained_layout=True)
    x = np.arange(3)
    for i, metric in enumerate(["delta_rmse_pct", "relative_improvement_pct"]):
        vals = [m9[m9["dataset"] == ds][metric].mean() for ds in ("field", "spwla", "forward")]
        ax.bar(x + (i - 0.5) * 0.32, vals, width=0.32, label=metric, color=["#3B5B92", "#C65D4B"][i], alpha=0.9)
    ax.set_xticks(x, [PLOT_DATASET_LABELS[d] for d in ("field", "spwla", "forward")], rotation=15)
    ax.set_ylabel("%")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(["ΔRMSE", "Relative improvement"], frameon=False)
    paths["active"] = WORK / "fig_active_learning.png"
    fig.savefig(paths["active"], dpi=260)
    fig.savefig(FIGURE / "fig_active_learning.pdf")
    fig.savefig(FIGURE / "fig_active_learning.svg")
    plt.close(fig)
    return paths


def _esc(text: object) -> str:
    return html.escape(str(text), quote=False)


def _r(text: object, *, bold: bool = False) -> str:
    bold_xml = "<w:b/>" if bold else ""
    return (
        "<w:r><w:rPr>"
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/>'
        f"{bold_xml}</w:rPr><w:t>{_esc(text)}</w:t></w:r>"
    )


def paragraph(text: str = "", style: str | None = None, *, bold: bool = False, align: str | None = None) -> str:
    ppr = ""
    if style:
        ppr += f'<w:pStyle w:val="{style}"/>'
    if align:
        ppr += f'<w:jc w:val="{align}"/>'
    if ppr:
        ppr = f"<w:pPr>{ppr}</w:pPr>"
    return f"<w:p>{ppr}{_r(text, bold=bold) if text else ''}</w:p>"


def table_xml(rows: list[list[str]]) -> str:
    col_count = max(len(row) for row in rows)
    grid = "".join('<w:gridCol w:w="1600"/>' for _ in range(col_count))
    parts = [
        '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="0" w:type="auto"/>'
        '<w:tblLook w:firstRow="1" w:noHBand="0" w:noVBand="1"/></w:tblPr>'
        f"<w:tblGrid>{grid}</w:tblGrid>"
    ]
    for row_index, row in enumerate(rows):
        parts.append("<w:tr>")
        for cell in row:
            shade = '<w:shd w:fill="D9EAF7"/>' if row_index == 0 else ""
            parts.append(
                "<w:tc><w:tcPr>"
                '<w:tcW w:w="1600" w:type="dxa"/>'
                f"{shade}</w:tcPr>"
                f"{paragraph(str(cell), bold=row_index == 0)}"
                "</w:tc>"
            )
        parts.append("</w:tr>")
    parts.append("</w:tbl>")
    return "".join(parts)


def image_xml(rid: str, path: Path, *, max_width_in: float = 6.5) -> str:
    image = Image.open(path)
    width_px, height_px = image.size
    width_in = min(max_width_in, width_px / 220)
    height_in = width_in * height_px / width_px
    cx = int(width_in * 914400)
    cy = int(height_in * 914400)
    return f"""
<w:p>
  <w:pPr><w:jc w:val="center"/></w:pPr>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:docPr id="{rid[3:]}" name="{_esc(path.name)}"/>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr><pic:cNvPr id="0" name="{_esc(path.name)}"/><pic:cNvPicPr/></pic:nvPicPr>
              <pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
              <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/><w:sz w:val="21"/></w:rPr><w:pPr><w:spacing w:line="360" w:lineRule="auto" w:after="120"/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="0"/><w:spacing w:before="240" w:after="160"/></w:pPr><w:rPr><w:b/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="1"/><w:spacing w:before="200" w:after="120"/></w:pPr><w:rPr><w:b/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/><w:sz w:val="26"/></w:rPr></w:style>
  <w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:color="BFBFBF"/><w:left w:val="single" w:sz="4" w:color="BFBFBF"/><w:bottom w:val="single" w:sz="4" w:color="BFBFBF"/><w:right w:val="single" w:sz="4" w:color="BFBFBF"/><w:insideH w:val="single" w:sz="4" w:color="D9D9D9"/><w:insideV w:val="single" w:sz="4" w:color="D9D9D9"/></w:tblBorders></w:tblPr></w:style>
</w:styles>"""


def content_types(image_count: int) -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""


def document_xml(body: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
<w:body>{body}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1200" w:bottom="1440" w:left="1200" w:header="720" w:footer="720" w:gutter="0"/></w:sectPr></w:body>
</w:document>"""


def relationships(images: list[Path]) -> str:
    rels = [
        '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>',
        '<Relationship Id="rIdSettings" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>',
    ]
    for index, path in enumerate(images, start=1):
        rels.append(
            f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{path.name}"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>"
    )


def build_body(tables: dict[str, pd.DataFrame], figures: dict[str, Path]) -> tuple[str, list[Path]]:
    summary = tables["summary"]
    paragraphs = [
        paragraph("4 实验结果与分析", "Heading1"),
        paragraph("本节依据已完成的多种子实验输出重新组织结果。评价以实际产物 metrics.csv、predictions.csv 和 trust_metrics.csv 为准；其中油田实测数据来源于致密砂泥岩储层，更接近生产解释场景，SPWLA 数据代表经过筛选的高质量公开实测数据，正演数据则代表理想机理条件。三类数据共同用于区分方法在真实生产、优质实测和理论验证数据上的表现差异。"),
        paragraph("4.1 评价数据覆盖与图表安排", "Heading2"),
        table_xml(dataset_coverage_table(tables)),
        paragraph("表4-1 三类数据集评价结果覆盖情况。"),
    ]
    images = []
    if "dataset" in figures:
        images.append(figures["dataset"])
        paragraphs.append(image_xml(f"rId{len(images)}", figures["dataset"]))
        paragraphs.append(paragraph("图4-1 三类数据集的测井—目标分布、相关性与完整性对比。油田实测数据反映生产资料中的噪声、缺失与井间差异；SPWLA 数据质量更高；正演数据分布受机理模型控制。", align="center"))

    paragraphs.extend(
        [
            paragraph("4.2 点预测精度", "Heading2"),
            paragraph("在终态预算 N=500 下，M1 表示仅点预测的 Bi-LSTM 基线，M5 表示融合不确定性但不引入主动学习的本文融合方法，M9 表示融合不确定性与主动学习的完整方法。由于不同储层参数量纲不同，渗透率在 log10(PERM) 空间评价，其余目标在归一化或原始比例空间评价。"),
            table_xml(point_accuracy_table(summary)),
            paragraph("表4-2 终态预算 N=500 下点预测精度对比。单元格为 RMSE/R²。"),
        ]
    )
    images.append(figures["accuracy"])
    paragraphs.append(image_xml(f"rId{len(images)}", figures["accuracy"]))
    paragraphs.append(paragraph("图4-2 终态预算下三类数据集的 RMSE 对比。油田实测数据整体误差更高，说明真实生产资料的非均质性和标签解释误差显著增加了建模难度。", align="center"))

    paragraphs.extend(
        [
            paragraph("4.3 不确定性质量", "Heading2"),
            paragraph("不确定性质量从覆盖率、区间宽度、校准误差、负对数似然和不确定性—误差相关性五个角度评价。理想情况下，PICP 应接近 0.95，NMPIW 不宜过宽，UCE 与 NLL 越低越好，Spearman ρ 越高说明模型越能把较大的不确定性分配给高误差样本。"),
            table_xml(uncertainty_table(summary)),
            paragraph("表4-3 终态预算 N=500 下 M5 与 M9 的不确定性质量。"),
        ]
    )
    images.append(figures["uncertainty"])
    paragraphs.append(image_xml(f"rId{len(images)}", figures["uncertainty"]))
    paragraphs.append(paragraph("图4-3 M5 与 M9 的覆盖率、归一化区间宽度和风险排序能力对比。覆盖率普遍高于名义 95%，说明当前区间偏保守；在生产实测数据上，该保守性有助于减少高风险样本被误判为可信输出。", align="center"))

    paragraphs.extend(
        [
            paragraph("4.4 可信/存疑输出效果", "Heading2"),
            paragraph("可信输出并不只追求较高可信比例，更重要的是存疑样本是否对应更高风险。若存疑样本 MAE 高于可信样本 MAE，则说明该判定机制能够将更可能失效的预测推送到人工复核或补充标定流程。"),
            table_xml(trust_table(tables["trust"])),
            paragraph("表4-4 终态预算 N=500 下可信/存疑判定结果。MAE差值为存疑MAE减可信MAE。"),
        ]
    )
    images.append(figures["trust"])
    paragraphs.append(image_xml(f"rId{len(images)}", figures["trust"]))
    paragraphs.append(paragraph("图4-4 可信比例与 MAE 差值对比。部分组合中存疑样本数较少，因此存疑 MAE 为空；这类结果说明区间较保守，模型倾向于给出更宽的可信覆盖。", align="center"))

    paragraphs.extend(
        [
            paragraph("4.5 主动学习效率", "Heading2"),
            paragraph("主动学习效率以 M9 的学习曲线为核心评价对象，并以随机采样策略 M6 作为基线。ΔRMSE 描述从小样本到终态预算的误差下降幅度，相对改善率反映 M9 相比 M6 的终态优势，AULC 越小说明较少标注预算下获得较低误差。"),
            table_xml(active_table(tables["active_efficiency"])),
            paragraph("表4-5 M9 主动学习效率。"),
        ]
    )
    images.append(figures["active"])
    paragraphs.append(image_xml(f"rId{len(images)}", figures["active"]))
    paragraphs.append(paragraph("图4-5 M9 在三类数据集上的误差下降与相对改善。SPWLA 和正演数据的改善幅度较明显，油田实测数据改善较小但更具工程意义，因为其分布偏移、噪声和标签误差更接近实际生产。", align="center"))

    paragraphs.extend(
        [
            paragraph("4.6 综合分析", "Heading2"),
            paragraph("综合点预测、不确定性和主动学习结果可以看出，正演数据和 SPWLA 数据上的结果更容易表现出较高精度和较稳定的不确定性排序；但这两类数据分别受机理假设和竞赛筛选影响，不能完全代表中国致密砂泥岩储层生产资料。油田实测数据中，孔隙度、含水饱和度和渗透率的误差与不确定性表现更复杂，恰好体现了方法在实际生产环境中的价值：模型不仅给出预测值，还输出置信区间和可信/存疑状态，为人工复核、补充标定和主动学习选样提供依据。"),
            paragraph("因此，本文方法的核心贡献不在于单纯追求理想数据上的最低误差，而在于面向小样本、高噪声、强非均质的实测测井解释场景，形成“预测—不确定性—可信输出—主动补样”的闭环。该闭环能够把不确定性从附属诊断转化为可执行的生产决策信息。"),
        ]
    )
    return "".join(paragraphs), images


def write_docx(body: str, images: list[Path], output: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types(len(images)))
        docx.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/></Relationships>')
        docx.writestr("docProps/core.xml", f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"><dc:title>实验结果与分析</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy><dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified></cp:coreProperties>')
        docx.writestr("docProps/app.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Codex</Application></Properties>')
        docx.writestr("word/document.xml", document_xml(body))
        docx.writestr("word/styles.xml", styles_xml())
        docx.writestr("word/settings.xml", '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:zoom w:percent="100"/></w:settings>')
        docx.writestr("word/_rels/document.xml.rels", relationships(images))
        for image in images:
            docx.write(image, f"word/media/{image.name}")


def main() -> int:
    tables = _load_tables()
    figures = make_plots(tables)
    body, images = build_body(tables, figures)
    write_docx(body, images, OUTPUT)
    print(OUTPUT)
    print(f"figures={len(images)}")
    print(f"tables={5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
