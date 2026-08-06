from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape


def _paragraph_text(paragraph_xml: str) -> str:
    return re.sub(r"<[^>]+>", "", paragraph_xml).strip()


def _paragraph(text: str, template: str) -> str:
    match = re.search(r"(<w:pPr[\s\S]*?</w:pPr>)", template)
    properties = match.group(1) if match else ""
    return f"<w:p>{properties}<w:r><w:t>{escape(text)}</w:t></w:r></w:p>"


def _section_34_texts() -> list[str]:
    return [
        "3.4 可迁移性与泛化性验证",
        "3.4.1 基于 SPWLA 数据集的同域临井迁移测试",
        "本部分旨在检验模型在同一 SPWLA 数据体系内的临井迁移能力。模型使用 SPWLA 数据 Well0-Well7 训练完成后，直接迁移到未参与训练或校准的 Well8 上预测，无需重新训练或微调。",
        "测试流程：取 N=20 条件下训练完成的 M5（本文融合方法，无主动学习）及 M9（完整方法，主动学习后）模型，直接对 Well8 测井数据进行预测。SPWLA 数据采用 CALI、DEN、GR、NEU、log10(RDEP)、log10(RMED) 六条输入曲线，RDEP 与 RMED 不再合并为 RT，因此不再进行实测数据与 SPWLA 数据之间的映射、重采样或对齐。",
        "评估指标：计算点估计 RMSE、PICP（目标95%）、MPIW 以及 UCE。若 PICP 显著低于 95%，说明模型不确定性校准失效；若 MPIW 过宽，说明模型过度保守。同时计算认知不确定性占比，分析临井迁移后不确定性来源的变化。",
        "对比基线：随机猜测（使用训练井标签均值）以及直接在目标井数据上训练的上限模型。",
        "3.4.2 基于实测数据集的区块内部迁移测试",
        "本部分旨在检验模型在同一实测测井资料内部、不同井之间的迁移能力。由于区块4数据井数较多，为兼顾实际运行效率与工程应用场景，本研究不再使用涧字号.csv 作为迁移测试数据。",
        "测试流程：仅使用区块4(筛选).csv。按井号排序后，从前 200 口井中随机选取 20 口井训练模型；迁移验证时，从 200 口井之后随机选取 1 口井作为测试井，模型直接对该井进行预测，无需重新训练或微调。输入采用区块4数据中完整的 GR、CAL、SP、AC、CNL、DEN 与 log10(RT) 七条曲线，目标为 PHIF、SW 与 PERM。",
        "评估指标：计算点估计 RMSE、PICP（目标95%）、MPIW 以及 UCE。若 PICP 显著低于 95%，说明模型不确定性校准失效；若 MPIW 过宽，说明模型过度保守。同时计算认知不确定性占比，分析训练井与迁移测试井之间不确定性来源的变化。",
        "对比基线：随机猜测（使用训练井标签均值）以及直接在目标井数据上训练的上限模型。",
        "3.4.3 基于正演生成数据集的理想条件泛化测试",
        "正演生成数据不含实测噪声，标签严格由岩石物理模型定义。该测试用于评估模型是否过度依赖油田数据中的特定噪声模式或系统偏差。",
        "测试流程：使用油田实测数据训练得到的模型（N=20 和 N=50 两种条件），直接对正演生成数据集进行预测。正演数据需生成与油田数据相同输入输出维度的样本（7输入，3输出），其中电阻率曲线同样先转换为 log10(RT) 后再归一化。",
        "评估指标：除 RMSE、PICP、MPIW 外，重点观察不确定性-误差相关性（高相关性表明模型能正确识别自身错误）。如果模型在正演数据上 PICP 仍接近 95% 且 MPIW 较窄，说明不确定性量化方法具有较好的泛化能力；反之，则说明方法过度拟合了实测数据的噪声分布。",
        "分析：比较 N=20 与 N=50 模型的表现，探讨小样本程度对泛化能力的影响。",
    ]


def main() -> int:
    if len(sys.argv) not in {2, 3}:
        raise SystemExit("usage: update_docx_section_34.py INPUT.docx [OUTPUT.docx]")
    source_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) == 3 else source_path
    if source_path == output_path:
        backup = source_path.with_suffix(".section34.bak.docx")
        shutil.copy2(source_path, backup)
    with TemporaryDirectory() as tmp:
        temp = Path(tmp)
        with ZipFile(source_path) as source:
            source.extractall(temp)
        document = temp / "word" / "document.xml"
        xml = document.read_text(encoding="utf-8")
        paragraphs = re.findall(r"<w:p[\s\S]*?</w:p>", xml)
        start = next(
            index
            for index, paragraph in enumerate(paragraphs)
            if _paragraph_text(paragraph) == "3.4 可迁移性与泛化性验证"
        )
        end = next(
            index
            for index in range(start + 1, len(paragraphs))
            if _paragraph_text(paragraphs[index]).startswith("3.5 ")
        )
        heading_template = paragraphs[start]
        subheading_template = paragraphs[start + 1]
        body_template = paragraphs[start + 2]
        replacement_xml = []
        for text in _section_34_texts():
            if text == "3.4 可迁移性与泛化性验证":
                template = heading_template
            elif text.startswith("3.4."):
                template = subheading_template
            else:
                template = body_template
            replacement_xml.append(_paragraph(text, template))
        document.write_text(
            xml.replace("".join(paragraphs[start:end]), "".join(replacement_xml), 1),
            encoding="utf-8",
        )
        with ZipFile(output_path, "w", ZIP_DEFLATED) as target:
            for item in temp.rglob("*"):
                if item.is_file():
                    target.write(item, item.relative_to(temp).as_posix())
    print(f"updated={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
