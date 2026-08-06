from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import warnings


MODULE_PATH = Path(__file__).parents[1] / "tools" / "build_manuscript_figures.py"
DATA_MODULE_PATH = Path(__file__).parents[1] / "tools" / "manuscript_figure_data.py"
STYLE_MODULE_PATH = Path(__file__).parents[1] / "src" / "bnn_inversion" / "publication_style.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_manuscript_figures", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_paired_rmse_improvement_uses_baseline_minus_candidate():
    module = _load_module()
    frame = pd.DataFrame(
        {
            "baseline_rmse": [2.0, 4.0],
            "candidate_rmse": [1.5, 5.0],
        }
    )
    result = module.paired_rmse_improvement(
        frame, baseline="baseline_rmse", candidate="candidate_rmse"
    )
    assert result.tolist() == pytest.approx([25.0, -25.0])


def test_validate_seed_coverage_rejects_missing_seed():
    module = _load_module()
    frame = pd.DataFrame(
        {
            "dataset": ["field"] * 4,
            "target": ["PERM"] * 4,
            "budget_N": [100] * 4,
            "seed": [0, 1, 2, 4],
        }
    )
    with pytest.raises(ValueError, match="missing seeds"):
        module.validate_seed_coverage(
            frame, group_columns=["dataset", "target", "budget_N"]
        )


def test_select_median_seed_does_not_select_best_rmse():
    module = _load_module()
    frame = pd.DataFrame(
        {
            "seed": [0, 1, 2, 3, 4],
            "rmse": [0.10, 0.20, 0.30, 0.40, 0.90],
        }
    )
    assert module.select_median_seed(frame) == 2


def test_expected_figure_contract_has_twelve_vector_pairs():
    module = _load_module()
    artifacts = module.expected_artifacts(Path("figure"))
    assert len(artifacts) == 12
    assert len({item.figure_id for item in artifacts}) == 12
    assert all(item.pdf_path.suffix == ".pdf" for item in artifacts)
    assert all(item.svg_path.suffix == ".svg" for item in artifacts)
    assert not any("smoke" in str(path).lower() for item in artifacts for path in (item.pdf_path, item.svg_path))


def test_budget_columns_distinguish_active_initial_and_final_budget():
    module = _load_path("manuscript_figure_data", DATA_MODULE_PATH)
    frame = pd.DataFrame(
        {
            "section": ["active", "active", "backbone"],
            "method": ["M9", "M5", "M5"],
            "budget_N": [200, 200, 500],
        }
    )
    result = module.add_budget_columns(frame)
    assert result["initial_budget"].tolist() == [100, 200, 500]
    assert result["final_budget"].tolist() == [200, 200, 500]


def test_publication_style_uses_180_mm_width_and_restrained_palette():
    module = _load_path("publication_style", STYLE_MODULE_PATH)
    assert module.FIGURE_WIDTH_IN == pytest.approx(180 / 25.4)
    assert set(module.PALETTE.values()) <= {
        "#3B5B92", "#8FA8C2", "#C65D4B", "#4D4D4D", "#B8B8B8", "#4F7A5A"
    }


def test_layout_audit_detects_overlapping_figure_text():
    module = _load_path("publication_style", STYLE_MODULE_PATH)
    fig, _ = module.new_figure(1, 1, height_mm=80)
    fig.text(0.5, 0.5, "overlap", fontsize=8)
    fig.text(0.5, 0.5, "overlap", fontsize=8)
    audit = module.audit_layout(fig, "X")
    assert audit["overlap_count"] >= 1


def test_publication_style_renders_chinese_without_missing_glyph_warning():
    module = _load_path("publication_style", STYLE_MODULE_PATH)
    fig, ax = module.new_figure(1, 1, height_mm=80)
    ax.set_xlabel("终态标注预算 RMSE")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fig.canvas.draw()
    assert not [item for item in caught if "Glyph" in str(item.message)]
