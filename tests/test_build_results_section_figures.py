from pathlib import Path

from tools import build_results_section_docx as report


def test_make_plots_exports_requested_vector_figures(tmp_path: Path, monkeypatch) -> None:
    """The four report plots are also delivered as vector assets."""
    monkeypatch.setattr(report, "WORK", tmp_path / "work")
    monkeypatch.setattr(report, "FIGURE", tmp_path / "figure", raising=False)

    report.make_plots(report._load_tables())

    expected_stems = {
        "fig_active_learning",
        "fig_point_accuracy",
        "fig_trust_output",
        "fig_uncertainty_quality",
    }
    for stem in expected_stems:
        for suffix in (".pdf", ".svg"):
            exported = tmp_path / "figure" / f"{stem}{suffix}"
            assert exported.exists()
            assert exported.stat().st_size > 0
