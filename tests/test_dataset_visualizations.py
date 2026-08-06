from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).parents[1] / "tools" / "build_dataset_visualizations.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_dataset_visualizations", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_representative_group_uses_median_length_not_longest():
    module = _module()
    frame = pd.DataFrame({"well": ["A"] * 2 + ["B"] * 4 + ["C"] * 7})
    selected, anonymous_id, length = module.select_representative_group(frame, "well")
    assert selected == "B"
    assert anonymous_id == "Group-002"
    assert length == 4


def test_robust_zscore_centers_median_and_handles_constant_column():
    module = _module()
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 100.0], "constant": [5.0] * 4})
    result = module.robust_zscore(frame, ["x", "constant"])
    assert np.median(result["x"]) == 0.0
    assert result["constant"].eq(0.0).all()


def test_fixed_sample_respects_limit_and_track_order():
    module = _module()
    frame = pd.DataFrame({"depth": np.arange(1000), "value": np.arange(1000)})
    sample = module.fixed_sample(frame, 100, seed=20260707)
    track = module.ordered_track_sample(frame, "depth", 60)
    assert len(sample) == 100
    assert len(track) == 60
    assert track["depth"].is_monotonic_increasing


def test_panel_label_is_anchored_to_axes_not_pre_layout_figure_coordinates():
    module = _module()
    fig, ax = module.plt.subplots()
    module._panel_label(fig, ax, "(a)")
    assert not fig.texts
    assert ax.texts[-1].get_transform() == ax.transAxes
    assert ax.texts[-1].get_position() == (-0.13, 1.02)


def test_missingness_axis_limit_is_nonnegative_for_complete_data():
    module = _module()
    assert module.missingness_axis_limit(pd.Series([0.0, 0.0])) == (0.0, 5.0)


def test_crossplot_specs_follow_available_dataset_fields():
    module = _module()
    assert module.crossplot_specs("field") == [
        ("DEN", "PHIF", "DEN", "PHIF"),
        ("RT", "SW", "log10(RT)", "SW"),
    ]
    assert module.crossplot_specs("spwla") == [
        ("DEN", "PHIF", "DEN", "PHIF"),
        ("GR", "VSH", "GR", "VSH"),
        ("RDEP", "SW", "log10(RDEP)", "SW"),
    ]
    assert module.crossplot_specs("forward")[1] == ("GR", "VSH", "GR", "VSH")


def test_individual_target_files_cover_each_dataset_target_once():
    module = _module()
    specs = module.individual_target_specs()
    assert sum(len(targets) for targets in specs.values()) == 9
    assert specs["field"] == ["PHIF", "SW", "PERM"]
    assert specs["spwla"] == ["PHIF", "SW", "VSH"]
    assert specs["forward"] == ["PHIF", "SW", "VSH"]
