from __future__ import annotations

from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIGURE_WIDTH_MM = 180.0
FIGURE_WIDTH_IN = FIGURE_WIDTH_MM / 25.4
MIN_FONT_PT = 7.5
PALETTE = {
    "primary": "#3B5B92",
    "secondary": "#8FA8C2",
    "negative": "#C65D4B",
    "text": "#4D4D4D",
    "light": "#B8B8B8",
    "trusted": "#4F7A5A",
}


def apply_sci_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": PALETTE["text"],
            "axes.labelcolor": PALETTE["text"],
            "xtick.color": PALETTE["text"],
            "ytick.color": PALETTE["text"],
            "text.color": PALETTE["text"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.transparent": False,
            "savefig.facecolor": "white",
        }
    )


def new_figure(rows: int, cols: int, *, height_mm: float, **kwargs):
    apply_sci_style()
    return plt.subplots(
        rows,
        cols,
        figsize=(FIGURE_WIDTH_IN, height_mm / 25.4),
        constrained_layout=True,
        **kwargs,
    )


def add_panel_labels(axes) -> None:
    flat = getattr(axes, "flat", [axes])
    for index, ax in enumerate(flat):
        ax.text(
            -0.12,
            1.04,
            f"({chr(97 + index)})",
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            fontfamily="Times New Roman",
            ha="left",
            va="bottom",
            clip_on=False,
        )


def _intersects(a, b, *, tolerance: float = 1.0) -> bool:
    return not (
        a.x1 <= b.x0 + tolerance
        or b.x1 <= a.x0 + tolerance
        or a.y1 <= b.y0 + tolerance
        or b.y1 <= a.y0 + tolerance
    )


def audit_layout(fig, figure_id: str) -> dict[str, object]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    texts = [text for text in fig.texts if text.get_visible() and text.get_text().strip()]
    boxes = [(text, text.get_window_extent(renderer=renderer)) for text in texts]
    out = [text.get_text() for text, box in boxes if box.x0 < canvas.x0 or box.y0 < canvas.y0 or box.x1 > canvas.x1 or box.y1 > canvas.y1]
    overlaps = [
        (left.get_text(), right.get_text())
        for (left, a), (right, b) in combinations(boxes, 2)
        if _intersects(a, b)
    ]
    font_sizes = [text.get_fontsize() for ax in fig.axes for text in ([ax.xaxis.label, ax.yaxis.label] + list(ax.get_xticklabels()) + list(ax.get_yticklabels())) if text.get_visible() and text.get_text().strip()]
    return {
        "figure_id": figure_id,
        "width_mm": fig.get_figwidth() * 25.4,
        "height_mm": fig.get_figheight() * 25.4,
        "minimum_font_pt": min(font_sizes) if font_sizes else MIN_FONT_PT,
        "out_of_bounds_count": len(out),
        "overlap_count": len(overlaps),
        "out_of_bounds_text": " | ".join(out),
        "overlap_text": " | ".join(f"{a}/{b}" for a, b in overlaps),
    }
