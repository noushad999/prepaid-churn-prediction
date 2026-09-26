"""Render the README hero diagram (light + dark SVG). Run: python docs/assets/make_overview.py"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

STAGES = [
    ("01", "Operator data", ["subscribers", "monthly usage", "recharge events", "network & market"]),
    ("02", "Snapshots", ["point-in-time features", "60-day silence label", "leakage-tested"]),
    ("03", "Model", ["LightGBM", "out-of-time validated", "calibrated scores"]),
    ("04", "Explain & decide", ["SHAP reason codes", "root-cause driver", "expected-value targeting"]),
    ("05", "Serve & monitor", ["CRM campaign list", "FastAPI, ~5 ms", "PSI drift checks"]),
]

THEMES = {
    "light": dict(bg="#ffffff", card="#f6f8fa", edge="#d0d7de", ink="#1f2328", muted="#59636e", accent="#2a78d6"),
    "dark": dict(bg="#0d1117", card="#161b22", edge="#30363d", ink="#e6edf3", muted="#9198a1", accent="#3987e5"),
}


def render(theme: str, path: Path) -> None:
    t = THEMES[theme]
    fig = plt.figure(figsize=(14, 3.3), facecolor=t["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 3.3)
    ax.axis("off")
    w, h, gap, y0 = 2.35, 2.5, 0.42, 0.4
    x0 = (14 - (len(STAGES) * w + (len(STAGES) - 1) * gap)) / 2
    for i, (num, title, lines) in enumerate(STAGES):
        x = x0 + i * (w + gap)
        highlight = i == 2
        ax.add_patch(
            FancyBboxPatch(
                (x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.14",
                fc=t["card"], ec=t["accent"] if highlight else t["edge"], lw=1.6 if highlight else 1.0,
            )
        )
        ax.text(x + 0.2, y0 + h - 0.32, num, color=t["accent"], fontsize=10, fontweight="bold", va="center")
        ax.text(x + 0.2, y0 + h - 0.72, title, color=t["ink"], fontsize=13.5, fontweight="bold", va="center")
        for j, line in enumerate(lines):
            ax.text(x + 0.2, y0 + h - 1.2 - j * 0.34, line, color=t["muted"], fontsize=10.5, va="center")
        if i < len(STAGES) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + w + 0.06, y0 + h / 2), (x + w + gap - 0.06, y0 + h / 2),
                    arrowstyle="-|>", mutation_scale=12, color=t["muted"], lw=1.2,
                )
            )
    fig.savefig(path, facecolor=t["bg"])
    plt.close(fig)


if __name__ == "__main__":
    out = Path(__file__).parent
    for theme in THEMES:
        render(theme, out / f"overview-{theme}.svg")
