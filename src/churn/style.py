"""Shared visual language for every chart and README figure.

Warm editorial palette (ivory surface, ink text, clay accent) with serif titles and sans labels.
The categorical slots (clay, blue, teal) pass the colour-blind and contrast checks on the ivory
surface. The bundled Source Serif 4 / Source Sans 3 fonts are used when the repo is checked out,
with matplotlib defaults as the fallback.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager  # noqa: E402

LIGHT = {
    "surface": "#faf9f5",
    "card": "#f0eee6",
    "edge": "#e3dfd3",
    "ink": "#141413",
    "ink_2": "#5e5d59",
    "muted": "#8a887f",
    "grid": "#e8e6dc",
    "clay": "#c8623f",
    "blue": "#3f6fa8",
    "teal": "#2a9d8f",
    "clay_soft": "#efd9cc",
}
DARK = {
    "surface": "#1f1e1d",
    "card": "#2a2926",
    "edge": "#3b3a36",
    "ink": "#f5f3ec",
    "ink_2": "#bfbdb4",
    "muted": "#8f8d85",
    "grid": "#34332f",
    "clay": "#d97757",
    "blue": "#6f9bd1",
    "teal": "#4bb5a6",
    "clay_soft": "#4a3228",
}
SERIES = [LIGHT["clay"], LIGHT["blue"], LIGHT["teal"]]

FONT_DIR = Path(__file__).resolve().parents[2] / "docs" / "assets" / "fonts"
SERIF, SANS = "DejaVu Serif", "DejaVu Sans"
if FONT_DIR.exists():
    for ttf in FONT_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(ttf))
    SERIF, SANS = "Source Serif 4 Display", "Source Sans 3"

matplotlib.rcParams.update(
    {
        "font.family": SANS,
        "svg.fonttype": "path",
        "axes.titlepad": 12,
    }
)
