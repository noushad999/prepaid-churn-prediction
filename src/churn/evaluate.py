"""Metrics that retention teams actually use, plus report charts.

Accuracy is meaningless at ~4% churn (predicting "nobody churns" scores 96%). What matters:
ranking quality (ROC-AUC, KS), precision on the rare class (PR-AUC), and how many churners a
campaign of fixed size reaches (lift and capture rate in the top deciles).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# validated categorical slots 1-3 (blue, orange, aqua) + recessive neutrals
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def top_share_stats(y: np.ndarray, p: np.ndarray, share: float) -> tuple[float, float, float]:
    """(precision, capture rate, lift) when targeting the top `share` of the base."""
    k = max(1, int(round(share * len(y))))
    top = y[np.argsort(-p, kind="stable")[:k]]
    precision = top.mean()
    return precision, top.sum() / max(y.sum(), 1), precision / y.mean()


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    fpr, tpr, _ = roc_curve(y, p)
    out = {
        "n": int(len(y)),
        "churn_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "ks": float(np.max(tpr - fpr)),
    }
    if p.min() >= 0 and p.max() <= 1:
        out["brier"] = float(brier_score_loss(y, p))
        out["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    for share in (0.05, 0.10, 0.20):
        prec, cap, lift = top_share_stats(y, p, share)
        pct = int(share * 100)
        out[f"precision_top{pct}"] = float(prec)
        out[f"capture_top{pct}"] = float(cap)
        out[f"lift_top{pct}"] = float(lift)
    return out


def gains_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    df = (
        pd.DataFrame({"y": y, "p": p}).sort_values("p", ascending=False, kind="stable").reset_index(drop=True)
    )
    df["decile"] = np.arange(len(df)) * bins // len(df) + 1
    g = df.groupby("decile").agg(subscribers=("y", "size"), churners=("y", "sum"), avg_score=("p", "mean"))
    g["churn_rate"] = g["churners"] / g["subscribers"]
    g["lift"] = g["churn_rate"] / df["y"].mean()
    g["cum_capture"] = g["churners"].cumsum() / df["y"].sum()
    return g.reset_index()


# ---------------------------------------------------------------------------- charts
def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _figure(ncols: int = 1, width: float = 6.4, height: float = 4.2):
    fig, axes = plt.subplots(1, ncols, figsize=(width * ncols, height), facecolor=SURFACE)
    return fig, np.atleast_1d(axes)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_roc_pr(y: np.ndarray, scores: dict[str, np.ndarray], path: Path) -> None:
    fig, (a1, a2) = _figure(2)
    for (name, p), color in zip(scores.items(), SERIES, strict=False):
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC {roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        a2.plot(rec, prec, color=color, lw=2, label=f"{name}  (AP {average_precision_score(y, p):.3f})")
    a1.plot([0, 1], [0, 1], color=INK_2, lw=1, ls="--")
    a2.axhline(y.mean(), color=INK_2, lw=1, ls="--")
    _style(a1, "ROC curve (out-of-time test month)", "False positive rate", "True positive rate")
    _style(a2, "Precision-recall", "Recall (share of churners caught)", "Precision")
    for a in (a1, a2):
        a.legend(frameon=False, fontsize=9, labelcolor=INK)
    _save(fig, path)


def plot_gains(y: np.ndarray, scores: dict[str, np.ndarray], path: Path) -> None:
    fig, (ax,) = _figure()
    for (name, p), color in zip(scores.items(), SERIES, strict=False):
        cum = np.concatenate([[0], np.cumsum(y[np.argsort(-p, kind="stable")])]) / y.sum()
        idx = np.linspace(0, len(y), 1000).astype(int)
        ax.plot(idx / len(y) * 100, cum[idx] * 100, color=color, lw=2, label=name)
    ax.plot([0, 100], [0, 100], color=INK_2, lw=1, ls="--", label="Random targeting")
    ax.axvline(10, color=GRID, lw=1)
    _style(
        ax,
        "Cumulative gains: churners reached vs campaign size",
        "% of base targeted (by risk)",
        "% of churners reached",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    _save(fig, path)


def plot_calibration(y: np.ndarray, p: np.ndarray, path: Path, bins: int = 20) -> None:
    fig, (ax,) = _figure(width=5.2)
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")
    g = df.groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean"))
    lim = max(g["pred"].max(), g["obs"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color=INK_2, lw=1, ls="--", label="Perfect calibration")
    ax.plot(g["pred"], g["obs"], color=SERIES[0], lw=2, marker="o", ms=5, label="LightGBM (ventiles)")
    _style(ax, "Calibration", "Predicted churn probability", "Observed churn rate")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    _save(fig, path)


def plot_driver_importance(importance: dict[str, float], labels: dict[str, str], path: Path) -> None:
    items = list(importance.items())[::-1]
    fig, (ax,) = _figure(width=7.5, height=4.0)
    ax.barh(
        [labels.get(k, k).split(":")[0] for k, _ in items], [v for _, v in items], color=SERIES[0], height=0.6
    )
    for i, (_, v) in enumerate(items):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=9, color=INK_2)
    _style(ax, "Churn drivers (mean |SHAP| per family)", "Mean absolute contribution (log-odds)", "")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def plot_profit(
    curves: dict[str, np.ndarray], policy_k: int, n: int, currency: str, path: Path, max_share: float = 0.15
) -> None:
    fig, (ax,) = _figure(width=7.0)
    k_max = int(max_share * n)
    for (name, c), color in zip(curves.items(), SERIES, strict=False):
        ax.plot(np.arange(k_max + 1) / n * 100, c[: k_max + 1] / 1e3, color=color, lw=2, label=name)
    model_curve = next(iter(curves.values()))
    px, py = policy_k / n * 100, model_curve[policy_k] / 1e3
    ax.scatter([px], [py], s=60, color=SERIES[0], edgecolor=SURFACE, lw=2, zorder=5)
    ax.annotate(
        f"Model policy: target {policy_k:,} subs ({policy_k / n:.1%} of base)\nnet +{py:,.0f}k {currency}",
        (px, py),
        xytext=(0.55, 0.93),
        textcoords="axes fraction",
        va="top",
        fontsize=9,
        color=INK,
        arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.8),
    )
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xlim(0, max_share * 100)
    _style(
        ax,
        "Retention campaign profit vs targeting depth",
        "% of base targeted",
        f"Net profit (thousand {currency})",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower left")
    _save(fig, path)
