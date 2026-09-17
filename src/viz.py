"""Shared matplotlib styling + chart helpers.

Light-mode PNG output using the validated default dataviz palette.
Roles, not raw hex, at the call sites.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# --- palette (light mode) ---------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# categorical slots, fixed order, never cycled
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# sequential blue ramp, light -> dark
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]

DIVERGING = ("#2a78d6", "#f0efec", "#d03b3b")  # blue <- neutral -> red

SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
DIV_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("divbr", DIVERGING)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.labelcolor": INK_SECONDARY,
    "axes.labelsize": 10,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.labelcolor": INK_SECONDARY,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
})


def clean(ax, which="y"):
    """Recessive chrome: drop top/right spines, grid on the value axis only."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=which, color=GRID, linewidth=0.8)
    ax.grid(axis="x" if which == "y" else "y", visible=False)
    return ax


def save(fig, name, title=None, subtitle=None):
    """Save to figures/<name>.png with an optional title block.

    The title block is placed ABOVE the figure box in constant point offsets
    (savefig bbox="tight" captures it), so it never lands on the axes no matter
    the figure height.
    """
    h = fig.get_size_inches()[1]
    if title:
        fig.text(0.0, 1 + 0.34 / h, title, ha="left", va="bottom",
                 fontsize=13, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.0, 1 + 0.10 / h, subtitle, ha="left", va="bottom",
                 fontsize=9.5, color=INK_SECONDARY)
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> figures/{name}.png")
    return path


def label_bars(ax, bars, values, fmt="{:.1%}", horizontal=False, pad=0.006):
    """Direct-label each bar. Text wears ink tokens, never the series color."""
    for bar, val in zip(bars, values):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if horizontal:
            ax.text(bar.get_width() + pad, bar.get_y() + bar.get_height() / 2,
                    fmt.format(val), va="center", ha="left",
                    fontsize=9, color=INK_SECONDARY)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + pad,
                    fmt.format(val), ha="center", va="bottom",
                    fontsize=9, color=INK_SECONDARY)


def baseline_ref(ax, value, label, horizontal=False):
    """Draw the marginal-rate reference line that rate charts are read against.

    Labelled at the axis edge in axes-fraction coordinates so it cannot collide
    with a direct bar label.
    """
    if horizontal:
        ax.axvline(value, color=INK_MUTED, linestyle="--", linewidth=1.2, zorder=1)
        ax.text(value, 1.005, f" {label}", transform=ax.get_xaxis_transform(),
                va="bottom", ha="left", fontsize=8.5, color=INK_MUTED)
    else:
        ax.axhline(value, color=INK_MUTED, linestyle="--", linewidth=1.2, zorder=1)
        ax.text(1.006, value, f" {label}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=8.5, color=INK_MUTED)


def wilson(k, n, z=1.96):
    """Wilson score interval - honest error bars on a proportion.

    Needed because several cuts here (VIP, deck T, large groups) have tiny n,
    where a raw rate difference is mostly noise.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        lo, hi = centre - half, centre + half
    return np.where(n > 0, lo, np.nan), np.where(n > 0, hi, np.nan)


def rate_table(df, by, target="Transported", min_n=0):
    """Transport rate + count + Wilson CI per level of `by`. TRAIN-ONLY frame."""
    import pandas as pd
    g = df.groupby(by, dropna=False, observed=True)[target]
    out = pd.DataFrame({"n": g.size(), "k": g.sum()})
    out["rate"] = out["k"] / out["n"]
    lo, hi = wilson(out["k"].to_numpy(), out["n"].to_numpy())
    out["lo"], out["hi"] = lo, hi
    out["ci_width"] = out["hi"] - out["lo"]
    return out[out["n"] >= min_n]
