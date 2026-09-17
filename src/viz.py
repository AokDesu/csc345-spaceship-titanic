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


def cluster_bootstrap(df, cluster, statistic, reps=2000, seed=0, alpha=0.05):
    """Percentile bootstrap interval for a statistic whose unit is not the row.

    `wilson()` above is the right tool for a proportion over independent
    observations. This is the tool for the case where the observations are
    *not* independent because they share a cluster -- and finding 1 is exactly
    that case. Its 3,323 cabin "pairs" come from 3,067 passengers, each of whom
    appears in several pairs, so treating the pairs as 3,323 independent
    observations (or attaching a textbook standard error to them) reports an
    interval far narrower than the data earns. Resampling whole clusters keeps
    the dependence intact.

    df         frame whose rows are observations
    cluster    column naming the independent unit to resample (e.g. "Group")
    statistic  callable(frame) -> float, or -> dict of named floats
    reps, seed fixed by default, so the interval is reproducible run to run

    `statistic` is handed a resampled copy of `df` whose `cluster` column has
    been REPLACED by a fresh integer id per draw. A cluster drawn twice
    therefore arrives under two ids rather than collapsing back into one --
    that collapse is the classic cluster-bootstrap bug and it silently narrows
    the interval. Any key nested inside the cluster (`Cabin` inside `Group`)
    must be paired with that id by the statistic, for the same reason.

    Returns a DataFrame indexed by statistic name with columns point, lo, hi,
    se, and `reps` / `seed` / `alpha` / `clusters` recorded in `.attrs`.
    """
    import pandas as pd

    rng = np.random.default_rng(seed)
    sub = df[df[cluster].notna()]
    order = np.argsort(sub[cluster].to_numpy(), kind="stable")
    counts = sub.groupby(cluster, observed=True).size().sort_index().to_numpy()
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    n_clusters = len(counts)

    def _as_dict(value):
        return value if isinstance(value, dict) else {"statistic": float(value)}

    point = _as_dict(statistic(df))
    draws = {name: np.empty(reps) for name in point}

    for r in range(reps):
        pick = rng.integers(0, n_clusters, n_clusters)
        lens = counts[pick]
        # Ragged gather, vectorised: offset each drawn block into a flat range.
        flat = np.arange(lens.sum()) + np.repeat(
            starts[pick] - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)
        rep = sub.iloc[order[flat]].copy()
        rep[cluster] = np.repeat(np.arange(n_clusters), lens)  # fresh id per DRAW
        for name, value in _as_dict(statistic(rep)).items():
            draws[name][r] = value

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = pd.DataFrame({
        "point": pd.Series(point),
        "lo": {k: np.percentile(v, lo_q) for k, v in draws.items()},
        "hi": {k: np.percentile(v, hi_q) for k, v in draws.items()},
        "se": {k: v.std(ddof=1) for k, v in draws.items()},
        "p_positive": {k: float((v > 0).mean()) for k, v in draws.items()},
    })
    out.attrs.update(reps=reps, seed=seed, alpha=alpha, clusters=n_clusters,
                     draws=draws)
    return out


def _selftest():
    """Behavioural checks for cluster_bootstrap. Run: python src/viz.py

    No test runner is configured in this project, so these live here rather
    than pulling in test infrastructure the analysis does not otherwise need.
    """
    import pandas as pd

    # 20 clusters of 50 identical rows. Every row in a cluster carries the same
    # value, so the effective sample size is 20, not 1,000. A row-level
    # bootstrap would report an SE ~sqrt(50) too small; clustering must not.
    rng = np.random.default_rng(7)
    values = rng.normal(size=20)
    df = pd.DataFrame({"unit": np.repeat(np.arange(20), 50),
                       "x": np.repeat(values, 50)})
    out = cluster_bootstrap(df, "unit", lambda d: d["x"].mean(),
                            reps=1500, seed=1)
    se_cluster = values.std(ddof=1) / np.sqrt(20)
    se_rows = df["x"].std(ddof=1) / np.sqrt(len(df))
    got = out.loc["statistic", "se"]
    assert abs(got - se_cluster) < 0.25 * se_cluster, \
        f"clustered se {got:.4f} should track {se_cluster:.4f}"
    assert got > 3 * se_rows, \
        f"clustered se {got:.4f} must exceed the row-level {se_rows:.4f}"

    # Each DRAW must get its own id, so a cluster drawn twice counts twice.
    # Without the relabelling a groupby collapses the duplicate and the
    # interval comes out too narrow.
    seen = cluster_bootstrap(df, "unit", lambda d: d["unit"].nunique(),
                             reps=50, seed=2)
    assert seen.loc["statistic", "point"] == 20
    assert seen.attrs["draws"]["statistic"].min() == 20, \
        "duplicated clusters collapsed instead of being relabelled"

    # Reproducible: same seed, same interval.
    a = cluster_bootstrap(df, "unit", lambda d: d["x"].mean(), reps=200, seed=3)
    b = cluster_bootstrap(df, "unit", lambda d: d["x"].mean(), reps=200, seed=3)
    assert a["lo"].equals(b["lo"]) and a["hi"].equals(b["hi"])

    print("viz._selftest: cluster_bootstrap OK "
          f"(clustered se={got:.4f} vs row-level {se_rows:.4f})")


if __name__ == "__main__":
    _selftest()
