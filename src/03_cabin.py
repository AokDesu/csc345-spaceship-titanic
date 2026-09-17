"""Cabin: deck, number, side -- and whether deck is just HomePlanet in disguise.

Frames: `train` for every rate; `combined` for the deck x HomePlanet contingency
        and the cabin-number geometry (no target involved).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load import load, TARGET
import viz

DECK_ORDER = ["A", "B", "C", "D", "E", "F", "G", "T"]


def cramers_v(ct):
    """Bias-corrected Cramer's V -- how tightly two categoricals track."""
    chi2 = ((ct - np.outer(ct.sum(1), ct.sum(0)) / ct.values.sum()) ** 2
            / (np.outer(ct.sum(1), ct.sum(0)) / ct.values.sum())).values.sum()
    n = ct.values.sum()
    phi2 = chi2 / n
    r, k = ct.shape
    phi2c = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2c / max(1e-12, min(kc - 1, rc - 1)))


def main():
    train, test, combined = load()
    base = train[TARGET].mean()
    print("=" * 70)
    print("03 CABIN: DECK / NUM / SIDE")
    print("=" * 70)
    print(f"marginal transport rate (train) = {base:.4f}")

    # ---- deck x transport rate (TRAIN-ONLY) -------------------------------
    print(f"\n[rate by deck] frame=train")
    rt = viz.rate_table(train, "Deck").reindex(DECK_ORDER).dropna(subset=["n"])
    print(rt.round(4).to_string())

    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    bars = ax.bar(rt.index, rt["rate"], color=viz.SERIES[0], width=0.62)
    ax.errorbar(rt.index, rt["rate"],
                yerr=[rt["rate"] - rt["lo"], rt["hi"] - rt["rate"]],
                fmt="none", ecolor=viz.INK_MUTED, elinewidth=1.2, capsize=3)
    viz.label_bars(ax, bars, rt["rate"], pad=0.04)
    for x, n in zip(range(len(rt)), rt["n"]):
        ax.text(x, 0.02, f"n={int(n):,}", ha="center", fontsize=8,
                color=viz.SURFACE if n > 200 else viz.INK_MUTED,
                rotation=90 if n < 200 else 0)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("deck")
    ax.set_ylabel("transport rate")
    viz.clean(ax)
    viz.baseline_ref(ax, base, f"all passengers {base:.1%}")
    viz.save(fig, "03_rate_by_deck",
             "Deck moves the rate from 36% to 73% -- deck T is 5 rows, ignore it",
             f"train only, n={int(rt['n'].sum()):,} with a known deck - bars are 95% Wilson intervals")

    # ---- is deck a proxy for HomePlanet? (COMBINED for the contingency) ---
    print(f"\n[deck x HomePlanet] frame=combined")
    ct = pd.crosstab(combined["Deck"], combined["HomePlanet"]).reindex(DECK_ORDER).fillna(0)
    print(ct.astype(int).to_string())
    print("\n  row-normalised (P(HomePlanet | deck)):")
    ctn = ct.div(ct.sum(1), axis=0)
    print(ctn.round(3).to_string())
    print(f"\n  Cramer's V (deck, HomePlanet) = {cramers_v(ct):.4f}")
    exclusive = ctn.max(axis=1)
    print(f"  decks that are single-planet (>=99% one planet): "
          f"{exclusive[exclusive >= 0.99].index.tolist()}")
    print(f"  decks that are mixed (<99%): {exclusive[exclusive < 0.99].index.tolist()}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    im = ax.imshow(ctn.values, cmap=viz.SEQ_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ctn.columns)), ctn.columns)
    ax.set_yticks(range(len(ctn.index)), ctn.index)
    ax.set_xlabel("home planet")
    ax.set_ylabel("deck")
    ax.grid(False)
    for i in range(ctn.shape[0]):
        for j in range(ctn.shape[1]):
            v = ctn.values[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                    color=viz.SURFACE if v > 0.55 else viz.INK_SECONDARY)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("share of the deck's passengers", color=viz.INK_SECONDARY, fontsize=9)
    cb.outline.set_visible(False)
    viz.save(fig, "03_deck_by_homeplanet",
             "Deck is largely a home-planet label, not an independent signal",
             f"train+test - Cramer's V = {cramers_v(ct):.2f}; A/B/C are Europa-only, G is Earth-only")

    # ---- does deck survive controlling for HomePlanet? (TRAIN-ONLY) -------
    print(f"\n[deck effect within HomePlanet stratum] frame=train")
    strat = (train.dropna(subset=["Deck", "HomePlanet"])
             .groupby(["HomePlanet", "Deck"], observed=True)[TARGET]
             .agg(["size", "mean"]).rename(columns={"size": "n", "mean": "rate"}))
    strat = strat[strat["n"] >= 30]
    print(strat.round(4).to_string())
    for hp, g in strat.groupby("HomePlanet", observed=True):
        if len(g) > 1:
            print(f"  {hp}: deck rate spread = "
                  f"{g['rate'].max() - g['rate'].min():.4f} "
                  f"({g['rate'].idxmin()[1]}={g['rate'].min():.3f} .. "
                  f"{g['rate'].idxmax()[1]}={g['rate'].max():.3f})")

    piv = strat["rate"].unstack("HomePlanet").reindex(DECK_ORDER).dropna(how="all")
    npiv = strat["n"].unstack("HomePlanet").reindex(piv.index)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    x = np.arange(len(piv))
    w = 0.26
    for i, hp in enumerate(piv.columns):
        off = (i - (len(piv.columns) - 1) / 2) * w
        b = ax.bar(x + off, piv[hp], w, color=viz.SERIES[i], label=hp)
        viz.label_bars(ax, b, piv[hp], pad=0.012)
    ax.set_xticks(x, piv.index)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("deck")
    ax.set_ylabel("transport rate")
    ax.legend(loc="upper left", ncols=3)
    viz.clean(ax)
    viz.baseline_ref(ax, base, f"marginal {base:.1%}")
    viz.save(fig, "03_deck_within_homeplanet",
             "Within a single home planet, deck still moves the rate",
             "train only, cells with n>=30 - so deck is not purely a planet proxy")

    # ---- side (TRAIN-ONLY) ------------------------------------------------
    print(f"\n[rate by side] frame=train")
    rs = viz.rate_table(train, "Side")
    print(rs.round(4).to_string())
    print(f"  S - P gap = {rs.loc['S','rate'] - rs.loc['P','rate']:+.4f}")

    print(f"\n[side x deck rate] frame=train")
    sd = (train.dropna(subset=["Deck", "Side"])
          .groupby(["Deck", "Side"], observed=True)[TARGET]
          .agg(["size", "mean"]).rename(columns={"size": "n", "mean": "rate"}))
    sd = sd[sd["n"] >= 30]
    print(sd.round(4).to_string())

    piv = sd["rate"].unstack("Side").reindex(DECK_ORDER).dropna(how="all")
    fig, ax = plt.subplots(figsize=(7.5, 3.9))
    x = np.arange(len(piv))
    w = 0.34
    for i, s in enumerate(piv.columns):
        b = ax.bar(x + (i - 0.5) * w, piv[s], w, color=viz.SERIES[i],
                   label=f"side {s}" + (" (starboard)" if s == "S" else " (port)"))
        viz.label_bars(ax, b, piv[s], pad=0.012)
    ax.set_xticks(x, piv.index)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("deck")
    ax.set_ylabel("transport rate")
    ax.legend(loc="upper left", ncols=2)
    viz.clean(ax)
    viz.baseline_ref(ax, base, f"marginal {base:.1%}")
    viz.save(fig, "03_side_by_deck",
             "Starboard beats port on every single deck",
             f"train only, cells with n>=30 - overall S {rs.loc['S','rate']:.1%} vs P {rs.loc['P','rate']:.1%}")

    # ---- cabin number geometry (COMBINED) ---------------------------------
    print(f"\n[cabin number geometry] frame=combined")
    cn = combined.dropna(subset=["CabinNum", "Deck"])
    print(cn.groupby("Deck", observed=True)["CabinNum"]
          .agg(["size", "min", "max", "median"]).to_string())

    gg = (combined.dropna(subset=["CabinNum"])
          .groupby("Group", observed=True)["CabinNum"]
          .agg(["size", "nunique", "min", "max"]))
    gg = gg[gg["size"] >= 2]
    span = (gg["max"] - gg["min"])
    shared = gg["nunique"] == 1
    print(f"  multi-member groups with a cabin: {len(gg)}")
    print(f"  ...all in ONE identical cabin: {int(shared.sum())} ({shared.mean():.1%})")
    # Deliberately check the middle case: is the remainder merely ADJACENT?
    print(f"  ...split across 2+ cabins but span <= 2: "
          f"{int(((~shared) & (span <= 2)).sum())}")
    print(f"  ...split, span > 10                   : "
          f"{int(((~shared) & (span > 10)).sum())}")
    print(f"  span of the split groups: median={span[~shared].median():.0f} "
          f"max={span[~shared].max():.0f}")
    decks = combined.dropna(subset=["Deck"]).groupby("Group", observed=True)["Deck"].nunique()
    split_multideck = int((decks.reindex(gg.index[~shared]).dropna() > 1).sum())
    print(f"  split groups that also span >1 DECK: {split_multideck} "
          f"of {int((~shared).sum())}")
    print("  -> there is no 'adjacent cabin' case. A group either shares one")
    print("     identical cabin (2/3) or is scattered across different decks")
    print("     entirely (1/3). CabinNum is a location, not a group id.")

    # rate by cabin-number decile within deck (TRAIN-ONLY)
    print(f"\n[rate by cabin-number decile, per deck] frame=train")
    tr = train.dropna(subset=["CabinNum", "Deck"]).copy()
    tr["NumBin"] = tr.groupby("Deck", observed=True)["CabinNum"].transform(
        lambda s: pd.qcut(s, 10, labels=False, duplicates="drop"))
    nb = (tr.groupby(["Deck", "NumBin"], observed=True)[TARGET]
          .agg(["size", "mean"]).rename(columns={"size": "n", "mean": "rate"}))
    nb = nb[nb["n"] >= 40]
    for deck, g in nb.groupby("Deck", observed=True):
        print(f"  deck {deck}: rate across num-deciles "
              f"{g['rate'].min():.3f}..{g['rate'].max():.3f} "
              f"(spread {g['rate'].max()-g['rate'].min():.3f}, {len(g)} bins)")

    # Small multiples, not six overlaid lines: six series exceeds the palette's
    # validated all-pairs cap, and one panel per deck is easier to read anyway.
    big = [d for d, g in nb.groupby("Deck", observed=True) if len(g) >= 8]
    fig, axes = plt.subplots(1, len(big), figsize=(2.05 * len(big), 3.3),
                             sharey=True)
    for ax, deck in zip(np.atleast_1d(axes), big):
        g = nb.loc[deck]
        ax.plot(g.index, g["rate"], marker="o", markersize=5, linewidth=2,
                color=viz.SERIES[0])
        ax.axhline(base, color=viz.INK_MUTED, linestyle="--", linewidth=1.1, zorder=1)
        ax.set_title(f"deck {deck}", fontsize=10.5, color=viz.INK)
        ax.set_ylim(0, 1.0)
        ax.set_xticks([0, 5, 9], ["bow", "mid", "stern"])
        viz.clean(ax)
    np.atleast_1d(axes)[0].set_ylabel("transport rate")
    fig.supxlabel("cabin-number decile within the deck", y=-0.04,
                  fontsize=9.5, color=viz.INK_SECONDARY)
    viz.save(fig, "03_rate_by_cabinnum",
             "Position along a deck carries its own signal - strongest on deck G",
             "train only, decks with >=8 populated deciles and n>=40 per bin")


if __name__ == "__main__":
    main()
