"""Age, HomePlanet, Destination, VIP.

Frames: `train` for every rate; `combined` for distributions.
VIP gets a base-rate check first -- it is a ~2% class and a rate gap on it is
mostly noise until the interval says otherwise.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load import load, SPEND_COLS, TARGET
import viz

AGE_BINS = [0, 5, 12, 17, 25, 35, 50, 65, 200]
AGE_LABELS = ["0-4", "5-12", "13-17", "18-25", "26-35", "36-50", "51-65", "66+"]


def main():
    train, test, combined = load()
    base = train[TARGET].mean()
    print("=" * 70)
    print("05 AGE / HOMEPLANET / DESTINATION / VIP")
    print("=" * 70)
    print(f"marginal transport rate (train) = {base:.4f}")

    # ---- Age (TRAIN-ONLY for rate; COMBINED for distribution) -------------
    print(f"\n[age distribution] frame=combined")
    print(combined["Age"].describe().round(2).to_string())
    print(f"  age==0 rows: {int((combined['Age'] == 0).sum())} "
          f"(infants, not a missing-value sentinel: Age has "
          f"{combined['Age'].isna().sum()} explicit NaNs)")

    t = train.dropna(subset=["Age"]).copy()
    t["AgeBin"] = pd.cut(t["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    print(f"\n[rate by age bin] frame=train")
    ra = viz.rate_table(t, "AgeBin").dropna(subset=["n"])
    ra = ra[ra["n"] > 0]
    print(ra.round(4).to_string())

    kids = t[t["Age"] < 13]
    adults = t[t["Age"] >= 18]
    print(f"  children (<13): rate={kids[TARGET].mean():.4f} n={len(kids)}")
    print(f"  adults  (>=18): rate={adults[TARGET].mean():.4f} n={len(adults)}")
    print(f"  gap = {kids[TARGET].mean() - adults[TARGET].mean():+.4f}")
    # is the child effect just CryoSleep?
    kc = kids[kids["CryoSleep"] == False]
    ac = adults[adults["CryoSleep"] == False]
    print(f"  ...among AWAKE only: children {kc[TARGET].mean():.4f} (n={len(kc)}) "
          f"vs adults {ac[TARGET].mean():.4f} (n={len(ac)}) "
          f"-> gap {kc[TARGET].mean() - ac[TARGET].mean():+.4f}")
    print(f"  children who spend anything: "
          f"{float((kids['TotalSpend'] > 0).mean()):.1%} vs adults "
          f"{float((adults['TotalSpend'] > 0).mean()):.1%}")

    fig, ax = plt.subplots(figsize=(8, 3.9))
    bars = ax.bar(ra.index.astype(str), ra["rate"], color=viz.SERIES[0], width=0.62)
    ax.errorbar(range(len(ra)), ra["rate"],
                yerr=[ra["rate"] - ra["lo"], ra["hi"] - ra["rate"]],
                fmt="none", ecolor=viz.INK_MUTED, elinewidth=1.2, capsize=3)
    viz.label_bars(ax, bars, ra["rate"], pad=0.03)
    for x, n in zip(range(len(ra)), ra["n"]):
        ax.text(x, 0.02, f"n={int(n):,}", ha="center", fontsize=8, color=viz.SURFACE)
    ax.set_ylim(0, 0.95)
    ax.set_xlabel("age band")
    ax.set_ylabel("transport rate")
    viz.clean(ax)
    viz.baseline_ref(ax, base, f"all passengers {base:.1%}")
    viz.save(fig, "05_rate_by_age",
             "Only the under-13s break from the flat adult rate",
             f"train only, n={int(ra['n'].sum()):,} with a known age - bars are 95% Wilson intervals")

    # age histogram by outcome
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i, (val, lab) in enumerate([(True, "transported"), (False, "not transported")]):
        ax.hist(t.loc[t[TARGET] == val, "Age"], bins=np.arange(0, 81, 2),
                histtype="step", linewidth=2, color=viz.SERIES[i], label=lab)
    ax.axvline(13, color=viz.INK_MUTED, linestyle="--", linewidth=1.2)
    ax.text(13.4, ax.get_ylim()[1] * 0.94, " age 13", fontsize=8.5, color=viz.INK_MUTED)
    ax.set_xlabel("age")
    ax.set_ylabel("passengers")
    ax.legend(loc="upper right")
    viz.clean(ax)
    viz.save(fig, "05_age_hist",
             "The two classes separate only at the young end",
             f"train only, n={len(t):,}")

    # ---- HomePlanet + Destination (TRAIN-ONLY) ----------------------------
    print(f"\n[rate by HomePlanet] frame=train")
    rh = viz.rate_table(train, "HomePlanet")
    print(rh.round(4).to_string())

    print(f"\n[rate by Destination] frame=train")
    rd = viz.rate_table(train, "Destination")
    print(rd.round(4).to_string())

    print(f"\n[HomePlanet x Destination] frame=train")
    hd = (train.dropna(subset=["HomePlanet", "Destination"])
          .groupby(["HomePlanet", "Destination"], observed=True)[TARGET]
          .agg(["size", "mean"]).rename(columns={"size": "n", "mean": "rate"}))
    print(hd.round(4).to_string())

    # does Destination survive controlling for HomePlanet?
    print("  destination spread within each home planet:")
    for hp, g in hd.groupby("HomePlanet", observed=True):
        g = g[g["n"] >= 40]
        if len(g) > 1:
            print(f"    {hp}: {g['rate'].min():.3f}..{g['rate'].max():.3f} "
                  f"(spread {g['rate'].max() - g['rate'].min():.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, rr, name in [(axes[0], rh.dropna(subset=["n"]), "home planet"),
                         (axes[1], rd.dropna(subset=["n"]), "destination")]:
        rr = rr[rr.index.notna()] if hasattr(rr.index, "notna") else rr
        bars = ax.bar(rr.index.astype(str), rr["rate"], color=viz.SERIES[0], width=0.55)
        ax.errorbar(range(len(rr)), rr["rate"],
                    yerr=[rr["rate"] - rr["lo"], rr["hi"] - rr["rate"]],
                    fmt="none", ecolor=viz.INK_MUTED, elinewidth=1.2, capsize=3)
        viz.label_bars(ax, bars, rr["rate"], pad=0.025)
        for x, n in zip(range(len(rr)), rr["n"]):
            ax.text(x, 0.02, f"n={int(n):,}", ha="center", fontsize=8, color=viz.SURFACE)
        ax.set_ylim(0, 0.9)
        ax.set_xlabel(name)
        ax.set_ylabel("transport rate")
        viz.clean(ax)
        viz.baseline_ref(ax, base, f"{base:.1%}")
        ax.tick_params(axis="x", labelrotation=8)
    viz.save(fig, "05_rate_by_origin_destination",
             "Europa departs at 66%, Earth at 42%; destination matters less",
             f"train only - bars are 95% Wilson intervals")

    # ---- VIP: base rate FIRST (TRAIN-ONLY) --------------------------------
    print(f"\n[VIP] frame=train")
    print(f"  VIP base rate in combined: "
          f"{float((combined['VIP'] == True).mean()):.4f} "
          f"({int((combined['VIP'] == True).sum())} of {len(combined)})")
    rv = viz.rate_table(train, "VIP")
    print(rv.round(4).to_string())
    if True in rv.index and False in rv.index:
        gap = rv.loc[True, "rate"] - rv.loc[False, "rate"]
        overlap = not (rv.loc[True, "hi"] < rv.loc[False, "lo"]
                       or rv.loc[False, "hi"] < rv.loc[True, "lo"])
        print(f"  VIP - non-VIP gap = {gap:+.4f}")
        print(f"  VIP 95% CI = [{rv.loc[True,'lo']:.4f}, {rv.loc[True,'hi']:.4f}] "
              f"(width {rv.loc[True,'ci_width']:.4f}, n={int(rv.loc[True,'n'])})")
        print(f"  intervals overlap: {overlap} -> the gap is "
              f"{'NOT distinguishable from noise' if overlap else 'real but rests on a tiny class'}")

    # Does the VIP deficit survive conditioning on home planet? VIPs are not
    # spread evenly across planets, so the unconditional gap could be a planet
    # effect in disguise -- or the planet mix could be MASKING a larger one.
    print(f"\n[VIP x HomePlanet] frame=combined for the mix, train for the rates")
    mix = pd.crosstab(combined["VIP"], combined["HomePlanet"], normalize="index")
    print(mix.round(4).to_string())
    print(f"  VIPs by planet: "
          f"{combined.loc[combined['VIP'] == True, 'HomePlanet'].value_counts().to_dict()}")
    print("  -> there is not a single VIP from Earth")
    for hp, sub in train.dropna(subset=["VIP", "HomePlanet"]).groupby("HomePlanet",
                                                                     observed=True):
        rr = viz.rate_table(sub, "VIP")
        if True in rr.index and False in rr.index:
            print(f"    {hp:7s} VIP {rr.loc[True,'rate']:.4f} "
                  f"(n={int(rr.loc[True,'n'])}, CI[{rr.loc[True,'lo']:.3f},"
                  f"{rr.loc[True,'hi']:.3f}])  non-VIP {rr.loc[False,'rate']:.4f} "
                  f"(n={int(rr.loc[False,'n'])})  gap {rr.loc[True,'rate']-rr.loc[False,'rate']:+.4f}")
    print("  -> the within-planet gap is LARGER than the -0.1244 unconditional one:")
    print("     VIPs concentrate on Europa, the highest-rate planet, which was")
    print("     masking the deficit rather than creating it")

    fig, ax = plt.subplots(figsize=(6, 3.3))
    rvv = rv.reindex([False, True])
    bars = ax.bar(["non-VIP", "VIP"], rvv["rate"], color=viz.SERIES[0], width=0.45)
    ax.errorbar(range(2), rvv["rate"],
                yerr=[rvv["rate"] - rvv["lo"], rvv["hi"] - rvv["rate"]],
                fmt="none", ecolor=viz.INK_MUTED, elinewidth=1.6, capsize=5)
    viz.label_bars(ax, bars, rvv["rate"], pad=0.055)
    for x, n in zip(range(2), rvv["n"]):
        ax.text(x, 0.02, f"n={int(n):,}", ha="center", fontsize=8.5, color=viz.SURFACE)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("transport rate")
    viz.clean(ax)
    viz.baseline_ref(ax, base, f"marginal {base:.1%}")
    viz.save(fig, "05_vip",
             "VIP is 2% of the ship -- and the deficit grows once you control for planet",
             f"train only - n={int(rvv.loc[True,'n'])} VIPs; read the interval, not the bar")


if __name__ == "__main__":
    main()
