"""Overview: missingness, dtypes, cardinality, target balance.

Frames: `combined` for missingness and cardinality (train+test),
        `train` for the target balance. Stated per block below.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load import load, SPEND_COLS, CAT_COLS, TARGET
import viz

RAW_COLS = ["PassengerId", "HomePlanet", "CryoSleep", "Cabin", "Destination",
            "Age", "VIP", *SPEND_COLS, "Name"]


def main():
    train, test, combined = load()
    print("=" * 70)
    print("01 OVERVIEW")
    print("=" * 70)

    # ---- target balance (TRAIN-ONLY) --------------------------------------
    vc = train[TARGET].value_counts(dropna=False)
    rate = train[TARGET].mean()
    print(f"\n[target balance] frame=train  n={len(train)}")
    print(vc.to_string())
    print(f"  base transport rate = {rate:.4f}")
    print(f"  target nulls        = {train[TARGET].isna().sum()}")
    print(f"  -> {'balanced (~50/50)' if 0.45 < rate < 0.55 else 'IMBALANCED'}; "
          f"majority-class baseline accuracy = {max(rate, 1 - rate):.4f}")

    # ---- missingness (COMBINED, plus train/test split) --------------------
    print(f"\n[missingness] frame=combined (train+test, n={len(combined)})")
    miss = pd.DataFrame({
        "train_n": train[RAW_COLS].isna().sum(),
        "train_pct": train[RAW_COLS].isna().mean() * 100,
        "test_n": test[RAW_COLS].isna().sum(),
        "test_pct": test[RAW_COLS].isna().mean() * 100,
        "combined_n": combined[RAW_COLS].isna().sum(),
        "combined_pct": combined[RAW_COLS].isna().mean() * 100,
    }).sort_values("combined_pct", ascending=False)
    print(miss.round(2).to_string())

    complete = miss[miss["combined_n"] == 0].index.tolist()
    print(f"\n  complete columns (zero nulls): {complete}")
    nz = miss[miss["combined_n"] > 0]["combined_pct"]
    print(f"  columns with nulls: {len(nz)}  "
          f"range {nz.min():.2f}%..{nz.max():.2f}%  spread {nz.max()-nz.min():.2f}pp")
    print(f"  -> missingness is {'UNIFORM' if nz.max()-nz.min() < 1.0 else 'NON-UNIFORM'} "
          f"across nullable columns")

    # how many rows are fully complete?
    any_null = combined[RAW_COLS].isna().any(axis=1)
    print(f"  rows with >=1 null: {any_null.sum()} ({any_null.mean():.1%}) "
          f"-- dropping incomplete rows would cost {any_null.mean():.1%} of the data")
    nulls_per_row = combined[RAW_COLS].isna().sum(axis=1)
    print(f"  nulls per row: {nulls_per_row.value_counts().sort_index().to_dict()}")

    # is missingness correlated across columns? (does it clump in bad rows?)
    exp_multi = None
    p = combined[RAW_COLS].isna().mean()
    exp_multi = (1 - np.prod(1 - p)) * len(combined)
    print(f"  rows with >=1 null expected under independence: {exp_multi:.0f} "
          f"vs observed {any_null.sum()} -> missingness is "
          f"{'INDEPENDENT across columns' if abs(exp_multi - any_null.sum()) / any_null.sum() < 0.05 else 'CLUMPED'}")

    # ---- dtypes + cardinality (COMBINED) ----------------------------------
    print(f"\n[dtypes & cardinality] frame=combined")
    card = pd.DataFrame({
        "dtype": combined[RAW_COLS].dtypes.astype(str),
        "nunique": combined[RAW_COLS].nunique(dropna=True),
    })
    print(card.to_string())
    for c in CAT_COLS + ["Destination"]:
        if c in combined:
            print(f"  {c}: {combined[c].value_counts(dropna=False).to_dict()}")

    # ---- figures ----------------------------------------------------------
    print("\n[figures]")

    # missingness bar - single series, no legend, direct labels
    m = miss[miss["combined_n"] > 0].sort_values("combined_pct")
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(m) + 1.6))
    bars = ax.barh(m.index, m["combined_pct"], color=viz.SERIES[0], height=0.62)
    viz.label_bars(ax, bars, m["combined_pct"], fmt="{:.2f}%",
                   horizontal=True, pad=0.03)
    ax.set_xlabel("missing (% of combined rows)")
    ax.set_xlim(0, m["combined_pct"].max() * 1.22)
    viz.clean(ax, which="x")
    viz.save(fig, "01_missingness",
             "Missingness is uniform, low, and spread across almost every column",
             f"train+test, n={len(combined)} - complete: {', '.join(complete)}")

    # nulls-per-row distribution
    npr = nulls_per_row.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.bar(npr.index.astype(str), npr.values, color=viz.SERIES[0], width=0.6)
    viz.label_bars(ax, bars, npr.values, fmt="{:,.0f}",
                   pad=npr.max() * 0.015)
    ax.set_xlabel("number of missing fields in the row")
    ax.set_ylabel("passengers")
    ax.set_ylim(0, npr.max() * 1.15)
    viz.clean(ax)
    viz.save(fig, "01_nulls_per_row",
             "Nulls scatter one-per-row rather than clumping into bad records",
             f"train+test, n={len(combined)}")

    # target balance
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    counts = [int((~train[TARGET]).sum()), int(train[TARGET].sum())]
    bars = ax.barh(["Not transported", "Transported"], counts,
                   color=[viz.SERIES[1], viz.SERIES[0]], height=0.55)
    viz.label_bars(ax, bars, [c / len(train) for c in counts],
                   fmt="{:.1%}", horizontal=True, pad=40)
    ax.set_xlabel("passengers")
    ax.set_xlim(0, max(counts) * 1.2)
    viz.clean(ax, which="x")
    viz.save(fig, "01_target_balance",
             "The target is almost perfectly balanced",
             f"train only, n={len(train)} - rate {rate:.3f}")


if __name__ == "__main__":
    main()
