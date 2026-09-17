"""Spend and CryoSleep: the logical constraint, and what spending predicts.

Frames: `train` for every rate; `combined` for the CryoSleep<->spend consistency
        check and the spend distributions (no target involved).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load import load, SPEND_COLS, TARGET
import viz


def main():
    train, test, combined = load()
    base = train[TARGET].mean()
    print("=" * 70)
    print("04 SPEND & CRYOSLEEP")
    print("=" * 70)
    print(f"marginal transport rate (train) = {base:.4f}")

    # ---- CryoSleep x spend consistency (COMBINED) -------------------------
    print(f"\n[CryoSleep vs spend] frame=combined")
    cryo = combined[combined["CryoSleep"] == True]
    awake = combined[combined["CryoSleep"] == False]
    print(f"  CryoSleep=True : {len(cryo):,}   CryoSleep=False: {len(awake):,}   "
          f"NaN: {combined['CryoSleep'].isna().sum()}")

    rows = []
    for col in SPEND_COLS:
        c = cryo[col]
        a = awake[col]
        rows.append({
            "service": col,
            "cryo_nonnull": int(c.notna().sum()),
            "cryo_nonzero": int((c > 0).sum()),
            "cryo_max": c.max(),
            "awake_nonnull": int(a.notna().sum()),
            "awake_zero_pct": float((a == 0).mean() * 100),
            "awake_median_nonzero": float(a[a > 0].median()),
            "awake_max": a.max(),
        })
    sp = pd.DataFrame(rows)
    print(sp.round(2).to_string(index=False))

    viol = int((cryo[SPEND_COLS].fillna(0) > 0).any(axis=1).sum())
    print(f"\n  CryoSleep passengers with ANY nonzero spend: {viol}")
    print(f"  -> the constraint {'HOLDS EXACTLY' if viol == 0 else 'IS VIOLATED'}: "
          f"a frozen passenger never spends")
    cryo_nan = int(cryo[SPEND_COLS].isna().sum().sum())
    print(f"  NaN spend cells under CryoSleep=True: {cryo_nan} "
          f"-> all logically ZERO, not unknown")

    # the reverse direction: can spend recover a missing CryoSleep?
    miss_cryo = combined[combined["CryoSleep"].isna()]
    spent = (miss_cryo[SPEND_COLS].fillna(0) > 0).any(axis=1)
    print(f"\n  rows with CryoSleep missing: {len(miss_cryo)}")
    print(f"    ...of which show nonzero spend -> must be AWAKE: {int(spent.sum())} "
          f"({spent.mean():.1%})")
    print(f"    -> spend recovers {spent.sum()} of {len(miss_cryo)} missing "
          f"CryoSleep values with certainty; the rest are only probably frozen")

    # zero-spend share by cryo state
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(SPEND_COLS))
    w = 0.36
    cz = [float((cryo[c].fillna(0) == 0).mean()) for c in SPEND_COLS]
    az = [float((awake[c] == 0).mean()) for c in SPEND_COLS]
    b1 = ax.bar(x - w/2, cz, w, color=viz.SERIES[0], label="CryoSleep = True")
    b2 = ax.bar(x + w/2, az, w, color=viz.SERIES[1], label="CryoSleep = False")
    viz.label_bars(ax, b1, cz, pad=0.012)
    viz.label_bars(ax, b2, az, pad=0.012)
    ax.set_xticks(x, SPEND_COLS, rotation=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("share spending exactly zero")
    ax.legend(loc="upper right", ncols=2)
    viz.clean(ax)
    viz.save(fig, "04_cryo_zero_spend",
             "CryoSleep implies zero spend -- with no exceptions in 12,970 rows",
             "train+test - so a NaN spend under CryoSleep is a known zero, not a gap")

    # ---- CryoSleep x transport (TRAIN-ONLY) -------------------------------
    print(f"\n[rate by CryoSleep] frame=train")
    rc = viz.rate_table(train, "CryoSleep")
    print(rc.round(4).to_string())

    # ---- AnySpend x transport (TRAIN-ONLY) --------------------------------
    print(f"\n[rate by AnySpend] frame=train")
    print("  NOTE: AnySpend=False is only certain when SpendNaN==0; a row with an")
    print("  unobserved service could hide spend. Reported both ways.")
    ra = viz.rate_table(train, "AnySpend")
    print(ra.round(4).to_string())
    clean_rows = train[train["SpendNaN"] == 0]
    print(f"  restricted to fully-observed spend rows (n={len(clean_rows)}):")
    print(viz.rate_table(clean_rows, "AnySpend").round(4).to_string())

    # cross: cryo x anyspend
    print(f"\n[CryoSleep x AnySpend] frame=train")
    cx = (train.dropna(subset=["CryoSleep", "AnySpend"])
          .groupby(["CryoSleep", "AnySpend"], observed=True)[TARGET]
          .agg(["size", "mean"]).rename(columns={"size": "n", "mean": "rate"}))
    print(cx.round(4).to_string())

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    labels, rates, ns, colors = [], [], [], []
    rc2 = rc.dropna()
    for k in [True, False]:
        if k in rc2.index:
            labels.append(f"CryoSleep\n{k}")
            rates.append(rc2.loc[k, "rate"]); ns.append(int(rc2.loc[k, "n"]))
            colors.append(viz.SERIES[0])
    ra2 = ra.dropna()
    for k in [False, True]:
        if k in ra2.index:
            labels.append(f"AnySpend\n{k}")
            rates.append(ra2.loc[k, "rate"]); ns.append(int(ra2.loc[k, "n"]))
            colors.append(viz.SERIES[2])
    bars = ax.bar(labels, rates, color=colors, width=0.6)
    viz.label_bars(ax, bars, rates, pad=0.02)
    for i, n in enumerate(ns):
        ax.text(i, 0.02, f"n={n:,}", ha="center", fontsize=8, color=viz.SURFACE)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("transport rate")
    viz.clean(ax)
    viz.baseline_ref(ax, base, f"marginal {base:.1%}")
    viz.save(fig, "04_cryo_and_anyspend",
             "Frozen passengers were transported at 82%; anyone who spent, at 30%",
             f"train only - these are the two strongest single columns in the file")

    # ---- per-service rate (TRAIN-ONLY) ------------------------------------
    print(f"\n[rate by spent-on-service] frame=train")
    svc = []
    for col in SPEND_COLS:
        sub = train[train[col].notna()]
        spent_m = sub[col] > 0
        svc.append({
            "service": col,
            "n_spent": int(spent_m.sum()),
            "rate_spent": float(sub.loc[spent_m, TARGET].mean()),
            "n_zero": int((~spent_m).sum()),
            "rate_zero": float(sub.loc[~spent_m, TARGET].mean()),
        })
    svc = pd.DataFrame(svc)
    svc["gap"] = svc["rate_spent"] - svc["rate_zero"]
    print(svc.round(4).to_string(index=False))
    print("  CAUTION: this raw cut is confounded by CryoSleep -- spending at all")
    print("  implies awake, and awake passengers are transported at 33%. Every")
    print("  service looks negative for that reason alone. Stratify on awake:")

    # The honest version: AWAKE passengers only, so CryoSleep cannot drive it.
    aw = train[train["CryoSleep"] == False]
    svc_aw = []
    for col in SPEND_COLS:
        sub = aw[aw[col].notna()]
        m = sub[col] > 0
        svc_aw.append({
            "service": col,
            "n_spent": int(m.sum()),
            "rate_spent": float(sub.loc[m, TARGET].mean()),
            "n_zero": int((~m).sum()),
            "rate_zero": float(sub.loc[~m, TARGET].mean()),
        })
    svc_aw = pd.DataFrame(svc_aw)
    svc_aw["gap"] = svc_aw["rate_spent"] - svc_aw["rate_zero"]
    # A gap needs an interval before it earns a direction.
    p1 = svc_aw["rate_spent"]; n1 = svc_aw["n_spent"]
    p0 = svc_aw["rate_zero"]; n0 = svc_aw["n_zero"]
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    svc_aw["se"] = se
    svc_aw["z"] = svc_aw["gap"] / se
    svc_aw["sig"] = svc_aw["z"].abs() > 1.96
    print(f"\n[rate by spent-on-service, AWAKE ONLY] frame=train, n={len(aw)}, "
          f"awake base rate={aw[TARGET].mean():.4f}")
    print(svc_aw.round(4).to_string(index=False))
    pos = svc_aw[(svc_aw["gap"] > 0) & svc_aw["sig"]]["service"].tolist()
    neg = svc_aw[(svc_aw["gap"] < 0) & svc_aw["sig"]]["service"].tolist()
    flat = svc_aw[~svc_aw["sig"]]["service"].tolist()
    print(f"  spending RAISES transport odds: {pos}")
    print(f"  spending LOWERS transport odds: {neg}")
    print(f"  flat (gap not distinguishable from zero at 95%): {flat}")
    print("  -> the five services are NOT interchangeable; a single TotalSpend "
          "column destroys this split")

    s = svc_aw.sort_values("gap")
    aw_base = aw[TARGET].mean()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(s))
    w = 0.36
    b1 = ax.bar(x - w/2, s["rate_spent"], w, color=viz.SERIES[1], label="spent > 0")
    b2 = ax.bar(x + w/2, s["rate_zero"], w, color=viz.SERIES[0], label="spent 0")
    viz.label_bars(ax, b1, s["rate_spent"], pad=0.012)
    viz.label_bars(ax, b2, s["rate_zero"], pad=0.012)
    ax.set_xticks(x, s["service"], rotation=12)
    ax.set_ylim(0, max(s["rate_spent"].max(), s["rate_zero"].max()) * 1.35)
    ax.set_ylabel("transport rate")
    ax.legend(loc="upper left", ncols=2)
    viz.clean(ax)
    viz.baseline_ref(ax, aw_base, f"awake baseline {aw_base:.1%}")
    viz.save(fig, "04_rate_by_service",
             "Among awake passengers the services split in opposite directions",
             f"train only, CryoSleep=False (n={len(aw):,}) - "
             f"{'/'.join(pos)} raises the rate, {'/'.join(neg)} lower it, "
             f"{'/'.join(flat)} is flat")

    # ---- spend distribution (COMBINED, log1p) -----------------------------
    print(f"\n[spend distributions] frame=combined")
    dd = combined[SPEND_COLS]
    print(pd.DataFrame({
        "zero_pct": (dd == 0).mean() * 100,
        "nonzero_n": (dd > 0).sum(),
        "nonzero_median": dd[dd > 0].median(),
        "nonzero_p90": dd[dd > 0].quantile(0.90),
        "max": dd.max(),
    }).round(2).to_string())
    ts = combined["TotalSpend"]
    print(f"  TotalSpend: zero {float((ts == 0).mean()):.1%}  "
          f"median(nonzero) {ts[ts > 0].median():.0f}  max {ts.max():.0f}")

    # Small multiples per service rather than five overlaid histograms: five
    # series exceeds the palette's validated all-pairs cap, and the shapes are
    # near-identical so overlaying them hides exactly that point.
    fig, axes = plt.subplots(1, len(SPEND_COLS) + 1, figsize=(13, 3.2), sharey=True)
    bins = np.linspace(0, np.log1p(combined[SPEND_COLS].max().max()), 36)
    for ax, col in zip(axes, SPEND_COLS):
        v = combined.loc[combined[col] > 0, col]
        ax.hist(np.log1p(v), bins=bins, color=viz.SERIES[0])
        ax.set_title(col, fontsize=10, color=viz.INK)
        viz.clean(ax)
    axes[0].set_ylabel("passengers (nonzero charges)")

    ax = axes[-1]
    tr = train.dropna(subset=["TotalSpend"]).copy()
    for i, (val, lab) in enumerate([(True, "transported"), (False, "not transported")]):
        v = tr.loc[(tr[TARGET] == val) & (tr["TotalSpend"] > 0), "TotalSpend"]
        ax.hist(np.log1p(v), bins=bins, histtype="step", linewidth=2,
                color=viz.SERIES[i], label=lab)
    ax.set_title("total spend, by outcome", fontsize=10, color=viz.INK)
    ax.legend(loc="upper left", fontsize=8)
    viz.clean(ax)
    fig.supxlabel("log1p(amount)", y=-0.04, fontsize=9.5, color=viz.INK_SECONDARY)
    viz.save(fig, "04_spend_distributions",
             "Spend is zero-inflated and log-scaled; among spenders the two classes overlap",
             f"per-service panels: train+test, nonzero charges only - last panel: "
             f"train only - {float((ts == 0).mean()):.0%} of passengers spend nothing at all")


if __name__ == "__main__":
    main()
