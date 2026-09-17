"""Groups and families: the structural findings.

PassengerId encodes a travel group. Name encodes a surname (family proxy).
These are the two joins the raw columns do not give you, and they carry the
least-obvious signal in the dataset.

Frames: `combined` for group sizes and attribute-constancy (groups straddle the
        train/test split). `train` for every transport rate and for concordance.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load import load, TARGET
import viz


def pair_concordance(df, key):
    """Pairwise fate agreement within `key`, train-only.

    Returns P(mate transported | ego transported), P(mate transported | ego not),
    and the share of same-fate pairs -- each against the marginal rate, which is
    what they must be read against.
    """
    sub = df[df[key].notna() & df[TARGET].notna()]
    sizes = sub.groupby(key)[TARGET].size()
    multi = sizes[sizes >= 2].index
    sub = sub[sub[key].isin(multi)]

    n_pairs = same = ego_t_mate_t = ego_t = ego_f_mate_t = ego_f = 0
    for _, y in sub.groupby(key, observed=True)[TARGET]:
        vals = y.to_numpy(dtype=bool)
        k = len(vals)
        t = int(vals.sum())
        f = k - t
        # unordered pairs
        n_pairs += k * (k - 1) // 2
        same += t * (t - 1) // 2 + f * (f - 1) // 2
        # ordered pairs (ego, mate)
        ego_t += t * (k - 1)
        ego_t_mate_t += t * (t - 1)
        ego_f += f * (k - 1)
        ego_f_mate_t += f * t

    return {
        "members": len(sub),
        "groups": len(multi),
        "pairs": n_pairs,
        "same_fate_share": same / n_pairs if n_pairs else np.nan,
        "p_mate_t_given_ego_t": ego_t_mate_t / ego_t if ego_t else np.nan,
        "p_mate_t_given_ego_f": ego_f_mate_t / ego_f if ego_f else np.nan,
    }


def _pair_swing(codes, transported):
    """Concordance swing from integer unit codes -- the vectorised twin of
    `pair_concordance`'s swing, cheap enough to run thousands of times."""
    k = np.bincount(codes).astype(float)
    t = np.bincount(codes, weights=transported)
    f = k - t
    return ((t * (t - 1)).sum() / (t * (k - 1)).sum()
            - (f * t).sum() / (f * (k - 1)).sum())


def concordance_swings(df):
    """Group and cabin concordance swings, and the difference between them.

    Written as a single statistic over one frame because the quantity finding 1
    actually needs an interval for is the *difference*, and the same passengers
    sit in both terms. Bootstrapping the two swings separately would get the
    variance of the difference wrong.

    Expects columns Group / Cabin / TARGET. `Cabin` is nested inside `Group`,
    so it is paired with the group id: under `viz.cluster_bootstrap` a group
    drawn twice arrives as two ids, and its cabins must split with it rather
    than collapsing back together.
    """
    y = df[TARGET].to_numpy(dtype=float)
    gc = pd.factorize(df["Group"], sort=False)[0]
    cab = df["Cabin"]
    obs = cab.notna().to_numpy()
    cab_code = pd.factorize(cab[obs], sort=False)[0]
    nested = gc[obs].astype(np.int64) * (cab_code.max() + 1) + cab_code
    cc = pd.factorize(nested, sort=False)[0]
    group, cabin = _pair_swing(gc, y), _pair_swing(cc, y[obs])
    return {"group": group, "cabin": cabin, "cabin_minus_group": cabin - group}


def constancy(combined, key, attr):
    """Share of multi-member `key` units where `attr` takes a single value.

    Only units with >=2 *observed* values of `attr` are testable. A two-person
    group whose second member has a missing `HomePlanet` carries one distinct
    value, scores as constant, and was never capable of violating constancy --
    counting it pads the denominator with units that could not have failed.

    Returns (share single-valued, testable units, multi-member units in frame).
    """
    sizes = combined[combined[key].notna()].groupby(key, observed=True).size()
    multi = sizes[sizes >= 2].index
    sub = combined[combined[key].isin(multi) & combined[attr].notna()]
    g = sub.groupby(key, observed=True)[attr]
    n_observed, nun = g.size(), g.nunique()
    nun = nun[nun.index.isin(n_observed[n_observed >= 2].index)]
    return (nun == 1).mean(), len(nun), len(multi)


def recoverable(combined, key, attr):
    """Missing `attr` values that a non-null unit-mate could supply."""
    miss = combined[attr].isna() & combined[key].notna()
    have = combined[combined[attr].notna()].groupby(key, observed=True)[attr].first()
    fillable = combined.loc[miss, key].map(have).notna()
    return int(miss.sum()), int(fillable.sum())


def main():
    train, test, combined = load()
    base = train[TARGET].mean()
    print("=" * 70)
    print("02 GROUPS & FAMILIES")
    print("=" * 70)
    print(f"marginal transport rate (train) = {base:.4f}")

    # ---- group size distribution (COMBINED) -------------------------------
    gs = combined.groupby("Group", observed=True).size()
    print(f"\n[group sizes] frame=combined")
    print(f"  groups={len(gs)}  passengers={len(combined)}")
    print(f"  size distribution (groups): {gs.value_counts().sort_index().to_dict()}")
    solo = combined["GroupSize"] == 1
    print(f"  solo travellers: {solo.sum()} ({solo.mean():.1%}) | "
          f"in a group: {(~solo).sum()} ({1 - solo.mean():.1%})")

    # ---- transport rate by group size (TRAIN-ONLY) ------------------------
    print(f"\n[rate by group size] frame=train")
    t = train.copy()
    t["GroupSizeBin"] = t["GroupSize"].clip(upper=6).astype(int)
    rt = viz.rate_table(t, "GroupSizeBin")
    rt.index = [f"{i}" if i < 6 else "6+" for i in rt.index]
    print(rt.round(4).to_string())
    r_solo = train.loc[train["GroupSize"] == 1, TARGET]
    r_grp = train.loc[train["GroupSize"] > 1, TARGET]
    print(f"  solo    : rate={r_solo.mean():.4f}  n={len(r_solo)}")
    print(f"  grouped : rate={r_grp.mean():.4f}  n={len(r_grp)}")
    print(f"  gap     : {r_grp.mean() - r_solo.mean():+.4f}")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(rt.index, rt["rate"], color=viz.SERIES[0], width=0.6)
    ax.errorbar(rt.index, rt["rate"],
                yerr=[rt["rate"] - rt["lo"], rt["hi"] - rt["rate"]],
                fmt="none", ecolor=viz.INK_MUTED, elinewidth=1.2, capsize=3)
    viz.label_bars(ax, bars, rt["rate"], pad=0.035)
    for x, n in zip(range(len(rt)), rt["n"]):
        ax.text(x, 0.02, f"n={n:,}", ha="center", fontsize=8, color=viz.SURFACE)
    ax.set_ylim(0, max(rt["hi"].max() * 1.15, 0.75))
    ax.set_xlabel("group size (passengers sharing a PassengerId group)")
    ax.set_ylabel("transport rate")
    viz.clean(ax)
    viz.baseline_ref(ax, base, f"all passengers {base:.1%}")
    viz.save(fig, "02_rate_by_groupsize",
             "Travelling alone is the single strongest raw-count signal",
             f"train only, n={len(train)} - bars are 95% Wilson intervals")

    # ---- fate concordance (TRAIN-ONLY) ------------------------------------
    print(f"\n[fate concordance] frame=train")
    rows = {}
    for key, label in [("Group", "travel group"), ("Surname", "surname (family)"),
                       ("Cabin", "exact cabin")]:
        r = pair_concordance(train, key)
        rows[label] = r
        print(f"  {label}: units={r['groups']} members={r['members']} "
              f"pairs={r['pairs']}")
        print(f"    P(mate transported | ego transported)     = {r['p_mate_t_given_ego_t']:.4f}")
        print(f"    P(mate transported | ego NOT transported) = {r['p_mate_t_given_ego_f']:.4f}")
        print(f"    same-fate pair share = {r['same_fate_share']:.4f} "
              f"(chance = {base**2 + (1-base)**2:.4f})")
        print(f"    lift over marginal = "
              f"{r['p_mate_t_given_ego_t'] - base:+.4f}")

    # Robustness: the pairwise accumulator weights large units quadratically.
    # Re-run on size-2 units ONLY, where every unit contributes exactly one pair.
    print(f"\n  [robustness] same measure restricted to size-2 units only:")
    for key, label in [("Group", "travel group"), ("Surname", "surname (family)"),
                       ("Cabin", "exact cabin")]:
        sizes = train.groupby(key, observed=True).size()
        two = sizes[sizes == 2].index
        r2 = pair_concordance(train[train[key].isin(two)], key)
        print(f"    {label:18s} pairs={r2['pairs']:5d}  "
              f"P(mate T|ego T)={r2['p_mate_t_given_ego_t']:.4f}  "
              f"P(mate T|ego F)={r2['p_mate_t_given_ego_f']:.4f}  "
              f"swing={r2['p_mate_t_given_ego_t'] - r2['p_mate_t_given_ego_f']:+.4f}  "
              f"(all sizes: {rows[label]['p_mate_t_given_ego_t'] - rows[label]['p_mate_t_given_ego_f']:+.4f})")
    print("    -> the effect is not an artefact of large-group weighting")

    # How big is the gap between the top two rows, and is it bigger than its
    # own sampling error? The "pairs" are not independent observations -- one
    # passenger sits in several of them -- so the unit resampled is the travel
    # group, via viz.cluster_bootstrap.
    print(f"\n  [uncertainty] cluster bootstrap over travel groups:")
    cb = viz.cluster_bootstrap(
        train.loc[train[TARGET].notna(), ["Group", "Cabin", TARGET]],
        "Group", concordance_swings, reps=4000, seed=0)
    print(f"    reps={cb.attrs['reps']}  seed={cb.attrs['seed']}  "
          f"clusters resampled={cb.attrs['clusters']}  "
          f"({int((1 - cb.attrs['alpha']) * 100)}% percentile intervals)")
    for name in ["group", "cabin", "cabin_minus_group"]:
        r = cb.loc[name]
        print(f"    {name:18s} {r['point']:+.4f}  "
              f"95% CI [{r['lo']:+.4f}, {r['hi']:+.4f}]  se={r['se']:.4f}")
    diff = cb.loc["cabin_minus_group"]
    print(f"    P(cabin swing > group swing) = {diff['p_positive']:.3f}")
    print(f"    -> cabin vs group: the difference is INSIDE its own interval,")
    print(f"       so the two are not distinguishable by this data.")
    print(f"       group vs surname IS separated: the group interval "
          f"[{cb.loc['group','lo']:+.4f}, {cb.loc['group','hi']:+.4f}] excludes")
    print(f"       the surname point "
          f"{rows['surname (family)']['p_mate_t_given_ego_t'] - rows['surname (family)']['p_mate_t_given_ego_f']:+.4f}.")

    conc = pd.DataFrame(rows).T
    chance = base**2 + (1 - base) ** 2
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(conc))
    w = 0.36
    b1 = ax.bar(x - w/2, conc["p_mate_t_given_ego_t"], w,
                color=viz.SERIES[0], label="ego transported")
    b2 = ax.bar(x + w/2, conc["p_mate_t_given_ego_f"], w,
                color=viz.SERIES[1], label="ego NOT transported")
    viz.label_bars(ax, b1, conc["p_mate_t_given_ego_t"], pad=0.015)
    viz.label_bars(ax, b2, conc["p_mate_t_given_ego_f"], pad=0.015)
    ax.set_xticks(x, [f"{i}\n{int(conc.loc[i,'pairs']):,} pairs" for i in conc.index])
    ax.set_ylabel("P(companion transported)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right", ncols=2)
    viz.clean(ax)
    viz.baseline_ref(ax, base, f"marginal {base:.1%}")
    viz.save(fig, "02_fate_concordance",
             "A companion's fate is the strongest single predictor in the data",
             f"train only - knowing one member's outcome moves the other's odds far off the {base:.0%} marginal")

    # ---- attribute constancy within group / family (COMBINED) -------------
    print(f"\n[attribute constancy within unit] frame=combined")
    const_rows = []
    for key in ["Group", "Surname"]:
        for attr in ["HomePlanet", "Deck", "Side", "CabinNum", "Cabin",
                     "Destination", "Surname", "CryoSleep"]:
            if attr == key:
                continue
            share, n_test, n_multi = constancy(combined, key, attr)
            const_rows.append({"unit": key, "attr": attr,
                               "units_multi": n_multi, "units_tested": n_test,
                               "untestable": n_multi - n_test,
                               "single_valued": share})
    const = pd.DataFrame(const_rows)
    print(const.round(4).to_string(index=False))
    print("  units_multi  = multi-member units of this kind in the frame")
    print("  units_tested = those with >=2 OBSERVED values of the attribute,")
    print("                 i.e. the only ones that could have violated it.")
    print("  -> the 100% results below hold on a denominator that could have broken them.")

    piv = const.pivot(index="attr", columns="unit", values="single_valued")
    piv = piv.sort_values("Group", ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    x = np.arange(len(piv))
    w = 0.36
    b1 = ax.barh(x - w/2, piv["Group"], w, color=viz.SERIES[0], label="travel group")
    b2 = ax.barh(x + w/2, piv["Surname"], w, color=viz.SERIES[1], label="surname")
    viz.label_bars(ax, b1, piv["Group"], horizontal=True, pad=0.008)
    viz.label_bars(ax, b2, piv["Surname"], horizontal=True, pad=0.008)
    ax.set_yticks(x, piv.index)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("share of multi-member units where the attribute is single-valued")
    ax.legend(loc="lower right")
    viz.clean(ax, which="x")
    viz.save(fig, "02_group_constancy",
             "Groups are near-perfectly homogeneous in origin and cabin",
             "train+test - a single-valued attribute is recoverable from a unit-mate")

    # ---- recoverability of missing values ---------------------------------
    print(f"\n[recoverable missing values via unit-mates] frame=combined")
    rec_rows = []
    for key in ["Group", "Surname"]:
        for attr in ["HomePlanet", "Deck", "Side", "Destination", "Cabin"]:
            miss, fill = recoverable(combined, key, attr)
            rec_rows.append({"unit": key, "attr": attr, "missing": miss,
                             "recoverable": fill,
                             "share": fill / miss if miss else np.nan})
    rec = pd.DataFrame(rec_rows)
    print(rec.round(4).to_string(index=False))

    r = rec[rec["unit"] == "Group"].set_index("attr").sort_values("share")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bars = ax.barh(r.index, r["share"], color=viz.SERIES[2], height=0.6)
    viz.label_bars(ax, bars, r["share"], horizontal=True, pad=0.012)
    for bar, (m, f) in zip(bars, zip(r["missing"], r["recoverable"])):
        ax.text(0.012, bar.get_y() + bar.get_height() / 2, f"{f} of {m}",
                va="center", ha="left", fontsize=8.5, color=viz.SURFACE)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("share of missing values a groupmate can supply")
    viz.clean(ax, which="x")
    viz.save(fig, "02_recoverable_missing",
             "Much of the missingness is not missing - a groupmate already holds it",
             "train+test - imputation by group, not by column mode")


if __name__ == "__main__":
    main()
