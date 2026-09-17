"""Load Spaceship Titanic data and perform the structural decode.

Single source of truth for parsing PassengerId / Cabin / Name.
Every analysis script imports from here rather than re-implementing splits.

Key constraint: test.csv has no Transported column. Use `combined` for
missingness and feature distributions; use `train` for anything that computes
a target rate.

VERIFIED (see verify()): no PassengerId group straddles the train/test split --
Kaggle split group-wise, so group-level features carry no cross-split leakage.
Surnames DO straddle (1536 of 2217 train surnames also occur in test), so the
family join is the one that crosses the boundary. GroupSize is computed on the
combined frame anyway: it is identical either way here, and stays correct if
the split assumption ever changes.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

SPEND_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CAT_COLS = ["HomePlanet", "CryoSleep", "Destination", "VIP"]
TARGET = "Transported"


def _decode(df: pd.DataFrame) -> pd.DataFrame:
    """Split the three composite string columns into their components."""
    df = df.copy()

    # PassengerId -> gggg_pp
    pid = df["PassengerId"].str.split("_", expand=True)
    df["Group"] = pd.to_numeric(pid[0], errors="coerce").astype("Int64")
    df["MemberNo"] = pd.to_numeric(pid[1], errors="coerce").astype("Int64")

    # Cabin -> deck/num/side
    cab = df["Cabin"].str.split("/", expand=True)
    df["Deck"] = cab[0]
    df["CabinNum"] = pd.to_numeric(cab[1], errors="coerce").astype("Int64")
    df["Side"] = cab[2]

    # Name -> first / surname
    nm = df["Name"].str.split(" ", n=1, expand=True)
    df["FirstName"] = nm[0]
    df["Surname"] = nm[1]

    # Spend aggregates. NaN-aware: sum of all-NaN row stays NaN rather than 0,
    # so we do not silently invent a zero-spender.
    spend = df[SPEND_COLS]
    df["TotalSpend"] = spend.sum(axis=1).where(spend.notna().any(axis=1))
    df["AnySpend"] = (df["TotalSpend"] > 0).where(df["TotalSpend"].notna())
    # TotalSpend on a row with some NaN services is a PARTIAL sum. Carry the
    # count so downstream code can exclude partials rather than trust a
    # silently-low total.
    df["SpendNaN"] = spend.isna().sum(axis=1)

    return df


def load():
    """Return (train, test, combined), all structurally decoded.

    `combined` carries a `Split` column ('train'/'test') and has GroupSize /
    SurnameSize computed across both files, since groups straddle the split.
    """
    train = _decode(pd.read_csv(DATA / "train.csv"))
    test = _decode(pd.read_csv(DATA / "test.csv"))

    train["Split"] = "train"
    test["Split"] = "test"
    combined = pd.concat([train, test], ignore_index=True)

    # Group/family sizes must be computed on the combined frame.
    gsize = combined.groupby("Group")["PassengerId"].transform("size")
    combined["GroupSize"] = gsize
    ssize = combined.groupby("Surname", dropna=False)["PassengerId"].transform("size")
    combined["SurnameSize"] = ssize.where(combined["Surname"].notna())

    # Push the combined-derived sizes back onto train/test by PassengerId.
    size_map = combined.set_index("PassengerId")[["GroupSize", "SurnameSize"]]
    for df in (train, test):
        df["GroupSize"] = df["PassengerId"].map(size_map["GroupSize"])
        df["SurnameSize"] = df["PassengerId"].map(size_map["SurnameSize"])

    return train, test, combined


def verify(train, test, combined) -> int:
    """Print decode verification. Returns count of parse failures."""
    failures = 0

    print("=" * 70)
    print("STRUCTURAL DECODE VERIFICATION")
    print("=" * 70)

    print(f"\nrows: train={len(train)}  test={len(test)}  combined={len(combined)}")
    print(f"train columns: {len(train.columns)}  test columns: {len(test.columns)}")
    print(f"target present in train: {TARGET in train.columns} | "
          f"in test: {TARGET in test.columns}")

    # 1. PassengerId
    bad_pid = combined["Group"].isna() | combined["MemberNo"].isna()
    print(f"\n[1] PassengerId 'gggg_pp' parse failures: {bad_pid.sum()}")
    failures += int(bad_pid.sum())
    print(f"    PassengerId unique: {combined['PassengerId'].is_unique}")
    print(f"    PassengerId nulls : {combined['PassengerId'].isna().sum()}")
    print(f"    distinct groups   : {combined['Group'].nunique()}")
    print(f"    MemberNo range    : {combined['MemberNo'].min()}..{combined['MemberNo'].max()}")

    # Do groups straddle train/test?
    splits_per_group = combined.groupby("Group")["Split"].nunique()
    straddling = int((splits_per_group > 1).sum())
    print(f"    groups spanning BOTH train and test: {straddling} "
          f"of {len(splits_per_group)}")
    if straddling:
        n_pass = int(combined["Group"].isin(
            splits_per_group[splits_per_group > 1].index).sum())
        print(f"    -> {n_pass} passengers sit in a straddling group; group "
              f"features MUST be built on the combined frame")

    # 2. Cabin
    cab_present = combined["Cabin"].notna()
    bad_cab = cab_present & (combined["Deck"].isna() | combined["CabinNum"].isna()
                             | combined["Side"].isna())
    print(f"\n[2] Cabin 'deck/num/side' parse failures (non-null Cabin): {bad_cab.sum()}")
    failures += int(bad_cab.sum())
    print(f"    Cabin nulls: {(~cab_present).sum()}")
    print(f"    decks: {sorted(combined['Deck'].dropna().unique().tolist())}")
    print(f"    sides: {sorted(combined['Side'].dropna().unique().tolist())}")
    print(f"    CabinNum range: {combined['CabinNum'].min()}..{combined['CabinNum'].max()}")

    # 3. Name
    nm_present = combined["Name"].notna()
    bad_nm = nm_present & combined["Surname"].isna()
    print(f"\n[3] Name 'first surname' parse failures (non-null Name): {bad_nm.sum()}")
    failures += int(bad_nm.sum())
    print(f"    Name nulls: {(~nm_present).sum()}")
    print(f"    distinct surnames: {combined['Surname'].nunique()}")
    n_parts = combined.loc[nm_present, "Name"].str.split(" ").str.len()
    print(f"    name token counts: {n_parts.value_counts().to_dict()}")

    # 4. Derived
    print(f"\n[4] derived features")
    print(f"    GroupSize   range: {combined['GroupSize'].min()}..{combined['GroupSize'].max()}")
    print(f"    SurnameSize range: {combined['SurnameSize'].min()}..{combined['SurnameSize'].max()}")
    print(f"    TotalSpend  nulls: {combined['TotalSpend'].isna().sum()}")
    print(f"    rows with a PARTIAL TotalSpend (1-4 services NaN): "
          f"{int(((combined['SpendNaN'] > 0) & (combined['SpendNaN'] < 5)).sum())}")
    print(f"    rows with all 5 services NaN: "
          f"{int((combined['SpendNaN'] == 5).sum())}")

    print(f"\n{'PASS' if failures == 0 else 'FAIL'}: total parse failures = {failures}")
    return failures


if __name__ == "__main__":
    tr, te, comb = load()
    raise SystemExit(1 if verify(tr, te, comb) else 0)
