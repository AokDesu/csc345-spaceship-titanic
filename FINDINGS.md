# Spaceship Titanic — EDA findings

Dataset: 8,693 train / 4,277 test passengers. Target `Transported` is 50.36% positive,
so a majority-class guess scores 0.504 and every rate below should be read against that line.

All rates are **train-only**. Structural facts (constancy, missingness, cabin geometry) use
**train+test** (n=12,970) and are marked as such. Reproduce with `src/0*.py`; raw numbers in `output/*.log`.

---

## The headline: the raw columns are not the dataset

Four of the strongest signals are not columns at all — they are encoded *inside* string fields
that a naive `read_csv` treats as identifiers. `PassengerId` is a travel group. `Cabin` is
deck/number/side. `Name` is a family. Decoding those three fields is worth more than any
modelling choice made afterwards.

---

## 1. A companion's fate is the strongest single predictor in the file

`figures/02_fate_concordance.png`

Passengers travel in groups (`PassengerId` = `gggg_pp`). Knowing one member's outcome moves
another member's odds a long way off the 50.4% marginal:

| unit | pairs | P(companion transported \| ego transported) | \| ego **not** transported | swing |
|---|---:|---:|---:|---:|
| exact cabin | 3,323 | **65.0%** | 54.6% | 10.4pp |
| travel group | 4,501 | **60.7%** | 51.5% | 9.3pp |
| surname (family) | 18,804 | 51.6% | 47.0% | 4.6pp |

Fate is *correlated within a group but far from determined* — 60.7%, not 95%. Groups were not
transported wholesale.

**How far does the ordering go?** As far as **group > surname**, and no further. The pairs are
not independent observations — one passenger sits in several of them — so the intervals below
come from a cluster bootstrap that resamples whole travel groups (`viz.cluster_bootstrap`, 4,000
replicates, seed 0; `output/02_groups.log:47-56`):

| swing | point | 95% CI |
|---|---:|---:|
| travel group | +9.3pp | [+5.7, +12.8] |
| exact cabin | +10.4pp | [+6.5, +14.5] |
| **cabin − group** | **+1.2pp** | **[−0.8, +3.2]** |

The group interval excludes the surname swing (+4.6pp), so **group beats surname** is a real
ordering. The cabin-minus-group difference sits inside its own interval, at P(cabin > group) =
0.868 — so **exact cabin and travel group are not distinguishable by this data.** Neither should
be prioritised over the other on this evidence.

Nor does any of this cleanly separate location from relationship — cabin-mates are mostly a
subset of groupmates (finding 9) and are also the closest social tie — so read it as the
pattern, not a mechanism.

**Robustness (checked, not assumed):** the accumulator weights large units quadratically, so it
was re-run on **size-2 units only**, where each unit contributes exactly one pair. The effect
gets *stronger*, not weaker — cabin swing +13.8pp (686 pairs) vs +10.4pp all-sizes; group
+10.3pp (841 pairs) vs +9.3pp. Not an artefact of group size.

## 2. Travelling alone costs you 11 points

`figures/02_rate_by_groupsize.png`

55.1% of all passengers travel solo (train+test). In **train**, solo travellers were
transported at **45.2%** and anyone in a group at **56.7%**. The rate climbs with group size to a peak at size 4 (64.1%) and then falls back
— but sizes 5+ have n≤509 and wide intervals, so read the solo-vs-grouped split as the finding
and the curve shape as decoration.

## 3. CryoSleep implies zero spend — with no exceptions in 22,374 observed spend cells

`figures/04_cryo_zero_spend.png`

Every one of the **4,581 frozen passengers** (train+test) spent exactly 0 on all five services:
**22,374 observed spend cells, zero violations.** The remaining 531 of their 22,905 cells are
NaN and are excluded from the test rather than assumed into it. Awake and unknown-CryoSleep rows
cannot break a rule of the form "frozen ⇒ zero", so the denominator here is 4,581 rows, not the
whole 12,970. This is a hard rule of the data generator, and it has two consequences:

- **531 NaN spend cells under `CryoSleep=True` are known zeros, not gaps.** Imputing them with a
  column median is actively wrong.
- **It runs backwards too.** Of 310 rows with `CryoSleep` missing, **174 (56.1%) show nonzero
  spend and are therefore certainly awake.** No model needed.

CryoSleep itself is the strongest raw column: **81.8% transported frozen vs 32.9% awake.**

## 4. HomePlanet is 100% constant within a group *and* within a surname

`figures/02_group_constancy.png`, `figures/02_recoverable_missing.png`

Across the **2,076 multi-member groups** and **2,185 surnames** in which at least two members
have a recorded `HomePlanet`: **zero violations.** Same for **Side** — a travel group is always
on one side of the ship (**2,068 groups**, zero violations).

Those denominators deliberately count only units that *could* have broken the rule. `constancy()`
takes multi-member units with two or more **observed** values of the attribute: a two-person
group whose second `HomePlanet` is missing carries one distinct value, scores as constant, and
was never a test of anything. 2,135 multi-member groups exist in train+test; 59 of them are
untestable for `HomePlanet` and 67 for `Side`. The 100% results survive on the stricter
denominator — which is what makes them worth trusting.

Deck is *not* constant (**67.5%** single-valued, 673 violating groups), and neither is
Destination (**48.5%**) nor CryoSleep (**40.7%**). See `output/02_groups.log:60-79`.

This turns imputation into lookup. Of the 278 missing `HomePlanet` values that have a surname,
**271 (97.5%) are recoverable** from a namesake elsewhere in the file — and 1,536 of 2,217 train
surnames also appear in test, so the lookup crosses the split. Recovering via *group* only reaches
45.5%, because most passengers travel alone.

**Note the asymmetry:** groups are homogeneous in almost everything, families only in origin.
Family recovers *more* missing values (families are bigger and span groups); group constancy
covers *more attributes*.

## 5. Nobody under 13 spends a single credit — and the switch flips exactly at 13

`figures/05_rate_by_age.png`, `figures/05_age_hist.png`

Of **1,157 passengers aged 0–12 (train+test)**, **not one** has a nonzero charge on any
service. At age 13 exactly, **53% of 219 (train+test)** do. This is a discrete rule, not a smooth
trend. Both counts come from notebook cell 48, the only place in the repo that computes them —
`src/05_demog.py` works on train alone, where there are 806 such children
(`output/05_demog.log:28`).

**The age effect is not just a CryoSleep artefact — but one stratification was not enough.**
Among **awake** passengers only (train), under-13s were transported at **68.2%** vs **30.5%** for
awake adults, a 37.8pp gap *larger* than the unstratified one (`output/05_demog.log:31`). That
comparison still carries a second confounder: **100% of awake under-13s spend nothing**, against
**2.2% of awake adults**, and finding 6 puts spending at roughly −15pp on its own. So it is partly
children-who-spend-nothing against adults-who-almost-all-spend. Holding spend constant as well:

| comparison (train) | children <13 | adults ≥18 | gap |
|---|---|---|---:|
| awake only | 68.2% (n=406) | 30.5% (n=4,576) | +37.8pp |
| awake **and** zero total spend | 68.2% (n=406) [63.5, 72.6] | 37.0% (n=100) [28.2, 46.8] | **+31.2pp** |

About a sixth of the published gap is the spend channel, not age. The effect survives at
**+31.2pp on non-overlapping intervals** — still large and real — but 37.8pp is not a measure of
"the age effect net of confounding". Ages 18–65 are flat at 46–49%; essentially the entire age
signal lives under 18.

## 6. Spending is not one variable — the services point in opposite directions

`figures/04_rate_by_service.png`

On a raw cut all five services look negative, but that is confounded: spending at all implies
being awake, and awake passengers sit at 32.9%. **Among awake passengers only** (n=5,439):

| service | spent >0 | spent 0 | gap | z | verdict |
|---|---:|---:|---:|---:|---|
| RoomService | 26.0% | 41.1% | **−15.1pp** | −11.7 | lowers |
| Spa | 27.8% | 40.1% | **−12.3pp** | −9.3 | lowers |
| VRDeck | 27.6% | 39.4% | **−11.7pp** | −9.0 | lowers |
| FoodCourt | 34.6% | 30.8% | **+3.8pp** | +2.9 | raises |
| ShoppingMall | 31.8% | 34.2% | −2.4pp | −1.9 | **flat** (95% CI crosses 0) |

RoomService, Spa and VRDeck — the private/in-cabin services — predict *not* being transported.
FoodCourt runs the **other way**, and ShoppingMall is flat: its interval [−4.9pp, +0.1pp] includes
zero, so it should not be read as negative. Collapsing these into one `TotalSpend` column destroys
the split entirely.

Separately, the binary "spent anything at all" is nearly as strong as CryoSleep: **78.7%** for
non-spenders vs **29.9%** for spenders. The interesting cell is awake-but-spent-nothing —
518 passengers at **61.6%**, well above the 50.4% marginal.

## 7. Starboard beats port on seven of the eight populated decks

`figures/03_side_by_deck.png`

Overall **S 55.5% vs P 45.1%** (+10.4pp on n=8,494, train) is solid. The per-deck picture is
weaker than an earlier draft of this file claimed, in three ways.

**Seven of the eight populated decks, not all of them.** The direction holds on A +11.1,
B +11.0, C +18.3, D +6.2, E +2.9, F +5.9, G +13.6pp — but `src/03_cabin.py` keeps only cells
with n ≥ 30, which silently drops **deck T**, where the sign *reverses*: P 1 of 4 (25.0%) versus
S 0 of 1 (0.0%). Five train rows make deck T meaningless, and that is a reason to name and
exclude it, not to write "every deck" (finding 8 already says to ignore deck T).

**Sign-consistent on seven, individually significant on four.** On a two-proportion z test the
per-deck gap clears its own 95% interval on B, C, F and G only. A (z=1.78), D (z=1.37) and
E (z=0.89) do not.

**The decks are not independent strata.** Finding 8 argues the opposite — that deck is largely a
home-planet label — and the two claims cannot both be right. Sign-consistency across seven
correlated decks is still real evidence (p ≈ 0.016 under a no-effect null), but it is not the
seven independent confirmations the earlier wording implied.

(The z tests and the no-effect null were re-derived in the 2026-09-17 EDA audit; `src/` computes
the per-deck rates but no test on them.)

## 8. Deck is *mostly* a home-planet label, but not only that

`figures/03_deck_by_homeplanet.png`, `figures/03_deck_within_homeplanet.png`

Cramér's V(deck, HomePlanet) = **0.75** (`output/03_cabin.log:61`). Decks A, B and C are 100%
Europa; deck G is 100% Earth; only D, E and F are mixed. (Deck T is all-Europa too, but on 10
passengers with a known planet, so it is not evidence of anything.) So "deck matters" and "home
planet matters" are **substantially the same finding** and should not be counted twice.

**But which of the two do you drop?** Cramér's V is symmetric and cannot say; the relationship is
not symmetric at all. On the same contingency table (n=12,390, train+test) the uncertainty
coefficients are **U(HomePlanet | Deck) = 0.632** and **U(Deck | HomePlanet) = 0.383**. Knowing
someone's deck nearly fixes their planet; knowing their planet leaves six decks open. **Deck
carries strictly more, so drop `HomePlanet` and keep `Deck`** — not the other way round, which is
the direction the rest of this section's framing invites. (Re-derived in the 2026-09-17 EDA
audit; `src/03_cabin.py` computes only the symmetric V.)

But deck is not purely a proxy. Within a single home planet the deck rate still swings hard:
Mars passengers on deck F were transported at **65.0%** vs **26.1%** on deck E (n=1,110 and 330).
Europa spans 49.6% (A) to 73.4% (B). Deck carries real information beyond origin.

Raw deck rates range from **E 35.7%** to **B 73.4%**. Deck T has 5 train rows — ignore it.

## 9. Cabin number is a location, not an ID

`figures/03_rate_by_cabinnum.png`

Of 2,068 multi-member groups with a cabin, **1,395 (67.5%) occupy one identical cabin.** The
remaining **673 each span more than one deck** — every single one. There is no "adjacent cabin" case at all: a
group either shares a single cabin or is scattered across the ship. (An earlier draft called
these groups "adjacently booked" — that was wrong. It also counted 1,397 and 671, because it
grouped on `CabinNum` rather than the full `Cabin` string; see below.)

Three of the 673 have a cabin-number span of ≤2, and they are the sharpest possible evidence that
the span is not a distance: groups **0923** (`B/38/S`, `C/38/S`), **3494** (`B/112/P`, `D/112/P`)
and **0006** (`G/0/S`, `F/2/S`) are *decks apart* at numeric spans of 0, 0 and 2. Cabin numbering
restarts on every deck — A ends at 109, F at 1,894 — so subtracting two numbers from different
decks measures nothing at all. The unanimous deck-spanning fact is the whole argument; an earlier
draft also quoted a "median split group spans 254 cabin numbers", which is that same meaningless
subtraction and has been dropped.

The first two of those three groups are why this section's counts moved: `CabinNum` restarts per
deck, so `C/38/S` and `B/38/S` share a *number* and not a cabin, and the old grouping scored them
as cohabiting. 673 is now also consistent with the 673 deck-constancy violations in finding 4 —
they are the same set of groups.

The number is therefore a position, not an id, and within a deck it carries real signal: across
cabin-number deciles the rate moves **0.363 on deck G** (37.1% → 73.4%) and **0.176 on deck F**,
both clearing a permutation null comfortably (p < 0.0001 and p = 0.0005). An earlier draft
quoted **deck C (0.240)** as the second example; on ten bins of median size 74 that spread lands
exactly on its own 95th percentile (p = 0.049), so it is not an independent confirmation. (Nulls
re-derived in the 2026-09-17 EDA audit.)

## 10. Missingness is uniform, independent, and expensive to drop

`figures/01_missingness.png`, `figures/01_nulls_per_row.png`

Every column except `PassengerId` is missing 2.03–2.39% — a 0.36pp spread across 12 columns.
It looks injected at random, per cell: 3,083 rows have ≥1 null against 3,052 expected under
full independence (within 1%).

The trap: any single column looks trivially droppable at ~2%, but **23.8% of all rows have at
least one null.** `dropna()` costs a quarter of the data. Given findings 3 and 4, most of that
is recoverable rather than lost.

## 11. VIP is too small to trust, and points the "wrong" way

`figures/05_vip.png`

VIP is **2.15% of the ship** — 273 of the 12,674 rows where `VIP` is actually recorded, with
296 rows missing it entirely (199 VIPs in train). Quoting 273/12,970 instead, as an earlier draft
did, silently assumes all 296 missing flags are False. VIPs were transported at 38.2% vs
50.6% for everyone else. The 95% Wilson intervals do *not* overlap ([31.7%, 45.1%] vs
[49.6%, 51.7%]), so the gap is real, though the VIP interval is **6× wider**.

The interesting part is what conditioning does. **There is not one VIP from Earth** — 66% are
Europa, 34% Mars — and Europa is the highest-rate planet. So the planet mix was *masking* the
VIP deficit, not causing it. Within planet the gap **widens**:

| planet | VIP rate | non-VIP rate | gap |
|---|---:|---:|---:|
| Europa | 48.9% (n=131) | 67.0% (n=1,958) | **−18.2pp** |
| Mars | 15.9% (n=63) | 53.4% (n=1,653) | **−37.5pp** |

Against the −12.4pp unconditional figure. The effect is real and larger than it first looks — but
it rests on 194 train VIPs across two planets, so the error bars stay wide.

## 12. Destination is weak — but not purely an origin artefact

`figures/05_rate_by_origin_destination.png`

55 Cancri e 61.0% / PSO J318.5-22 50.4% / TRAPPIST-1e 47.1% — an unconditional spread of 0.139.
Conditioning on home planet does **not** flatten it uniformly:

| planet | 55 Cancri e | PSO J318.5-22 | TRAPPIST-1e | spread |
|---|---:|---:|---:|---:|
| Earth | 50.4% (n=690) | 49.9% (n=712) | **38.9%** (n=3,101) | 0.115 |
| Europa | 69.0% (n=886) | 73.7% (n=19) | 63.5% (n=1,189) | 0.055 |
| Mars | 61.1% (n=193) | 44.9% (n=49) | 51.2% (n=1,475) | 0.162 |

So destination **survives** conditioning for Earth — TRAPPIST-1e-bound Earth passengers sit ~11pp
below the other two, on 3,101 rows — and it collapses only for Europa. Mars's 0.162 rests on a
49-row PSO cell and should not be leaned on. Destination is the weakest of the categoricals, but
"it is just origin in disguise" overstates it.

Origin is stronger — **Europa 65.9%, Mars 52.3%, Earth 42.4%** (train,
`output/05_demog.log:39-41`) — but it is **not clean**. It is confounded by CryoSleep, exactly as
spend (finding 6) and age (finding 5) are, and stratifying changes the answer:

| planet (train) | share frozen | frozen rate (n) | awake rate (n) |
|---|---:|---:|---:|
| Earth | 30.8% | 65.6% (1,382) | **32.1%** (3,106) |
| Europa | 43.9% | 98.9% (911) | **40.0%** (1,162) |
| Mars | 39.0% | 91.2% (669) | **27.7%** (1,047) |

("Share frozen" is among rows where `CryoSleep` is recorded, which is also the denominator of the
two rate columns.) Two things follow. The unconditional spread is 23.5pp (42.4 → 65.9); among
awake passengers it is **12.3pp** (27.7 → 40.0), so roughly **half the home-planet effect is
CryoSleep mix** — Europa freezes at 43.9% against Earth's 30.8%. And **the Earth/Mars ordering
reverses**: Mars beats Earth unconditionally (52.3% vs 42.4%), Earth beats Mars among the awake
(32.1% vs 27.7%). Home planet needs the same stratification the rest of this file applies
elsewhere. (Re-derived in the 2026-09-17 EDA audit; nothing in `src/` stratifies HomePlanet by
CryoSleep.)

---

## Data-quality notes worth carrying into any model

- **No PassengerId group straddles train/test** (0 of 9,280). Kaggle split group-wise, so group
  features carry no cross-split leakage. **Surnames do straddle** (1,536 shared), so a
  surname-level feature crosses the boundary — legitimate, but know that it does.
- **1,363 rows have a partial `TotalSpend`** (1–4 of 5 services NaN). Summing them silently
  understates the total. `src/load.py` exposes `SpendNaN` so these can be excluded or
  cryo-filled rather than trusted. No row has all five missing.
- `Age == 0` appears 260 times and is a genuine infant, not a missing-value sentinel — `Age`
  has 270 separate explicit NaNs.
- Cabin `T` exists (11 passengers total, 5 in train). Any deck-level statistic on it is noise.

## What this implies for modelling

1. Decode `PassengerId`, `Cabin` and `Name` before anything else — that is where the signal is.
2. Impute by **group and surname lookup**, not column mode, and fill cryo-implied spend with 0.
3. Keep the five spend columns separate; add a binary `AnySpend`; add `Age < 13`.
4. Add group-level aggregates (size, groupmate spend, groupmate cryo) — finding 1 says a
   companion's attributes predict yours. Build the **group**-level ones: finding 1 cannot
   separate cabin from group, so a cabin-level feature is untested, not better.
5. Don't double-count deck and home planet as independent evidence — and if you drop one, drop
   `HomePlanet`. Deck determines planet far more than planet determines deck (finding 8).
