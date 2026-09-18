# Spaceship Titanic — exploratory data analysis

An exploratory analysis of Kaggle's [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic)
dataset, written up two ways: **`FINDINGS.md`** for the twelve results with their
numbers, and **`notebooks/01_eda_walkthrough.ipynb`** as a guided walkthrough that
explains the reasoning behind each step for a reader new to data science.

Built for a CSC345 course presentation. No model here yet — this is the analysis
that comes before one.

## The headline

**The raw columns are not the dataset.** Four of the strongest signals in the file
are not columns at all — they are encoded inside string fields that a naive
`read_csv` treats as identifiers:

| Field | Actually contains |
|---|---|
| `PassengerId` (`gggg_pp`) | the passenger's **travel group** |
| `Cabin` (`deck/num/side`) | **deck**, cabin **number**, and **side** of the ship |
| `Name` | the **family** |

Decoding those three is worth more than any modelling choice made afterwards.
A companion's fate turns out to be the single strongest predictor in the file.

## The data

8,693 train / 4,277 test passengers. The target `Transported` is 50.36% positive,
so a majority-class guess scores 0.504 — every rate in `FINDINGS.md` should be read
against that line.

Two conventions hold throughout, because mixing them up is easy and produces wrong
numbers:

- **Rates are train-only.** `test.csv` has no `Transported` column.
- **Structural facts** (constancy, missingness, cabin geometry) use train+test,
  n=12,970, and are marked as such.

Rates on thin slices carry Wilson intervals; group-level statistics, where the same
passenger appears in many pairs, use a cluster bootstrap instead — a Wilson interval
on a pair denominator comes out roughly 1.7× too narrow.

## Layout

```
src/load.py       the structural decode — PassengerId -> Group/MemberNo,
                  Cabin -> Deck/CabinNum/Side, Name -> Surname.
                  Returns (train, test, combined). Does no imputation.
src/viz.py        shared chart helpers and the palette; wilson(), cluster_bootstrap()
src/0*.py         five analysis scripts: overview, groups, cabin, spend, demographics
output/*.log      each script's captured stdout — the raw numbers behind every claim
figures/          the 20 figures the scripts produce
figures/nb/       three highlight charts produced by the notebook
FINDINGS.md       12 findings, with data-quality notes and modelling implications
notebooks/        the guided walkthrough
data/            the competition CSVs (see Data below)
```

## Reproducing it

The analysis is deterministic: re-running the scripts regenerates all 20 figures
byte-identically and reproduces every log.

```bash
uv venv --python 3.12 && uv pip install -r requirements.txt

.venv/bin/python src/01_overview.py     # or any of src/0*.py
.venv/bin/jupyter lab                   # then open notebooks/
```

Pinned to Python 3.12, pandas 3.0.5, matplotlib 3.11.2, scikit-learn 1.9.1.
Full pins in `requirements.txt`.

## If you build a model on this

`FINDINGS.md` closes with five concrete recommendations. The one that costs the
most to get wrong:

> **Split validation by travel group, not by random row.** No `PassengerId` group
> straddles the train/test boundary — Kaggle split group-wise — but 1,536 surnames
> do. A random row split leaks information between training and validation, and the
> score will look better than it is.

## Data

`data/` holds the competition CSVs, included so the analysis runs on clone. The
passenger records are **synthetic** — generated for the competition, not real people.
The data originates from Kaggle's Spaceship Titanic competition and its use is
governed by that competition's rules; see the
[competition page](https://www.kaggle.com/competitions/spaceship-titanic) for the
current terms.
