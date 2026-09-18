"""Generate the Kaggle-hosted copy of the EDA walkthrough.

Kaggle notebooks run with no repository around them: there is no `src/` to
import and the competition CSVs mount read-only under /kaggle/input. This
script rewrites the walkthrough into a single self-contained notebook, so the
Kaggle copy is always generated from this repository rather than hand-maintained
beside it.

Three things have to travel with the notebook:

  * `load.py` and `viz.py`, inlined and path-patched, for the setup cell
  * `02_groups.py`, inlined unpatched -- the concordance section loads it by
    path, it resolves no paths of its own, and its main() is guarded so
    exec_module() only defines functions
  * the eight pre-rendered figures section 8 displays, base64-embedded: they
    are build products of `src/0*.py` and do not exist on Kaggle

Usage:

    python tools/build_kaggle_notebook.py [-o OUT]

Output defaults to kaggle/spaceship-titanic-eda.ipynb, which is untracked: the
generated notebook is a build artifact, this script is the source. The three
KAGGLE_* environment variables below repoint the generated paths so the result
can be executed locally against a simulated Kaggle layout before being pushed.
"""

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
NOTEBOOK = ROOT / "notebooks" / "01_eda_walkthrough.ipynb"

# Overridable so the generated notebook can be executed locally against a
# simulated Kaggle layout before it is pushed.
#
# SCRATCH is deliberately NOT under /kaggle/working: that directory is the
# kernel's published output store, so modules left there would ship as
# downloadable artifacts of what is meant to be a readable walkthrough.
SCRATCH = os.environ.get("KAGGLE_SCRATCH", "/tmp/spaceship_src")
KAGGLE_ROOT = os.environ.get("KAGGLE_ROOT", "/kaggle/working")
KAGGLE_DATA = os.environ.get("KAGGLE_DATA", "/kaggle/input/spaceship-titanic")

REPO_URL = "https://github.com/AokDesu/csc345-spaceship-titanic"

KERNEL_SLUG = "spaceship-titanic-eda-a-guided-walkthrough"
KERNEL_TITLE = "Spaceship Titanic EDA - a guided walkthrough"
BY_PATH_MODULE = "02_groups.py"
FIGURE_DATASET_SLUG = "spaceship-titanic-eda-figures"
FIGURE_DATASET_TITLE = "Spaceship Titanic EDA figures"

DATA_FINDER_TEMPLATE = '''def _kaggle_data_dir():
    """Locate the competition data among Kaggle's mounted inputs.

    The mount point is not reliably /kaggle/input/<competition>: attaching the
    competition through the API nests it under /kaggle/input/competitions/.
    Rather than encode either guess, find the directory that actually holds
    train.csv, and name the fix when nothing does.
    """
    preferred = Path("{data}")
    if (preferred / "train.csv").is_file():
        return preferred

    root = Path("/kaggle/input")
    mounted = None
    if root.is_dir():
        found = sorted(root.rglob("train.csv"))
        if found:
            return found[0].parent
        mounted = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_dir())

    raise FileNotFoundError(
        "No mounted input contains train.csv.\\n"
        f"  expected      : {{preferred}}\\n"
        f"  /kaggle/input : {{mounted if mounted else 'nothing (directory missing)'}}\\n"
        "  Fix: open the notebook on Kaggle, Add Input -> Competitions -> "
        "Spaceship Titanic, then re-run."
    )


DATA = _kaggle_data_dir()'''

SETUP_MARKER = 'sys.path.insert(0, str(ROOT / "src"))'
BY_PATH_MARKER = 'spec_from_file_location'
FIGURE_RE = re.compile(r'display\(Image\(str\(ROOT / "figures" / "([a-z0-9_]+)\.png"\)\)\)')


def patch_paths(source: str, module: str) -> str:
    """Repoint a module's project-relative paths at Kaggle's filesystem.

    Both modules resolve ROOT from __file__, which is meaningless once the file
    is written to a scratch directory. Each has exactly one ROOT line plus
    derived DATA/FIGURES lines. Substitutions that do not match raise, so a
    future edit to src/ fails the build rather than emitting a notebook that
    half works.
    """
    if "'''" in source:
        raise SystemExit(f"{module}: contains ''' and cannot be inlined as a raw string")

    before = source
    source = re.sub(
        r"^ROOT = Path\(__file__\)\.resolve\(\)\.parent\.parent$",
        f'ROOT = Path("{KAGGLE_ROOT}")  # patched by tools/build_kaggle_notebook.py',
        source, count=1, flags=re.M)
    if source == before:
        raise SystemExit(f"{module}: expected ROOT assignment not found")

    if module == "load.py":
        before = source
        source = re.sub(r'^DATA = ROOT / "data"$', "DATA = _kaggle_data_dir()", source, count=1, flags=re.M)
        if source == before:
            raise SystemExit("load.py: expected DATA assignment not found")
        # Kaggle names the mount after the competition, but that is an assumption
        # rather than a guarantee, and a wrong guess fails three cells later with
        # a bare FileNotFoundError. Find the directory that actually holds the
        # data, and say what was mounted when there is none.
        source = source.replace("DATA = _kaggle_data_dir()", DATA_FINDER_TEMPLATE.format(data=KAGGLE_DATA), 1)

    # viz.py's FIGURES.mkdir() is non-recursive; KAGGLE_ROOT exists, so it holds.
    return source


def _cell(kind: str, lines: list, metadata: dict = None) -> dict:
    src = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
    cell = {"cell_type": kind, "metadata": metadata or {}, "source": src}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def setup_cell(load_src: str, viz_src: str, groups_src: str) -> list:
    """Replace the local notebook's src/ import block."""
    return _cell("code", [
        f"# Generated from {REPO_URL}",
        "# The project's modules are inlined below so this notebook runs standalone on Kaggle.",
        "# load.py stays the single source of truth for the PassengerId / Cabin / Name decode.",
        "",
        "import sys",
        "from pathlib import Path",
        "",
        f'SCRATCH = Path("{SCRATCH}")',
        "SCRATCH.mkdir(parents=True, exist_ok=True)",
        "",
        "LOAD_PY = r'''",
        load_src.rstrip(),
        "'''",
        "",
        "VIZ_PY = r'''",
        viz_src.rstrip(),
        "'''",
        "",
        "GROUPS_PY = r'''",
        groups_src.rstrip(),
        "'''",
        "",
        '(SCRATCH / "load.py").write_text(LOAD_PY)',
        '(SCRATCH / "viz.py").write_text(VIZ_PY)',
        f'(SCRATCH / "{BY_PATH_MODULE}").write_text(GROUPS_PY)',
        "sys.path.insert(0, str(SCRATCH))",
        "",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "",
        "import load as L",
        "import viz",
        "",
        "# viz.py sets the 'Agg' backend (for scripts that only write files).",
        "# In a notebook we want charts to appear inline as well, so we save AND display.",
        "from IPython.display import Image, display",
        "",
        f'ROOT = Path("{KAGGLE_ROOT}")',
        '(ROOT / "figures" / "nb").mkdir(parents=True, exist_ok=True)',
        "",
        "def show(fig, name, title=None, subtitle=None):",
        '    """Save a figure to figures/nb/ and display it inline."""',
        '    path = viz.save(fig, f"nb/{name}", title, subtitle)',
        "    display(Image(str(path)))",
        "",
        'print("python     :", sys.version.split()[0])',
        'print("pandas     :", pd.__version__)',
        'print("data       :", L.DATA)',
        '',
        '_inp = Path("/kaggle/input")',
        'print("inputs     :", sorted(str(p.relative_to(_inp)) for p in _inp.rglob("*") if p.is_dir()) if _inp.is_dir() else "NONE MOUNTED")',
    ])


def figures_cell(names: list) -> dict:
    """Load section 8's charts from the attached figures dataset.

    These are outputs of src/0*.py, which does not run in this notebook. They
    ship as a small Kaggle dataset rather than base64 inside the notebook: a
    600 KB blob is not something a reader should have to scroll past, and it
    made the notebook eight times larger than its own content.
    """
    return _cell("code", [
        "# Section 8's charts are build products of the repository's analysis scripts,",
        "# which do not run here. They arrive as an attached dataset, so this notebook",
        "# still needs no internet access.",
        "",
        "def _figure_dir():",
        '    """Find the attached figures dataset among Kaggle\'s mounted inputs."""',
        '    probe = "%s.png"' % names[0],
        '    root = Path("/kaggle/input")',
        "    if root.is_dir():",
        '        for hit in sorted(root.rglob(probe)):',
        "            return hit.parent",
        f'    local = Path("{ROOT}") / "figures"',
        "    if (local / probe).is_file():",
        "        return local",
        "    raise FileNotFoundError(",
        f'        "Figures dataset not attached. Add Input -> Datasets -> "',
        f'        "{FIGURE_DATASET_SLUG}, then re-run."',
        "    )",
        "",
        "FIGURE_DIR = _figure_dir()",
        'print("figures    :", FIGURE_DIR)',
        "",
        "def repo_figure(name):",
        '    return Image(filename=str(FIGURE_DIR / f"{name}.png"))',
    ])


def write_figure_dataset(out_dir: Path, names: list) -> Path:
    """Stage the figures the notebook displays, ready for `kaggle datasets`."""
    user = kaggle_username()
    d = out_dir / "figures-dataset"
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.png"):
        old.unlink()
    for n in names:
        shutil.copy2(ROOT / "figures" / f"{n}.png", d / f"{n}.png")
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": FIGURE_DATASET_TITLE,
        "id": f"{user}/{FIGURE_DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2) + "\n")
    return d


def footer_cell() -> dict:
    return _cell("markdown", [
        "---",
        "",
        f"*Analysis scripts, the full findings write-up and this notebook's source: "
        f"[{REPO_URL.split('//')[1]}]({REPO_URL})*",
        "",
        "*Generated from that repository by `tools/build_kaggle_notebook.py`, with the",
        "project's modules inlined in the setup cell so it runs standalone here.*",
    ])


def kaggle_username() -> str:
    """Resolve the Kaggle account the notebook publishes under.

    The kernel id must be <username>/<slug>, and the username is NOT the GitHub
    one. It is read from the API credentials rather than guessed; KAGGLE_USERNAME
    overrides. Without either, a placeholder is written and the caller is told,
    so a push fails on an obviously wrong id instead of creating a kernel under
    a mistyped account.
    """
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]

    # Access-token auth (KAGGLE_API_TOKEN) writes no kaggle.json, so ask the CLI.
    try:
        out = subprocess.run(["kaggle", "config", "view"], capture_output=True,
                             text=True, timeout=30).stdout
        m = re.search(r"^-\s*username:\s*(\S+)$", out, re.M)
        if m and m.group(1) != "None":
            return m.group(1)
    except (OSError, subprocess.SubprocessError):
        pass

    # Legacy username+key credentials file.
    for c in (Path.home() / ".kaggle" / "kaggle.json",
              Path.home() / ".config" / "kaggle" / "kaggle.json"):
        if c.exists():
            try:
                return json.loads(c.read_text())["username"]
            except (json.JSONDecodeError, KeyError):
                pass
    return "UNKNOWN-SET-KAGGLE-USERNAME"


def write_metadata(out_dir: Path, notebook_name: str) -> tuple:
    """Emit kernel-metadata.json beside the notebook.

    enable_internet is off: it is the Kaggle default for a competition notebook,
    and the generated notebook needs no network - the competition data mounts
    locally and section 8's figures are embedded.
    """
    user = kaggle_username()
    meta = {
        "id": f"{user}/{KERNEL_SLUG}",
        "title": KERNEL_TITLE,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "false",
        "enable_gpu": "false",
        "enable_tpu": "false",
        "enable_internet": "false",
        "dataset_sources": [f"{user}/{FIGURE_DATASET_SLUG}"],
        "competition_sources": ["spaceship-titanic"],
        "kernel_sources": [],
        "model_sources": [],
    }
    path = out_dir / "kernel-metadata.json"
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path, user


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=str(ROOT / "kaggle" / "spaceship-titanic-eda.ipynb"))
    args = ap.parse_args()

    nb = json.loads(NOTEBOOK.read_text())
    cells = nb["cells"]

    load_src = patch_paths((SRC / "load.py").read_text(), "load.py")
    viz_src = patch_paths((SRC / "viz.py").read_text(), "viz.py")
    groups_src = (SRC / BY_PATH_MODULE).read_text()
    if "'''" in groups_src:
        raise SystemExit(f"{BY_PATH_MODULE}: contains ''' and cannot be inlined")

    # Locate cells by content, not index, so an inserted cell upstream cannot
    # silently rewrite the wrong one.
    setup_at = [i for i, c in enumerate(cells)
                if c["cell_type"] == "code" and SETUP_MARKER in "".join(c["source"])]
    if len(setup_at) != 1:
        raise SystemExit(f"expected 1 setup cell, found {len(setup_at)}")
    cells[setup_at[0]] = setup_cell(load_src, viz_src, groups_src)

    # The concordance cell loads 02_groups.py from src/; point it at SCRATCH.
    by_path = [i for i, c in enumerate(cells)
               if c["cell_type"] == "code" and BY_PATH_MARKER in "".join(c["source"])]
    if len(by_path) != 1:
        raise SystemExit(f"expected 1 spec_from_file_location cell, found {len(by_path)}")
    i = by_path[0]
    patched = "".join(cells[i]["source"]).replace(
        f'ROOT / "src" / "{BY_PATH_MODULE}"', f'SCRATCH / "{BY_PATH_MODULE}"')
    if patched == "".join(cells[i]["source"]):
        raise SystemExit(f"could not repoint {BY_PATH_MODULE} load path")
    cells[i]["source"] = patched.splitlines(keepends=True)

    # Swap figure-file displays for embedded data, collecting the names as we go.
    names = []
    for c in cells:
        if c["cell_type"] != "code":
            continue
        s = "".join(c["source"])
        found = FIGURE_RE.findall(s)
        if not found:
            continue
        for n in found:
            if not (ROOT / "figures" / f"{n}.png").exists():
                raise SystemExit(f"figure referenced but missing: figures/{n}.png")
            if n not in names:
                names.append(n)
        c["source"] = FIGURE_RE.sub(lambda m: f'display(repo_figure("{m.group(1)}"))', s).splitlines(keepends=True)

    # Kaggle executes the notebook on push; stored outputs would only be stale.
    for c in cells:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

    # Figure data goes immediately after setup, before its first use.
    cells.insert(setup_at[0] + 1, figures_cell(names))
    cells.append(footer_cell())

    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

    # nbformat >= 4.5 requires a stable id per cell; index-derived so a rebuild
    # of an unchanged notebook is byte-identical.
    nb["nbformat_minor"] = max(nb.get("nbformat_minor", 0), 5)
    for n, c in enumerate(cells):
        c["id"] = f"cell-{n:03d}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out}")
    print(f"  cells    : {len(cells)}")
    print(f"  figures  : {len(names)} (from attached dataset)")
    print(f"  size     : {out.stat().st_size / 1048576:.1f} MB")

    ds = write_figure_dataset(out.parent, names)
    print(f"staged {ds}  ({len(names)} figures)")

    meta_path, user = write_metadata(out.parent, out.name)
    print(f"wrote {meta_path}")
    if user.startswith("UNKNOWN"):
        print("  WARNING: Kaggle username unresolved - no ~/.kaggle/kaggle.json and no")
        print("           KAGGLE_USERNAME set. Fix that before `kaggle kernels push`.")
    else:
        print(f"  kernel   : {user}/{KERNEL_SLUG}")


if __name__ == "__main__":
    main()
