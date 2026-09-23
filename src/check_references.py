#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_references.py   (verification, not part of the pipeline)

Finds broken references WITHOUT running anything, so that they do not surface one
at a time as tracebacks halfway through the pipeline.

Three classes of problem, each of which bit this repository at some point:

  1. Constants asked of config that config does not define. A constant removed
     from config while a script still uses it stays invisible until that script
     runs.

  2. Column names that a rename touched in one place and not another. A name
     built by string interpolation does not appear in a grep for the full name,
     so a rename can update where a column is read and miss where it is built.

  3. Artefacts a script reads that nothing in the pipeline writes and that are
     not distributed in data/.

It is deliberately noisy: it reports candidates, not certainties, because a
string literal is not always a column name. Read it as a list to check, not as a
list of errors.

Run it from src/, like any other script.
"""
import re
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from lib import config as cfg

SCRIPTS = sorted(p for p in SRC.glob("*.py") if re.match(r"^\d\d_", p.name))
print(f"[info] {len(SCRIPTS)} numbered scripts in {SRC}")

problems = []

# =====================================================================
# 1. constants asked of config
# =====================================================================
print("\n" + "=" * 90)
print("1. CONSTANTS ASKED OF config")
print("=" * 90)
defined = {n for n in dir(cfg) if not n.startswith("_")}
used = {}
for p in SCRIPTS + [SRC / "lib" / "vonmises.py"]:
    if not p.exists():
        continue
    txt = p.read_text(encoding="utf-8")
    for m in re.finditer(r"\bcfg\.([A-Za-z_][A-Za-z0-9_]*)", txt):
        used.setdefault(m.group(1), set()).add(p.name)

for name in sorted(used):
    if name not in defined:
        who = ", ".join(sorted(used[name]))
        print(f"  BROKEN  cfg.{name:22s} <- {who}")
        problems.append(f"cfg.{name} does not exist; asked for by {who}")
unused = sorted(n for n in defined
                if n.isupper() and n not in used and n not in ("Path",))
print(f"  [ok] {len(used) - len(problems)} constants resolved")
if unused:
    print(f"  [info] defined but never used: {', '.join(unused)}")
    print("         Some of these record how a layer in data/ was baked in prep/,")
    print("         rather than being read by the pipeline itself.")

# =====================================================================
# 2. column names built by interpolation
# =====================================================================
print("\n" + "=" * 90)
print("2. COLUMN NAMES BUILT BY STRING INTERPOLATION")
print("=" * 90)
print("  An f-string that assembles a column name is invisible to a grep for the")
print("  full name, so a rename can update one end of the pair and not the other.\n")
for p in SCRIPTS:
    txt = p.read_text(encoding="utf-8")
    hits = re.findall(r'[\[\(]\s*f"([A-Za-z_][A-Za-z0-9_]*)\{', txt)
    lits = set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]{3,})"', txt))
    for prefix in sorted(set(hits)):
        cand = sorted(l for l in lits if l.startswith(prefix))
        print(f"  {p.name:30s} builds  {prefix}<...>")
        if cand:
            print(f"  {'':30s} and reads literally: {', '.join(cand)}")
            print(f"  {'':30s} >> check that the suffix matches")

# =====================================================================
# 3. columns read from the published CSVs
# =====================================================================
print("\n" + "=" * 90)
print("3. COLUMNS READ FROM results/")
print("=" * 90)
avail = {}
for f in sorted(cfg.RESULTS.glob("*.csv")):
    try:
        avail[f.name] = set(pd.read_csv(f, nrows=1).columns)
    except Exception as e:
        print(f"  [warn] {f.name}: {e}")

for p in SCRIPTS:
    txt = p.read_text(encoding="utf-8")
    for m in re.finditer(r'cfg\.RESULTS\s*/\s*"([^"]+\.csv)"\)?\s*\)?\[\[([^\]]+)\]\]',
                         txt):
        fn, cols = m.group(1), re.findall(r'"([^"]+)"', m.group(2))
        if fn not in avail:
            print(f"  {p.name}: reads {fn}, which does not exist yet")
            continue
        miss = [c for c in cols if c not in avail[fn]]
        if miss:
            print(f"  BROKEN  {p.name}: {fn} has no {miss}")
            problems.append(f"{p.name} reads {miss} from {fn}")
        else:
            print(f"  [ok]    {p.name}: {fn} -> {cols}")

# =====================================================================
# 4. columns present in the layers of data/
# =====================================================================
print("\n" + "=" * 90)
print("4. COLUMNS OF THE LAYERS IN data/")
print("=" * 90)
try:
    import geopandas as gpd
    for g in sorted(cfg.DATA.glob("*.gpkg")):
        try:
            cols = list(gpd.read_file(g, rows=1).columns)
            print(f"  {g.name:28s} {', '.join(c for c in cols if c != 'geometry')}")
        except Exception as e:
            print(f"  {g.name}: {e}")
except ImportError:
    print("  geopandas not available")

# =====================================================================
# 5. which script writes each artefact
# =====================================================================
print("\n" + "=" * 90)
print("5. ARTEFACTS IN results/ AND WHAT WRITES THEM")
print("=" * 90)
writers = {}
for p in SCRIPTS:
    txt = p.read_text(encoding="utf-8")
    consts = dict(re.findall(r'^([A-Z_][A-Z0-9_]*)\s*=\s*cfg\.RESULTS\s*/\s*"([^"]+)"',
                             txt, re.M))
    for m in re.finditer(r'(\w+)\.to_csv\(\s*([A-Za-z_][A-Za-z0-9_]*)', txt):
        target = m.group(2)
        if target in consts:
            writers.setdefault(consts[target], set()).add(p.name)
    for m in re.finditer(r'to_csv\(\s*cfg\.RESULTS\s*/\s*"([^"]+)"', txt):
        writers.setdefault(m.group(1), set()).add(p.name)

for f in sorted(avail):
    who = ", ".join(sorted(writers.get(f, []))) or "NOTHING in src/"
    flag = "     " if who != "NOTHING in src/" else "  !! "
    print(f"{flag}{f:36s} <- {who}")

print("""
  On the blanks: this detector only recognises a writer when the call is a bare
  name followed by .to_csv. Scripts that chain it, as in
  pd.DataFrame(rows).to_csv(OUT_CSV), are missed. Open the script before treating
  a blank as an orphan; covariates_cubetas.csv, in data/, is an input baked in
  prep/ and genuinely has no writer here.""")

# =====================================================================
print("\n" + "=" * 90)
if problems:
    print(f"{len(problems)} PROBLEMS")
    for s in problems:
        print("  " + s)
else:
    print("No broken references detected.")
print("=" * 90)
print("""
What this check cannot see, and has to be found by running:
  - column names assembled inside a loop from a variable
  - columns a script creates on the fly in a layer
  - errors of content: a name that exists but holds something else
""")
