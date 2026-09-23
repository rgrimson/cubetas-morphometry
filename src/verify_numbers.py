#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_numbers.py   (verification, not part of the pipeline)

Checks every figure quoted in the manuscript against the file that supports it.
Run it after the whole pipeline, and again on a clean clone before tagging.

Each check names the section of the paper where the number appears, so a mismatch
points at the sentence that has to change, not merely at a file. Quantities that
cannot be read off a table directly, such as the consecutive-radius runs of the
Ripley curve or the crossover of the drainage curve, are recomputed here from the
published series.

The reference values below are those of the submitted manuscript. If a check
fails, either the pipeline changed or the manuscript is out of date; the printed
line says which sentence to look at.

Run it from src/, like any other script.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import config as cfg

RES = cfg.RESULTS
ok = fail = skip = 0
issues = []


def check(where, label, got, want, tol=0.0015):
    """Compare one number against the manuscript."""
    global ok, fail, skip
    if got is None or (isinstance(got, float) and not np.isfinite(got)):
        print(f"  --  {where:14s} {label:44s} no data")
        skip += 1
        return
    good = abs(float(got) - float(want)) <= tol
    print(f"  {'OK ' if good else 'NO '} {where:14s} {label:44s} "
          f"paper {want:<10} repo {round(float(got), 4)}")
    if good:
        ok += 1
    else:
        fail += 1
        issues.append(f"{where}: {label}  paper {want}  repo {round(float(got), 4)}")


def one(df, **kw):
    """The single row matching all keyword filters, or None."""
    m = pd.Series(True, index=df.index)
    for k, v in kw.items():
        m &= (df[k] == v)
    s = df[m]
    return s.iloc[0] if len(s) == 1 else None


def runs(mask):
    """Longest run of consecutive True values."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


print("=" * 96)
print("THE FIGURES OF THE MANUSCRIPT, CHECKED AGAINST THE REPOSITORY")
print("=" * 96)

# ----------------------------------------------------------- census size
m = pd.read_csv(RES / "morphometry.csv")
vc = m[cfg.BASIN_FIELD].value_counts()
print("\n-- Section 3.2, size of the census")
check("3.2", "n total", len(m), 3210, 0.5)
check("3.2", "n MRB", vc.get("CMR"), 1779, 0.5)
check("3.2", "n RRB", vc.get("CRR"), 1431, 0.5)

# ------------------------------------------------------------- Table 1
print("\n-- Table 1, morphometric medians")
T1 = {"S_ha": (0.50, 0.66, 0.55), "P_m": (290.1, 319.7, 302.4),
      "a_m": (100.0, 109.2, 104.2), "b_m": (66.8, 80.0, 70.0),
      "Dm_m": (79.7, 92.0, 84.0), "IC": (1.11, 1.10, 1.10),
      "E": (1.37, 1.25, 1.30)}
for var, (cmr, crr, tot) in T1.items():
    if var not in m.columns:
        continue
    check("Table 1", f"{var} median MRB",
          m.loc[m[cfg.BASIN_FIELD] == "CMR", var].median(), cmr, 0.06)
    check("Table 1", f"{var} median RRB",
          m.loc[m[cfg.BASIN_FIELD] == "CRR", var].median(), crr, 0.06)
    check("Table 1", f"{var} median total", m[var].median(), tot, 0.06)

# -------------------------------------------------------- orientation R
print("\n-- Section 4.2 and Table 2, orientation resultants")
o = pd.read_csv(RES / "orientation_R.csv")
for (b, s, mth, R, az, lab) in [
        ("ALL", "all", "feret",       0.130, None, "R census, maximum-length axis"),
        ("ALL", "all", "moments",     0.140, None, "R census, second-moment axis"),
        ("CMR", "all", "feret",       0.100, None, "R MRB, maximum-length axis"),
        ("CMR", "all", "moments",     0.102, None, "R MRB, second-moment axis"),
        ("CRR", "all", "feret",       0.174, None, "R RRB, maximum-length axis"),
        ("CRR", "all", "moments",     0.191, None, "R RRB, second-moment axis"),
        ("CRR", "largest50", "feret", 0.564, 23.0, "matched protocol, RRB"),
        ("CMR", "largest50", "feret", 0.318, 28.0, "matched protocol, MRB")]:
    r = one(o, basin=b, subset=s, method=mth)
    if r is None:
        print(f"  --  4.2            {lab}: row missing")
        skip += 1
        continue
    check("4.2", lab, r.R, R, 0.0015)
    if az is not None:
        check("4.2", lab + " (azimuth)", r.mean_dir, az, 0.6)
r = one(o, basin="ALL", subset="all", method="feret")
if r is not None:
    check("4.2", "isotropy band of the census", r.iso_p95, 0.030, 0.004)

# ------------------------------------------------------------- Table A1
print("\n-- Table A1 and Section 4.2, modality (second-moment axis, unfiltered)")
t = pd.read_csv(RES / "TableA1_modality.csv")
mo = t[(t.estimator == "moments") & (t["filter"] == "no filter")]
for cl, R in [("<1 ha", 0.0966), ("1-2 ha", 0.1540), (">=2 ha", 0.3155)]:
    r = one(mo, basin="ALL", size_class=cl)
    if r is not None:
        check("4.2", f"R, area class {cl}", r.R, R, 0.0015)
r = one(mo, basin="ALL", size_class="all")
if r is not None:
    check("A.1", "BIC VM1, census", r.BIC_VM1, 11688.3, 0.6)
    check("A.1", "BIC VM1U, census", r.BIC_VM1U, 11671.2, 0.6)
    check("A.1", "BIC VM2, census", r.BIC_VM2, 11683.0, 0.6)
    check("Table A1", "mu1, census", r.mu1, 19.5, 0.2)
    check("Table A1", "kappa1, census", r.kappa1, 2.15, 0.05)
    check("4.2", "weight of the census mode", r.w1, 0.199, 0.005)
    print(f"      selection: BIC -> {r.model} (dBIC {r.dBIC_next}), "
          f"AIC -> {r.get('model_aic', '?')}")

# The 29 degree claim appears in Sections 4.2 and 5.1, in Table 3 and in
# Appendix A.2. It is the load-bearing number of the anti-aeolian argument.
if "min_dist_to_ref_deg" in mo.columns:
    mn = float(mo.min_dist_to_ref_deg.min())
    arg = mo.loc[mo.min_dist_to_ref_deg.idxmin()]
    print(f"\n  >>> closest fitted component to the 61 deg azimuth: {mn:.1f} deg "
          f"({arg.basin}/{arg.size_class}, mu {arg.mu1})")
    print("      The manuscript states 'no fitted mode closer than 29 degrees'.")
    print(f"      {'HOLDS' if mn >= 29 else 'DOES NOT HOLD: the threshold must be revised'}")
    if mn >= 29:
        ok += 1
    else:
        fail += 1
        issues.append(f"A.2: the 29 deg threshold does not hold; the minimum is {mn:.1f}")

nvm2 = int((mo.model == "VM2").sum())
print(f"\n  VM2 selected by BIC in {nvm2} of {len(mo)} cells (the paper says 1 of 12;")
print("  the count here includes an upper-decile row per basin, not one of the twelve)")
if "model_aic" in mo.columns:
    print(f"  VM2 selected by AIC in {int((mo.model_aic == 'VM2').sum())} "
          f"of {len(mo)} (Appendix A.1 says 8 of 12)")

# ---------------------------------------------------------- coupling
print("\n-- Section 4.4 and Figure 7, cubeta-cañada coupling")
v = pd.read_csv(RES / "valley_coupling.csv")
q = v[(v.group == "ALL/all") & (v.estimator == "moments")
      & (v.kind == "consistency_quartile")]
for lab, R in [("Q1", 0.067), ("Q2", 0.145), ("Q3", 0.196), ("Q4", 0.228)]:
    r = one(q, stratum=lab)
    if r is not None:
        check("4.4", f"R(theta_C - theta_V), quartile {lab}", r.R, R, 0.002)
for lab, R in [("50-100 m", 0.143), ("100-200 m", 0.162), (">200 m", 0.294)]:
    r = one(v[(v.group == "ALL/all") & (v.estimator == "moments")
              & (v.kind == "length_bin")], stratum=lab)
    if r is not None:
        check("4.4", f"R(theta_C - theta_V), {lab}", r.R, R, 0.002)

q4 = v[(v.kind == "consistency_quartile") & (v.estimator == "moments")
       & (v.stratum == "Q4")]
q1 = v[(v.kind == "consistency_quartile") & (v.estimator == "moments")
       & (v.stratum == "Q1")]
print(f"  Q4 above the permutation band: {int(q4.above.sum())} of {len(q4)} "
      f"(the paper says 12 of 12)")
print(f"  Q1 above the permutation band: {int(q1.above.sum())} of {len(q1)} "
      f"(the paper says 1 of 12)")

# ---------------------------------------------------------- drainage
print("\n-- Section 4.3, drainage")
d = pd.read_csv(RES / "drainage.csv")
for b, xs in [("CMR", 1.46), ("CRR", 1.30)]:
    s = d[(d.basin == b) & (d.subset == "all")].sort_values("threshold_km2")
    x, y = s.threshold_km2.values, s.RBC.values
    cross = None
    for i in range(len(y) - 1):
        if y[i] >= 0 >= y[i + 1] and y[i] != y[i + 1]:
            f = y[i] / (y[i] - y[i + 1])
            cross = float(np.exp(np.log(x[i]) + f * (np.log(x[i + 1]) - np.log(x[i]))))
            break
    check("4.3", f"crossover X* {b}", cross, xs, 0.05)

# ------------------------------------------------------ slope and divide
print("\n-- Sections 4.3 and 4.5, slope and distance to the divide")
tp = pd.read_csv(RES / "topographic_position.csv")
for var, rbc in [("slope_s50", -0.41), ("slope_s100", -0.28),
                 ("slope_s200", -0.20), ("slope_s500", -0.12)]:
    r = one(tp, basin="ALL", variable=var)
    if r is not None:
        check("4.3", f"RBC {var}", r.RBC, rbc, 0.006)
for b, rbc in [("CRR", -0.20), ("CMR", -0.06), ("ALL", -0.13)]:
    r = one(tp, basin=b, variable="dist_divide_km")
    if r is not None:
        check("4.5", f"RBC distance to the divide, {b}", r.RBC, rbc, 0.006)

# ------------------------------------------------------------- Ripley
print("\n-- Section 4.5, Ripley's L")
rp = pd.read_csv(RES / "ripley_L.csv")
EXPECTED_RUNS = {"CRR": (24, 0), "CMR": (8, 11)}
for b, area, n in [("CMR", 967.0, 1779), ("CRR", 925.9, 1431)]:
    s = rp[rp.window == b].sort_values("r_m")
    if not len(s):
        continue
    check("3.2", f"window {b} (km2)", s.area_km2.iloc[0], area, 0.06)
    check("4.5", f"n {b}", s.n.iloc[0], n, 0.5)
    ex = runs((s.L_obs - s.r_m).values > (s.env_hi - s.r_m).values)
    df_ = runs((s.L_obs - s.r_m).values < (s.env_lo - s.r_m).values)
    e_ex, e_df = EXPECTED_RUNS[b]
    good = (ex == e_ex) and (df_ == e_df)
    print(f"  {'OK ' if good else 'NO '} 4.5            {b}: excess over {ex} "
          f"consecutive radii, deficit over {df_}   paper: {e_ex} and {e_df}")
    if good:
        ok += 1
    else:
        fail += 1
        issues.append(f"4.5: runs in {b}, paper {e_ex}/{e_df}, repo {ex}/{df_}")

# ------------------------------------------------------- size dependence
print("\n-- Appendix A.5, area deciles")
sd = pd.read_csv(RES / "size_dependence.csv")
for b, R in [("CRR", 0.4868), ("CMR", 0.2724)]:
    s = sd[(sd.basin == b) & (sd.decile == 10)]
    if len(s) == 1:
        check("A.5", f"R, tenth decile {b}", s.R.iloc[0], R, 0.002)

# --------------------------------------------- Appendix A.4 and elongation
print("\n-- Appendix A.4 and Section 4.4, ring and estimator controls")
ec = RES / "coupling_estimator_control.csv"
if ec.exists():
    c = pd.read_csv(ec)
    p = c[(c.basin == "ALL") & (c.size_class == "all") & (c.estimator == "feret")
          & (c.ring == "fixed")]
    for lab, R in [("Q1", 0.065), ("Q2", 0.146), ("Q3", 0.178), ("Q4", 0.227)]:
        r = one(p, q=lab)
        if r is not None:
            check("A.4", f"maximum-length axis, quartile {lab}", r.R_delta, R, 0.002)
el = RES / "coupling_elongation.csv"
if el.exists():
    e = pd.read_csv(el)
    fx = e[e.ring == "fixed"]
    lo, hi = float(fx.RBC.min()), float(fx.RBC.max())
    good = abs(lo + 0.07) < 0.01 and abs(hi - 0.12) < 0.01
    print(f"  {'OK ' if good else 'NO '} 4.4            elongation RBC, fixed ring"
          f"{'':18s} paper -0.07 to +0.12   repo {lo:+.3f} to {hi:+.3f}")
    ok, fail = (ok + 1, fail) if good else (ok, fail + 1)

# ----------------------------------------------------------- summary
print("\n" + "=" * 96)
print(f"{ok} verified, {fail} failed, {skip} without data")
if issues:
    print("\nDifferences:")
    for s in issues:
        print("  " + s)
    print("\nEach line points at the sentence of the manuscript to revise.")
print("=" * 96)
