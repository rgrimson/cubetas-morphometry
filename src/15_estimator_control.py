#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15_estimator_control.py   (manuscript Appendix A.3 and A.4)

Three controls that the appendices report, and that the body of the paper leans
on without reproducing:

  A. COUPLING BY ESTIMATOR AND BY RING VARIANT (Appendix A.4)
     The agreement between cubeta orientation and the valley field, by quartile
     of local consistency, computed for the two orientation estimators and for
     the two ring geometries. The fixed ring is the one used throughout; the
     scaled ring is reported because it is what the appendix discards, and the
     reason has to be auditable: a ring placed at multiples of the rectangle
     length samples the terrain further away for larger bodies, so its
     consistency correlates with body size while the fixed ring's does not.

  B. ELONGATION AGAINST CONSISTENCY (Section 4.4)
     Whether cubetas sitting in a well-defined cañada are more elongated than
     those on open flats. They are not: the cañada sets the direction of
     elongation but not its degree.

  C. FERET DEPLETION NEAR THE GRID AXES (Appendix A.3)
     The maximum Feret diameter is fixed by two boundary vertices, which cannot
     share a perpendicular coordinate exactly, so azimuths are displaced away
     from the grid axes. The displacement is about atan(1/n_px), so the depleted
     zone is wider for bodies with fewer pixels, which is why the modality
     decision is taken on the second-moment axis instead.

Inputs : data/tensor_cubetas.gpkg   (both ring variants and the centroid reading)
         results/morphometry.csv
Outputs: results/coupling_estimator_control.csv
         results/coupling_elongation.csv
         results/grid_depletion.csv

Project: cubetas-morphometry | Python 3.11
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import config as cfg

OUT_QUART = cfg.RESULTS / "coupling_estimator_control.csv"
OUT_ELONG = cfg.RESULTS / "coupling_elongation.csv"
OUT_GRID  = cfg.RESULTS / "grid_depletion.csv"

# orientation estimators compared
THETAS = {"moments": "theta_mom_geo", "feret": "theta_feret_geo"}

# valley-field variants: (label, azimuth, coherence, consistency)
VFIELD = [("fixed",  "az_canada",      "coh",             "ring_consistency"),
          ("scaled", "az_scaled_ring", "coh_scaled_ring", "scaled_consistency")]

MIN_GROUP, MIN_CELL = 80, 30
PX_BINS   = [0, 20, 40, 80, 160, np.inf]
PX_LABELS = ["<20", "20-40", "40-80", "80-160", ">160"]
AXES      = np.array([0.0, 90.0, 180.0])
TOL_WIDE, TOL_NARROW = 7.0, 3.0      # degrees from a grid axis


# --------------------------------------------------------------------- stats
def align_R(theta, az):
    """Resultant of the axial difference between two azimuth series."""
    d = np.deg2rad(2.0 * (np.asarray(theta, float) - np.asarray(az, float)))
    d = d[np.isfinite(d)]
    return float(np.hypot(np.cos(d).mean(), np.sin(d).mean())) if d.size else np.nan


def axial_diff(a, b):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 180.0
    return np.minimum(d, 180.0 - d)


_ISO = {}
_rng = np.random.default_rng(cfg.ISO_SEED)


def iso_p95(n):
    """95th percentile of R under isotropy, by Monte Carlo. Cached per n."""
    if n < 2:
        return np.nan
    if n not in _ISO:
        a = _rng.uniform(0, 2 * np.pi, size=(cfg.N_ISO, n))
        _ISO[n] = float(np.percentile(
            np.hypot(np.cos(a).mean(1), np.sin(a).mean(1)), 95))
    return _ISO[n]


def rbc(hi, lo):
    """Rank-biserial correlation, positive when `hi` takes larger values."""
    a = np.asarray(hi, float); a = a[np.isfinite(a)]
    b = np.asarray(lo, float); b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    U, _ = mannwhitneyu(a, b, alternative="two-sided")
    return float(2 * U / (len(a) * len(b)) - 1)


# ---------------------------------------------------------------------- data
def load():
    t = gpd.read_file(cfg.TENSOR_GPKG, layer=cfg.TENSOR_LAYER).drop(columns="geometry")
    m = pd.read_csv(cfg.RESULTS / "morphometry.csv")

    need_t = {c for _, a, c2, c3 in VFIELD for c in (a, c2, c3)}
    need_m = {"S_ha", "E", "a_m", "theta_mom_geo", "theta_feret_geo",
              "theta_mbr_grid", "theta_mbr_geo"}
    for col, src, name in [(need_t, t, "tensor_cubetas.gpkg"),
                           (need_m, m, "morphometry.csv")]:
        miss = col - set(src.columns)
        assert not miss, f"{name} is missing {miss}"

    dup = [c for c in t.columns if c in m.columns and c != cfg.ID_FIELD]
    if dup:
        t = t.drop(columns=dup)
    d = m.merge(t, on=cfg.ID_FIELD, how="inner", validate="1:1")
    assert not [c for c in d.columns if c.endswith(("_x", "_y"))], "merge suffixes"

    d["size_class"] = pd.cut(d.S_ha, [0, 1, 2, np.inf],
                             labels=["<1 ha", "1-2 ha", ">=2 ha"])
    d["n_px"] = d.S_ha * 100.0          # 1 ha = 100 pixels of 10 m
    print(f"[info] n={len(d)}  {dict(d[cfg.BASIN_FIELD].value_counts())}")
    return d


# ------------------------------------------------------- A. estimator control
def part_quartiles(d):
    rows = []
    for vname, acol, _, ccol in VFIELD:
        for tname, tcol in THETAS.items():
            for b in ["ALL"] + cfg.BASINS:
                d0 = d if b == "ALL" else d[d[cfg.BASIN_FIELD] == b]
                for sc in ["all", "<1 ha", "1-2 ha", ">=2 ha"]:
                    s0 = d0 if sc == "all" else d0[d0.size_class == sc]
                    s0 = s0[np.isfinite(s0[acol]) & np.isfinite(s0[ccol])
                            & np.isfinite(s0[tcol])]
                    if len(s0) < MIN_GROUP:
                        continue
                    q = pd.qcut(s0[ccol], 4, labels=False, duplicates="drop")
                    for qi in sorted(pd.unique(q.dropna())):
                        s = s0[q == qi]
                        if len(s) < MIN_CELL:
                            continue
                        rows.append(dict(
                            ring=vname, estimator=tname, basin=b, size_class=sc,
                            q=f"Q{int(qi)+1}", n=len(s),
                            cons_med=round(float(s[ccol].median()), 3),
                            S_med=round(float(s.S_ha.median()), 3),
                            a_m_med=round(float(s.a_m.median()), 1),
                            R_delta=round(align_R(s[tcol].values, s[acol].values), 4),
                            iso_p95=round(iso_p95(len(s)), 4),
                            medE=round(float(s.E.median()), 3)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_QUART, index=False)
    print(f"[ok] {OUT_QUART.name}  ({len(out)} rows)")

    print("\n--- R(theta_C - theta_V) by quartile, pooled census ---")
    p = out[(out.basin == "ALL") & (out.size_class == "all")]
    print(p.pivot_table(index=["ring", "estimator"], columns="q",
                        values="R_delta", aggfunc="first").to_string())
    return out


# ------------------------------------------------------------ B. elongation
def part_elongation(d):
    rows = []
    for vname, _, _, ccol in VFIELD:
        for b in ["ALL"] + cfg.BASINS:
            d0 = d if b == "ALL" else d[d[cfg.BASIN_FIELD] == b]
            for sc in ["all", "<1 ha", "1-2 ha", ">=2 ha"]:
                s = d0 if sc == "all" else d0[d0.size_class == sc]
                s = s[np.isfinite(s[ccol])]
                if len(s) < MIN_GROUP:
                    continue
                q = pd.qcut(s[ccol], 4, labels=False, duplicates="drop")
                hi, lo = s.E[q == q.max()], s.E[q == 0]
                rows.append(dict(
                    ring=vname, basin=b, size_class=sc, n=len(s),
                    RBC=round(rbc(hi, lo), 3),
                    medE_Q4=round(float(hi.median()), 3),
                    medE_Q1=round(float(lo.median()), 3),
                    S_Q4=round(float(s.S_ha[q == q.max()].median()), 3),
                    S_Q1=round(float(s.S_ha[q == 0].median()), 3)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_ELONG, index=False)
    print(f"\n[ok] {OUT_ELONG.name}  ({len(out)} rows)")

    fx = out[out.ring == "fixed"]
    print(f"  fixed ring:  RBC from {fx.RBC.min():+.3f} to {fx.RBC.max():+.3f}")
    print(f"  scaled ring: RBC from {out[out.ring=='scaled'].RBC.min():+.3f} "
          f"to {out[out.ring=='scaled'].RBC.max():+.3f}")
    print("  (Section 4.4 quotes the fixed-ring range; positive = Q4 more elongated)")
    return out


# ------------------------------------------------------- C. grid depletion
def part_grid(d):
    """Azimuths back in grid coordinates, to measure the depleted zone."""
    conv = (d.theta_mbr_grid - d.theta_mbr_geo + 90) % 180 - 90
    grid = {"moments": (d.theta_mom_geo + conv) % 180,
            "feret":   (d.theta_feret_geo + conv) % 180}
    d = d.assign(px_bin=pd.cut(d.n_px, PX_BINS, labels=PX_LABELS))

    rows = []
    for name, v in grid.items():
        for pb in PX_LABELS:
            m = (d.px_bin == pb) & np.isfinite(v)
            vv = v[m].values % 180.0
            if vv.size < MIN_CELL:
                continue
            dist = np.min(np.abs(vv[:, None] - AXES[None, :]), axis=1)
            rows.append(dict(
                estimator=name, px_bin=pb, n=int(vv.size),
                frac_within_7deg=round(float((dist <= TOL_WIDE).mean()), 4),
                frac_within_3deg=round(float((dist <= TOL_NARROW).mean()), 4),
                expected_7deg=round(4 * TOL_WIDE / 180.0, 4),
                expected_3deg=round(4 * TOL_NARROW / 180.0, 4)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_GRID, index=False)
    print(f"\n[ok] {OUT_GRID.name}  ({len(out)} rows)")
    print(out.to_string(index=False))
    print("  A depleted zone that narrows with pixel count is a boundary effect;\n"
          "  the second-moment axis should stay near the expected fraction.")
    return out


# ------------------------------------------------------- numbers of A.4
def appendix_numbers(d):
    print("\n" + "=" * 74)
    print("APPENDIX A.4: numbers quoted in the text")
    print("=" * 74)
    pairs = [("az_canada", "az_centroid_unmasked",
              "field as used vs unmasked centroid reading"),
             ("az_canada", "az_scaled_ring", "fixed ring vs scaled ring")]
    for a, b, lab in pairs:
        if a in d.columns and b in d.columns:
            dd = axial_diff(d[a], d[b])
            dd = dd[np.isfinite(dd)]
            print(f"  {lab:45s} median {np.median(dd):5.2f} deg | "
                  f"within 20 deg: {100*(dd <= 20).mean():.0f} %")
    print(f"  expected within 20 deg by chance             "
          f"{100*40/180:.0f} %")

    for ring, ccol in [("fixed", "ring_consistency"), ("scaled", "scaled_consistency")]:
        s = d[[ccol, "a_m"]].dropna()
        rho = s[ccol].corr(s.a_m, method="spearman")
        q = pd.qcut(s[ccol], 4, labels=False, duplicates="drop")
        areas = d.loc[s.index, "S_ha"]
        print(f"  {ring:6s} ring: rho(consistency, rectangle length) = {rho:+.3f} | "
              f"median area Q1 {areas[q==0].median():.2f} ha, "
              f"Q4 {areas[q==q.max()].median():.2f} ha")


def main():
    cfg.ensure_dirs()
    d = load()
    part_quartiles(d)
    part_elongation(d)
    part_grid(d)
    appendix_numbers(d)


if __name__ == "__main__":
    main()
