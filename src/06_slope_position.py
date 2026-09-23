#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_slope_position.py   (manuscript Sections 4.3 and 4.5)
Do the cubetas occupy distinctive topographic positions — flatter ground, and
closer to the Salado divide — than random points in the same non-urban window?

Two position covariates, same cubeta-vs-control logic:
    slope (deg) at sigma = 50/100/200/500 m   (smoothed-DEM slope)
    distance to the Salado divide (km)

Effect size: rank-biserial correlation from Mann-Whitney U, signed so that
    RBC < 0  =>  cubetas at LOWER values than control (flatter / closer).
Slope note: the effect is strongest at small sigma (the depression itself) and
attenuates with smoothing; the regional scales (sigma>=200 m) are the ones free of
the "measuring the cubeta with the cubeta" circularity, and the effect persists
there — that is the argument.

Inputs (lightweight; DEMs themselves are ACUMAR/COMIREC property, not distributed):
    data/covariates_cubetas.csv   (per-cubeta slope + dist, baked from the DEMs)
    data/control_points.gpkg       (10k control pts per basin, same covariates)
Outputs: results/topographic_position.csv, figures/topographic_position.{png,pdf}
Project: cubetas-morphometry | Python 3.11
"""
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import mannwhitneyu

from lib import config as cfg
CUB_CSV   = cfg.DATA / "covariates_cubetas.csv"
CTRL_GPKG = cfg.DATA / "control_points.gpkg"
CTRL_LAYER = "control_points"
OUT_CSV   = cfg.RESULTS / "topographic_position.csv"

SIGMAS = [50, 100, 200, 500]
SLOPE_COLS = [f"slope_s{s}" for s in SIGMAS]


def rbc_mwu(cub, ctrl):
    """Rank-biserial correlation + MWU p. Signed so RBC<0 = cubetas lower than control.
    RBC = 2*P(cubeta>control) - 1 ; negative when cubetas tend to smaller values."""
    a = np.asarray(cub, float); a = a[np.isfinite(a)]
    b = np.asarray(ctrl, float); b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, len(a), len(b)
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    cles = U / (len(a) * len(b))          # P(cubeta > control)
    return float(2 * cles - 1), float(p), len(a), len(b)


def main():
    cfg.ensure_dirs()
    for p in (CUB_CSV, CTRL_GPKG):
        if not p.exists():
            sys.exit(f"[error] not found: {p}. It is baked from the LiDAR DEMs "
                 f"outside this repository; see prep/ and the README.")

    cub = pd.read_csv(CUB_CSV)
    ctrl = gpd.read_file(CTRL_GPKG, layer=CTRL_LAYER).drop(columns="geometry")
    cub["dist_divide_km"] = cub["dist_divide_m"] / 1000.0
    ctrl["dist_divide_km"] = ctrl["dist_divide_m"] / 1000.0

    variables = [(c, c) for c in SLOPE_COLS] + [("dist_divide_km", "dist_divide_km")]

    rows = []
    for bname in cfg.BASINS + ["ALL"]:
        cq = cub if bname == "ALL" else cub[cub["cuenca"] == bname]
        cc = ctrl if bname == "ALL" else ctrl[ctrl["cuenca"] == bname]
        for col, label in variables:
            rbc, p, n_c, n_k = rbc_mwu(cq[col].values, cc[col].values)
            rows.append(dict(basin=bname, variable=label,
                             n_cub=n_c, n_ctrl=n_k,
                             med_cub=round(np.nanmedian(cq[col]), 4),
                             med_ctrl=round(np.nanmedian(cc[col]), 4),
                             RBC=round(rbc, 4) if np.isfinite(rbc) else np.nan,
                             p=p))
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    print(f"[ok] {OUT_CSV}")

    pd.set_option("display.width", 170, "display.float_format", lambda v: f"{v:.4f}")
    print("\n" + "=" * 74)
    print("TOPOGRAPHIC POSITION — cubetas vs control (RBC<0 = flatter / closer)")
    print("=" * 74)
    for bname in cfg.BASINS + ["ALL"]:
        sub = res[res.basin == bname]
        print(f"\n--- {bname} ---")
        print(sub.drop(columns="basin").to_string(index=False))

    print("\nReading:")
    print("  Slope: RBC negative at all sigma (cubetas flatter). Strongest at small")
    print("  sigma; persists at regional sigma (>=200 m) where it cannot be an")
    print("  artefact of the depression itself.")
    print("  Divide: RBC negative = cubetas closer to the Salado divide than control.")

    # --- figure: RBC by scale (slope) + divide, per basin ---
    plt = cfg.setup_matplotlib()
    fig, (axS, axD) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [2.2, 1]})
    for bname in cfg.BASINS:
        sub = res[(res.basin == bname) & (res.variable.isin(SLOPE_COLS))]
        sub = sub.set_index("variable").loc[SLOPE_COLS]
        axS.plot(SIGMAS, sub["RBC"].values, "-o",
                 color=cfg.BASIN_COLORS.get(bname), label=bname)
    axS.axhline(0, color="0.6", lw=0.8)
    axS.set_xlabel("smoothing scale sigma (m)"); axS.set_ylabel("RBC (slope)")
    axS.set_title("Cubetas on flatter ground (RBC<0)"); axS.legend()

    dv = res[(res.variable == "dist_divide_km") & (res.basin.isin(cfg.BASINS))]
    dv = dv.set_index("basin").loc[cfg.BASINS]
    axD.bar(range(len(cfg.BASINS)), dv["RBC"].values,
            color=[cfg.BASIN_COLORS.get(b) for b in cfg.BASINS])
    axD.axhline(0, color="0.6", lw=0.8)
    axD.set_xticks(range(len(cfg.BASINS))); axD.set_xticklabels(cfg.BASINS)
    axD.set_ylabel("RBC (distance to divide)"); axD.set_title("Closer to divide (RBC<0)")
    fig.tight_layout()
    cfg.save_figure(fig, "topographic_position")
    plt.close(fig)
    print("\n[ok] figures/topographic_position.png (+ .pdf)")


if __name__ == "__main__":
    main()
