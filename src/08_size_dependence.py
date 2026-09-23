#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
08_size_dependence.py   (manuscript Figure A1, Appendix A.5)
Orientation and position resolved by area decile, per basin, to test whether the
signals depend on cubeta size.

By area decile, per basin:
  ORIENTATION  axial resultant length R (doubled long-axis azimuth) vs a Monte-Carlo
               band (p95) computed at each decile's n. R emerging above the band at
               large deciles = real, size-dependent orientation.
  POSITION     RBC (cubetas vs control) of distance to the Salado divide — non-circular
               at all sizes. RBC<0 = cubetas closer to the divide than control.
               (Slope EXCLUDED: at large size the slope raster still partly "sees" the
               cubeta itself -> circular. TPI R=2000 m also excluded from the paper
               figure: it does not scale monotonically with size; see note below.)

Interpretation is given in the manuscript Discussion, not here.

Inputs : data/covariates_cubetas.csv, data/cubetas_morphometry.gpkg,
         data/control_points.gpkg, data/cubetas.gpkg
Outputs: results/size_dependence.csv, figures/Figure_A1.{png,pdf}
Project: cubetas-morphometry | Python 3.11
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import mannwhitneyu

from lib import config as cfg
CUB_GPKG, CUB_LAYER = cfg.DATA / "cubetas.gpkg", cfg.CUBETAS_LAYER
MORPH_GPKG, MORPH_LAYER = cfg.DATA / "cubetas_morphometry.gpkg", "cubetas_morphometry"
CUB_CSV   = cfg.DATA / "covariates_cubetas.csv"
CTRL_GPKG = cfg.DATA / "control_points.gpkg"
OUT_CSV   = cfg.RESULTS / "size_dependence.csv"

POS_VARS = ["dist_divide"]      # distance to divide: non-circular at all sizes
AREA_TICKS = [0.1, 0.2, 0.5, 1, 2, 5]  # explicit ticks: the log range is narrow
#   (TPI R=2000 m dropped from the paper figure: it does not scale monotonically with
#    size and is not part of the causal chain; distance to the divide is the regional
#    proxy of low slope used in the manuscript. TPI remains available in covariates_cubetas.csv.)
N_DEC, N_ISO = 10, 999


def basin_field(g):
    return next((c for c in g.columns if c.lower() == cfg.BASIN_FIELD.lower()), None)


def rbc(cub, ctrl):
    a = np.asarray(cub, float); a = a[np.isfinite(a)]
    b = np.asarray(ctrl, float); b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    U, _ = mannwhitneyu(a, b, alternative="two-sided")
    return float(2 * (U / (len(a) * len(b))) - 1)


def axial_R(theta_deg):
    t = np.deg2rad(np.asarray(theta_deg, float)); t = t[np.isfinite(t)]
    if len(t) < 3:
        return np.nan
    return float(np.abs(np.mean(np.exp(1j * 2 * t))))


def isotropy_p95(n, rng):
    if n < 3:
        return np.nan
    Rs = [np.abs(np.mean(np.exp(1j * 2 * rng.uniform(0, np.pi, n)))) for _ in range(N_ISO)]
    return float(np.percentile(Rs, 95))


def main():
    cfg.ensure_dirs()
    rng = np.random.default_rng(cfg.SEED)

    cub = pd.read_csv(CUB_CSV)
    cub["dist_divide"] = cub["dist_divide_m"] / 1000.0
    ctrl = gpd.read_file(CTRL_GPKG, layer="control_points")
    ctrl["dist_divide"] = ctrl["dist_divide_m"] / 1000.0

    g = gpd.read_file(CUB_GPKG, layer=CUB_LAYER).to_crs(epsg=cfg.EPSG)
    idc = cfg.ID_FIELD
    g["area_ha"] = g.geometry.area / 1e4
    cub = cub.merge(g[[idc, "area_ha"]], on=idc, how="left")

    m = gpd.read_file(MORPH_GPKG, layer=MORPH_LAYER)
    # morphometry.csv is the single source for per-cubeta orientation. The same
    # column is in the derived layer, so it is dropped before the merge: two
    # columns of the same name would come back as theta_mom_geo_x and _y.
    if "theta_mom_geo" in m.columns:
        m = m.drop(columns="theta_mom_geo")
    shp = pd.read_csv(cfg.RESULTS / "morphometry.csv")[["cubeta_id", "theta_mom_geo"]]
    m = m.merge(shp, on="cubeta_id", how="left")
    # The second-moment axis, named explicitly and asserted rather than picked
    # from a list of candidates: the Feret exclusion zone around the grid axes
    # narrows with the number of pixels in the body, so on a size-resolved plot
    # Feret would carry a bias running in the same direction as the effect.
    tcol = "theta_mom_geo"
    assert tcol in m.columns, f"{tcol} missing; columns are {list(m.columns)}"
    assert m[tcol].notna().all(), f"{tcol} has {int(m[tcol].isna().sum())} nulls"
    print(f"[info] orientation column: {tcol}")
    cub = cub.merge(m[[idc, tcol]].rename(columns={tcol: "theta"}), on=idc, how="left")


    rows = []
    for b in cfg.BASINS:
        cq = cub[cub["cuenca"] == b].dropna(subset=["area_ha"]).copy()
        cc = ctrl[ctrl["cuenca"] == b]
        cq["dec"] = pd.qcut(cq["area_ha"].rank(method="first"), N_DEC, labels=False) + 1
        for d in range(1, N_DEC + 1):
            sub = cq[cq["dec"] == d]
            rec = dict(basin=b, decile=d, n=len(sub),
                       med_area_ha=round(sub["area_ha"].median(), 3),
                       R=round(axial_R(sub["theta"].values), 4),
                       R_iso_p95=round(isotropy_p95(len(sub), rng), 4))
            for v in POS_VARS:
                rec[f"RBC_{v}"] = round(rbc(sub[v].values, cc[v].values), 4)
            rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    print(f"[ok] {OUT_CSV}")

    pd.set_option("display.width", 170, "display.float_format", lambda v: f"{v:.4f}")
    for b in cfg.BASINS:
        print(f"\n--- {b} ---")
        print(res[res.basin == b].drop(columns="basin").to_string(index=False))

    # figure: rows = basins; left = orientation R vs decile (+ isotropy p95),
    #         right = position RBC vs decile
    plt = cfg.setup_matplotlib()
    fig, axes = plt.subplots(len(cfg.BASINS), 2, figsize=(11, 4.3 * len(cfg.BASINS)),
                             squeeze=False)
    for bi, b in enumerate(cfg.BASINS):
        df = res[res.basin == b]; x = df["med_area_ha"].values
        axR = axes[bi][0]
        axR.plot(x, df["R"], "-o", color="#8172B3", ms=4, label="orientation R")
        axR.plot(x, df["R_iso_p95"], "--", color="0.5", lw=1.2, label="isotropy p95")
        axR.set_xscale("log")
        axR.set_xticks(AREA_TICKS)
        axR.set_xticklabels([str(t) for t in AREA_TICKS])
        axR.minorticks_off()
        axR.set_xlabel("cubeta area (ha, log scale; decile medians)")
        axR.set_ylabel("axial R (long-axis orientation)")
        axR.annotate(cfg.basin_label(b), (0.03, 0.92), xycoords="axes fraction", fontsize=11, weight="bold")
        if bi == 0:
            axR.legend(fontsize=8, loc="upper center")

        axP = axes[bi][1]
        axP.axhline(0, color="0.6", lw=0.8)
        axP.plot(x, df["RBC_dist_divide"], "-o", color="#C44E52", ms=4,
                 label="distance to divide")
        axP.set_xscale("log")
        axP.set_xticks(AREA_TICKS)
        axP.set_xticklabels([str(t) for t in AREA_TICKS])
        axP.minorticks_off()
        axP.set_xlabel("cubeta area (ha, log scale; decile medians)")
        axP.set_ylabel("RBC vs control (< 0 = closer to divide)")
        axP.annotate(cfg.basin_label(b), (0.03, 0.92), xycoords="axes fraction", fontsize=11, weight="bold")
        if bi == 0:
            axP.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    cfg.save_figure(fig, "Figure_A1")
    plt.close(fig)
    print("\n[ok] figures/Figure_A1.png (+ .pdf)")


if __name__ == "__main__":
    main()
