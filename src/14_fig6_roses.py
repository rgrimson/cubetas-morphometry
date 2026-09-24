#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14_fig6_roses.py   (manuscript Figure 6)

The three axial directional signals of the paper, side by side, per basin:

  (a) CUBETA ORIENTATION (theta_C) — azimuth of the elongation axis of each
      body, from its second-order area moments (theta_C^M), read from
      morphometry.csv. The second-moment estimator is used because
      the modality decision is taken on it (Section 3.5, Appendix A.3).
  (b) CUBETA ALIGNMENT (theta_N) — axial bearing from each cubeta centroid to
      its nearest neighbour, computed directly from cubetas.gpkg.
  (c) VALLEY ORIENTATION FIELD (theta_V) — structure tensor of the smoothed
      relief (Bigün, Granlund & Wiklund 1991), read from tensor_cubetas.gpkg,
      which is baked outside this repository because the DEMs it derives
      from are not distributed (see prep/ and the README). Only cubetas with
      a ring-averaged tensor coherence above COH_MIN are used.

The three are independent: theta_V is read with the cubetas masked out of the
elevation model, theta_N is purely geometric, theta_C is the shape of the body.
The panels go from the landform outwards to the terrain, the order used in
Section 3.5 and in Figure 2.

All are axial (0-180) and are shown as mirrored roses (10-degree sectors) with
a common radial scale per basin, so the three can be compared visually.
Each panel reports n, the circular mean and the resultant length R. No burned-in title (caption
in the manuscript), consistent with the other figures.

Inputs : data/cubetas.gpkg, data/tensor_cubetas.gpkg,
         results/morphometry.csv
Output : figures/Figure_6[_RRB].{png,pdf}
Project: cubetas-morphometry | Python 3.11 | geopandas, pandas, scipy, matplotlib
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

# make the sibling config.py importable regardless of the current directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import config as cfg
CUB_GPKG    = cfg.DATA / "cubetas.gpkg"
CUB_LAYER   = cfg.CUBETAS_LAYER
TENSOR_GPKG = cfg.DATA / "tensor_cubetas.gpkg"
TENSOR_LAYER = "tensor_cubetas"
SHAPE_CSV    = cfg.RESULTS / "morphometry.csv"
THETA_C      = "theta_mom_geo"      # second-moment axis; see Appendix A.3

COH_MIN   = cfg.CONSISTENCY_TH   # ring-averaged tensor coherence; retains 95 %
                                 # of the census. NOT ring_consistency, which is
                                 # the axial resultant used to stratify Figure 7.
SECTOR    = 10           # rose sector width (degrees)
SHOW_MEAN = True         # thin axial line at the circular mean

# which basins to plot: cfg.BASINS for the 2x2 (MRB+RRB), or a subset,
# e.g. ["CRR"] for an RRB-only 1x2 version. Layout adapts automatically.
PLOT_BASINS = ["CRR"]

# Paul Tol scheme, colourblind-safe, shared with Figure 2 through config so
# the same signal is the same colour in both figures.
COL_CUBETA = cfg.SIGNAL_COLORS["theta_C"]
COL_VECINA = cfg.SIGNAL_COLORS["theta_N"]
COL_CANADA = cfg.SIGNAL_COLORS["theta_V"]

FIG_DIR = getattr(cfg, "FIGURES", cfg.BASE / "figures")


def basin_field(gdf):
    return next((c for c in gdf.columns if c.lower() == cfg.BASIN_FIELD.lower()), "cuenca")


def mean_axial(deg):
    deg = np.asarray(deg, float); deg = deg[np.isfinite(deg)]
    a2 = 2 * np.deg2rad(deg)
    C, S = np.mean(np.cos(a2)), np.mean(np.sin(a2))
    return (np.degrees(0.5 * np.arctan2(S, C))) % 180, float(np.hypot(C, S))


def frac_hist(az_axial, binw=SECTOR):
    """mirrored (0-360) fraction histogram of an axial sample."""
    az = az_axial[np.isfinite(az_axial)] % 180
    edges = np.arange(0, 180 + binw, binw)
    cnt, _ = np.histogram(az, bins=edges)
    frac = cnt / max(cnt.sum(), 1)
    centers = np.deg2rad(np.arange(0, 360, binw) + binw / 2)
    return centers, np.deg2rad(binw), np.concatenate([frac, frac])


def draw_rose(ax, az_axial, color, ymax, title, sub):
    centers, width, frac2 = frac_hist(az_axial)
    ax.bar(centers, frac2, width=width * 0.95, color=color,
           edgecolor="white", linewidth=0.4, alpha=0.9)
    if SHOW_MEAN:
        m, _ = mean_axial(az_axial)
        for off in (0.0, np.pi):
            ax.plot([np.deg2rad(m) + off] * 2, [0, ymax], color="#333333",
                    lw=1.0, ls="--", alpha=0.8, zorder=5)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_ylim(0, ymax); ax.set_yticklabels([])
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["N", "", "E", "", "S", "", "W", ""], fontsize=8)
    ax.set_title(f"{title}\n{sub}", fontsize=9.5, pad=14)


def main():
    plt = cfg.setup_matplotlib()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    cub = gpd.read_file(CUB_GPKG, layer=CUB_LAYER).to_crs(epsg=cfg.EPSG)
    ten = gpd.read_file(TENSOR_GPKG, layer=TENSOR_LAYER).to_crs(epsg=cfg.EPSG)
    shp = pd.read_csv(SHAPE_CSV)[["cubeta_id", THETA_C]]
    cub = cub.merge(shp, on="cubeta_id", how="left")
    bf_c, bf_t = basin_field(cub), basin_field(ten)

    basins = [b for b in cfg.BASINS if b in PLOT_BASINS]
    nrow = len(basins)
    fig, axes = plt.subplots(nrow, 3, figsize=(12.4, 4.9 * nrow),
                             subplot_kw={"projection": "polar"}, squeeze=False)
    panel = iter("abcdefghi")
    for r, b in enumerate(basins):
        lab = cfg.basin_label(b)                             # MRB, RRB
        cb = cub[cub[bf_c] == b]

        # (a) cubeta orientation, second-moment axis
        az_cub = cb.loc[np.isfinite(cb[THETA_C]), THETA_C].values
        m_s, R_s = mean_axial(az_cub)

        # (b) nearest-neighbour azimuth (all cubetas of the basin)
        cen = cb.geometry.centroid
        xy = np.c_[cen.x.values, cen.y.values]
        _, idx = cKDTree(xy).query(xy, k=2)
        v = xy[idx[:, 1]] - xy
        az_nn = np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 180
        m_n, R_n = mean_axial(az_nn)

        # (c) valley orientation field (tensor), coherence-filtered
        tb = ten[ten[bf_t] == b]
        az_can = tb.loc[np.isfinite(tb["az_canada"]) & (tb["coh"] > COH_MIN),
                        "az_canada"].values
        m_c, R_c = mean_axial(az_can)

        # common radial scale within the basin row, so the three are comparable
        ymax = 1.06 * max(frac_hist(az_cub)[2].max(),
                          frac_hist(az_nn)[2].max(),
                          frac_hist(az_can)[2].max())

        draw_rose(axes[r, 0], az_cub, COL_CUBETA, ymax,
                  f"({next(panel)}) {lab} — cubeta orientation " + r"($\theta_C$)",
                  f"n = {len(az_cub)} · mean {m_s:.0f}° · R = {R_s:.2f}")
        draw_rose(axes[r, 1], az_nn, COL_VECINA, ymax,
                  f"({next(panel)}) {lab} — cubeta alignment " + r"($\theta_N$)",
                  f"n = {len(az_nn)} · mean {m_n:.0f}° · R = {R_n:.2f}")
        draw_rose(axes[r, 2], az_can, COL_CANADA, ymax,
                  f"({next(panel)}) {lab} — valley orientation field " + r"($\theta_V$)",
                  f"n = {len(az_can)} · mean {m_c:.0f}° · R = {R_c:.2f}")

        print(f"[{lab}] theta_C: n={len(az_cub)} mean={m_s:.1f}° R={R_s:.3f}  |  "
              f"theta_N: n={len(az_nn)} mean={m_n:.1f}° R={R_n:.3f}  |  "
              f"theta_V: n={len(az_can)} mean={m_c:.1f}° R={R_c:.3f}")

    fig.tight_layout()
    suffix = "" if len(basins) == len(cfg.BASINS) else "_" + "_".join(
        cfg.basin_label(b) for b in basins)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"Figure_6.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"\n[ok] {FIG_DIR / f'Figure_6.png'} (+ .pdf)")


if __name__ == "__main__":
    main()