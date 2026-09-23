#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
07_drainage.py   (manuscript Figure 4, Section 4.3)
Are cubetas tied to the LOW-ORDER drainage (headwater cañadas) and away from the
larger collectors? Distance from each cubeta to the drainage network PRUNED to
channels of contributing area >= X, across an increasing series of thresholds X.

  RBC (cubetas vs control), signed so > 0 = cubetas CLOSER than control, < 0 = farther.
  The curve falls from strongly positive (cubetas hug the dense low-order network) to
  negative (cubetas avoid the large collectors). The crossover X* is the contributing
  area beyond which cubetas start to AVOID the channels.

Size reading: the threshold curve for the smallest vs largest size tercile, to see
whether X* shifts with cubeta size (consistent with the size dependence in 08).

Network = data/channel_points.csv: channel points carrying their contributing
area, extracted from the LiDAR-DEM flow accumulation (a derived product of
L. Migone; see prep/). Only the point coordinates and the contributing area are
needed, since distances are computed with a k-d tree, never along the lines.
The distributed set is pruned to contributing area >= 0.2 km2, the smallest
threshold this script evaluates, so the RBC curve and X* are unchanged. Terminology:
CONTRIBUTING AREA (area de aporte), never "caudal"/discharge — drainage area (km2),
not a modelled discharge.

Inputs : data/channel_points.csv, data/cubetas.gpkg, data/control_points.gpkg
Outputs: results/drainage.csv, figures/Figure_4.{png,pdf}
Project: cubetas-morphometry | Python 3.11
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu

from lib import config as cfg
FLOW_CSV   = cfg.CHANNELS_CSV
CUB_GPKG, CUB_LAYER = cfg.DATA / "cubetas.gpkg", cfg.CUBETAS_LAYER
CTRL_GPKG  = cfg.DATA / "control_points.gpkg"
OUT_CSV    = cfg.RESULTS / "drainage.csv"

AREA_COL   = "contrib_area_km2"   # contributing area (km2), never discharge
THRESHOLDS = np.geomspace(0.2, 1500, 16)   # fixed log range, comparable across basins
MIN_PTS    = 30                             # min channel points to evaluate a threshold


def basin_field(g):
    return next((c for c in g.columns if c.lower() == cfg.BASIN_FIELD.lower()), None)


def rbc(d_cub, d_ctrl):
    """RBC signed so > 0 = cubetas at SMALLER distance (closer) than control."""
    a = np.asarray(d_cub, float); a = a[np.isfinite(a)]
    b = np.asarray(d_ctrl, float); b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    U, _ = mannwhitneyu(a, b, alternative="two-sided")   # U ~ P(a > b)
    return float(1 - 2 * U / (len(a) * len(b)))


def crossover(x, y):
    """First X where the RBC curve crosses 0 from + to - (log-interpolated)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    for i in range(len(y) - 1):
        if y[i] >= 0 >= y[i + 1] and y[i] != y[i + 1]:
            t = y[i] / (y[i] - y[i + 1])
            return float(np.exp(np.log(x[i]) + t * (np.log(x[i + 1]) - np.log(x[i]))))
    return None


def curve(cub_xy, ctrl_xy, ch_xy, ch_area):
    """RBC vs contributing-area threshold for one cubeta set."""
    xs, rs = [], []
    for X in THRESHOLDS:
        m = ch_area >= X
        if m.sum() < MIN_PTS:
            break
        tree = cKDTree(ch_xy[m])
        d_cub = tree.query(cub_xy)[0]
        d_ctrl = tree.query(ctrl_xy)[0]
        xs.append(X); rs.append(rbc(d_cub, d_ctrl))
    return np.array(xs), np.array(rs)


def main():
    cfg.ensure_dirs()

    # Plain table: x_m, y_m are already in EPSG:5347, so there is nothing to project.
    flow = pd.read_csv(FLOW_CSV)
    assert {'x_m', 'y_m', AREA_COL}.issubset(flow.columns), \
        f"{FLOW_CSV.name} must have x_m, y_m and {AREA_COL}; has {list(flow.columns)}"
    bf_f = basin_field(flow) or "cuenca"
    cub = gpd.read_file(CUB_GPKG, layer=CUB_LAYER).to_crs(epsg=cfg.EPSG)
    bf_c = basin_field(cub) or "cuenca"
    cen = cub.geometry.centroid
    cub = cub.assign(_x=cen.x.values, _y=cen.y.values, area_ha=cub.geometry.area / 1e4)
    ctrl = gpd.read_file(CTRL_GPKG, layer="control_points").to_crs(epsg=cfg.EPSG)

    rows, curves, xstar = [], {}, {}
    print("=" * 70)
    print("DRAINAGE — cubeta-channel association by contributing area")
    print("  RBC > 0 = cubetas closer to channels >= X ; < 0 = farther")
    print("=" * 70)
    for b in cfg.BASINS:
        fb = flow[flow[bf_f] == b]
        ch_xy = np.c_[fb["x_m"].values, fb["y_m"].values]
        ch_area = fb[AREA_COL].values
        cb = cub[cub[bf_c] == b]
        cub_xy = np.c_[cb["_x"].values, cb["_y"].values]
        cc = ctrl[ctrl["cuenca"] == b]
        ctrl_xy = np.c_[cc.geometry.x.values, cc.geometry.y.values]

        # main curve (all cubetas)
        xs, rs = curve(cub_xy, ctrl_xy, ch_xy, ch_area)
        Xc = crossover(xs, rs)
        curves[(b, "all")] = (xs, rs); xstar[(b, "all")] = Xc
        for X, r in zip(xs, rs):
            rows.append(dict(basin=b, subset="all", threshold_km2=round(X, 3), RBC=round(r, 4)))
        print(f"\n[{b}] all cubetas (n={len(cb)}): X* = "
              + (f"{Xc:.2f} km2" if Xc else "no crossing"))

        # size reading: smallest vs largest tercile
        q1, q2 = cb["area_ha"].quantile([1/3, 2/3])
        for label, mask in [("small_tercile", cb["area_ha"] <= q1),
                            ("large_tercile", cb["area_ha"] >= q2)]:
            sxy = np.c_[cb.loc[mask, "_x"].values, cb.loc[mask, "_y"].values]
            xs_s, rs_s = curve(sxy, ctrl_xy, ch_xy, ch_area)
            Xc_s = crossover(xs_s, rs_s)
            curves[(b, label)] = (xs_s, rs_s); xstar[(b, label)] = Xc_s
            for X, r in zip(xs_s, rs_s):
                rows.append(dict(basin=b, subset=label, threshold_km2=round(X, 3), RBC=round(r, 4)))
            print(f"      X* {label} (n={int(mask.sum())}): "
                  + (f"{Xc_s:.2f} km2" if Xc_s else "no crossing"))

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\n[ok] {OUT_CSV}")

    # figure: (a) all cubetas both basins; (b) size terciles per basin
    plt = cfg.setup_matplotlib()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for b in cfg.BASINS:
        xs, rs = curves[(b, "all")]
        axA.plot(xs, rs, "-o", color=cfg.BASIN_COLORS.get(b), ms=4,  label=cfg.basin_label(b))
        Xc = xstar[(b, "all")]
        if Xc:
            axA.axvline(Xc, color=cfg.BASIN_COLORS.get(b), ls="--", lw=1, alpha=0.6)
    axA.axhline(0, color="0.5", ls=":", lw=1)
    axA.set_xscale("log")
    axA.set_xlabel("contributing-area threshold X — channels >= X (km$^2$)")
    axA.set_ylabel("RBC (cubetas vs control)\n> 0 = closer   ·   < 0 = farther")
    axA.annotate("all cubetas", (0.03, 0.05), xycoords="axes fraction", fontsize=9)
    axA.legend(frameon=False)

    for b in cfg.BASINS:
        for label, ls in [("small_tercile", ":"), ("large_tercile", "-")]:
            xs, rs = curves[(b, label)]
            axB.plot(xs, rs, ls, color=cfg.BASIN_COLORS.get(b), lw=1.8,
                     label=f"{cfg.basin_label(b)} {label.split('_')[0]}")
    axB.axhline(0, color="0.5", ls=":", lw=1)
    axB.set_xscale("log")
    axB.set_xlabel("contributing-area threshold X — channels >= X (km$^2$)")
    axB.annotate("by size (small vs large tercile)", (0.03, 0.05),
                 xycoords="axes fraction", fontsize=9)
    axB.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    cfg.save_figure(fig, "Figure_4")
    plt.close(fig)
    print("[ok] figures/Figure_4.png (+ .pdf)")


if __name__ == "__main__":
    main()
