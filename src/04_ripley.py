#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_ripley.py   (manuscript Figure 8, Section 4.5)
Ripley's L (homogeneous) for the cubeta point pattern, on the two non-urban windows
built by intersecting the study-area layers:

    CMR      = sector_no_urbano ∩ cuenca CMR
    CRR      = sector_no_urbano ∩ cuenca CRR

Points = cubeta centroids inside each window. CSR envelope = N_SIM homogeneous
Poisson simulations confined to the same window; observed and simulated share the
same edge-uncorrected estimator, so the Monte-Carlo envelope absorbs the edge bias.
Intensity is homogeneous: lambda = n / |W|.

CAVEAT (for interpretation, not for this script): a homogeneous L cannot separate
first-order intensity (terrain control) from second-order interaction. That is the
job of the inhomogeneous L (script 05).

Inputs : data/cubetas.gpkg, data/area_de_estudio.gpkg
Outputs: results/ripley_L.csv, figures/Figure_8.{png,pdf}
Project: cubetas-morphometry | Python 3.11
"""
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely import union_all
from scipy.spatial import cKDTree

from lib import config as cfg
CUB_GPKG  = cfg.DATA / "cubetas.gpkg"
CUB_LAYER = cfg.CUBETAS_LAYER
AREA_GPKG = cfg.AREA_GPKG
OUT_CSV   = cfg.RESULTS / "ripley_L.csv"


def L_of(coords, radii, area):
    n = len(coords)
    if n < 2:
        return np.full(len(radii), np.nan)
    tree = cKDTree(coords)
    counts = tree.count_neighbors(tree, radii)     # includes self and both orders
    K = area / (n * (n - 1)) * (counts - n)
    return np.sqrt(K / np.pi)


def random_in_polygon(poly, n, rng):
    minx, miny, maxx, maxy = poly.bounds
    out, got = [], 0
    while got < n:
        m = int((n - got) * 1.7) + 16
        xs = rng.uniform(minx, maxx, m); ys = rng.uniform(miny, maxy, m)
        mask = shapely.contains_xy(poly, xs, ys)
        out.append(np.column_stack([xs[mask], ys[mask]])); got += int(mask.sum())
    return np.vstack(out)[:n]


def run_max(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def build_windows(area_gpkg):
    sector = union_all(gpd.read_file(area_gpkg, layer=cfg.SECTOR_LAYER)
                       .to_crs(epsg=cfg.EPSG).geometry.values)
    cuencas = gpd.read_file(area_gpkg, layer=cfg.CUENCAS_LAYER).to_crs(epsg=cfg.EPSG)
    # match the basin field case-insensitively (robustness)
    bf = next((c for c in cuencas.columns if c.lower() == cfg.BASIN_FIELD.lower()), None)
    if bf is None:
        sys.exit(f"[error] no basin field like '{cfg.BASIN_FIELD}' in "
                 f"{cfg.CUENCAS_LAYER}. Columns: {list(cuencas.columns)}")
    win = {}
    for b in cfg.BASINS:
        cu = union_all(cuencas[cuencas[bf] == b].geometry.values)
        win[b] = sector.intersection(cu)
    return win


def main():
    cfg.ensure_dirs()
    rng = np.random.default_rng(cfg.SEED)

    cub = gpd.read_file(CUB_GPKG, layer=CUB_LAYER).to_crs(epsg=cfg.EPSG)
    cen = cub.geometry.centroid
    cx, cy = cen.x.values, cen.y.values

    win = build_windows(AREA_GPKG)

    rows, curves = [], {}
    for wname, poly in win.items():
        inside = shapely.contains_xy(poly, cx, cy)
        coords = np.column_stack([cx[inside], cy[inside]])
        area = float(poly.area)
        minx, miny, maxx, maxy = poly.bounds
        rmax = cfg.RMAX_FRAC * min(maxx - minx, maxy - miny)
        radii = np.linspace(rmax / cfg.N_RADII, rmax, cfg.N_RADII)

        print(f"\n[{wname}] n={len(coords)}  |W|={area/1e6:,.1f} km2  "
              f"lambda={len(coords)/(area/1e6):.3f}/km2  rmax={rmax/1000:.1f} km", flush=True)
        if len(coords) < 30:
            print("  too few points; skipped."); continue

        obs = L_of(coords, radii, area)
        sims = np.empty((cfg.N_SIM, len(radii)))
        for s in range(cfg.N_SIM):
            sims[s] = L_of(random_in_polygon(poly, len(coords), rng), radii, area)
        lo, hi = sims.min(0), sims.max(0)
        mean = sims.mean(0)
        p025, p975 = np.percentile(sims, [2.5, 97.5], axis=0)
        p005, p995 = np.percentile(sims, [0.5, 99.5], axis=0)

        above = obs > hi; below = obs < lo
        diag = []
        if run_max(above) >= 3:
            diag.append(f"CLUSTERING (L>env, run {run_max(above)}, "
                        f"from r~{radii[np.argmax(above)]/1000:.1f} km)")
        if run_max(below) >= 3:
            diag.append(f"REGULARITY (L<env, run {run_max(below)}, "
                        f"from r~{radii[np.argmax(below)]/1000:.1f} km)")
        print("  " + (" | ".join(diag) if diag else "compatible with CSR"))

        curves[wname] = dict(radii=radii, obs=obs, lo=lo, hi=hi, mean=mean,
                             p025=p025, p975=p975)
        for i, r in enumerate(radii):
            rows.append(dict(window=wname, n=len(coords), area_km2=area/1e6,
                             r_m=r, L_obs=obs[i], env_lo=lo[i], env_hi=hi[i],
                             env_mean=mean[i], env_p025=p025[i], env_p975=p975[i],
                             env_p005=p005[i], env_p995=p995[i]))

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\n[ok] {OUT_CSV}")

    # figure: L(r)-r per window, envelope + simulation mean
    plt = cfg.setup_matplotlib()
    # RRB first, then MRB: the order in which the manuscript discusses them
    wins = [w for w in ["CRR", "CMR"] if w in curves]
    fig, axes = plt.subplots(1, len(wins), figsize=(5.4 * len(wins), 4.6),
                             squeeze=False, sharey=True)
    for ax, wname in zip(axes[0], wins):
        c = curves[wname]; r = c["radii"]; rk = r / 1000
        ax.axhline(0, color="0.45", ls=":", lw=1.1)
        ax.fill_between(rk, c["lo"] - r, c["hi"] - r, color="0.88",
                        label="CSR envelope (min-max)")
        ax.fill_between(rk, c["p025"] - r, c["p975"] - r, color="0.62",
                        label="CSR 95%")
        ax.plot(rk, c["mean"] - r, color="0.20", ls="--", lw=1.6, label="CSR mean")
        ax.plot(rk, c["obs"] - r, color="#C44E52", lw=2.6, label="L observed")
        ax.set_xlabel("r (km)", fontsize=12)
        ax.tick_params(labelsize=11)
        ax.set_title(cfg.basin_label(wname), fontsize=13)
    axes[0][0].set_ylabel("L(r) - r (m)", fontsize=12)
    axes[0][0].legend(fontsize=10, loc="lower left")
    fig.tight_layout()
    cfg.save_figure(fig, "Figure_8")
    plt.close(fig)
    print("[ok] figures/Figure_8.png (+ .pdf)")


if __name__ == "__main__":
    main()
