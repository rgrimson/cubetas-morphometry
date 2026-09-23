#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_fig3_area_distribution.py   (manuscript Figure 3)
Distribution of cubeta surface area (S, ha) in both basins (CMR, CRR).

Density histograms on a logarithmic area axis with KDE curves and each basin's
median marked. Density (normalised) makes the two basins comparable despite their
different n. The logarithmic axis is used because area spans more than two orders
of magnitude.

Area is taken from the cubeta polygon geometry (EPSG:5347), so the figure is
consistent with the frozen analysis dataset (n by basin as in the inventory).
No burned-in title: the caption goes in the manuscript.

Inputs : data/cubetas.gpkg
Outputs: results/area_distribution.csv, figures/Figure_3.{png,pdf}
Project: cubetas-morphometry | Python 3.11
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import gaussian_kde

from lib import config as cfg
CUB_GPKG  = cfg.DATA / "cubetas.gpkg"
CUB_LAYER = cfg.CUBETAS_LAYER
OUT_CSV   = cfg.RESULTS / "area_distribution.csv"
TICKS     = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]


def basin_field(gdf):
    return next((c for c in gdf.columns if c.lower() == cfg.BASIN_FIELD.lower()), None)


def main():
    cfg.ensure_dirs()

    cub = gpd.read_file(CUB_GPKG, layer=CUB_LAYER).to_crs(epsg=cfg.EPSG)
    bf = basin_field(cub) or "cuenca"
    cub["S_ha"] = cub.geometry.area / 1e4
    cub = cub[np.isfinite(cub["S_ha"]) & (cub["S_ha"] > 0)]

    order = [b for b in cfg.BASINS if b in cub[bf].unique()]
    groups = [(b, cub.loc[cub[bf] == b, "S_ha"].values) for b in order]

    # summary table (for the manuscript / traceability)
    rows = []
    for b, s in groups:
        q = np.percentile(s, [25, 50, 75])
        rows.append(dict(basin=b, n=len(s), median_ha=round(q[1], 3),
                         p25_ha=round(q[0], 3), p75_ha=round(q[2], 3),
                         min_ha=round(float(s.min()), 3), max_ha=round(float(s.max()), 3)))
    all_s = cub["S_ha"].values
    q = np.percentile(all_s, [25, 50, 75])
    rows.append(dict(basin="ALL", n=len(all_s), median_ha=round(q[1], 3),
                     p25_ha=round(q[0], 3), p75_ha=round(q[2], 3),
                     min_ha=round(float(all_s.min()), 3), max_ha=round(float(all_s.max()), 3)))
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"[ok] {OUT_CSV}")
    for r in rows:
        print(f"  {r['basin']:>3}: n={r['n']:4d}  median={r['median_ha']:.3f} ha  "
              f"IQR {r['p25_ha']:.3f}-{r['p75_ha']:.3f}  "
              f"min={r['min_ha']:.3f}  max={r['max_ha']:.3f}")

    # figure
    plt = cfg.setup_matplotlib()
    fig, axA = plt.subplots(1, 1, figsize=(7.2, 4.4))

    # --- log-scale density + KDE ---
    allx = np.log10(all_s)
    bins = np.linspace(allx.min(), allx.max(), 30)
    xx = np.linspace(allx.min(), allx.max(), 300)
    for b, s in groups:
        x = np.log10(s); col = cfg.BASIN_COLORS.get(b, "#888888")
        axA.hist(x, bins=bins, density=True, histtype="stepfilled", alpha=0.38,
                 color=col, edgecolor=col, linewidth=1.2,
                 label=f"{cfg.basin_label(b)}  (n={len(s)}; median {np.median(s):.2f} ha)")
        axA.plot(xx, gaussian_kde(x)(xx), color=col, lw=2)
        axA.axvline(np.median(x), color=col, ls="--", lw=1.3, alpha=0.9)
    axA.set_xticks(np.log10(TICKS)); axA.set_xticklabels([str(t) for t in TICKS])
    axA.set_xlabel("surface area (ha, log scale)")
    axA.set_ylabel("density")
    axA.legend(frameon=False, fontsize=9, loc="upper right")
    axA.spines[["top", "right"]].set_visible(False)


    fig.tight_layout()
    cfg.save_figure(fig, "Figure_3")
    plt.close(fig)
    print("[ok] figures/Figure_3.png (+ .pdf)")


if __name__ == "__main__":
    main()
