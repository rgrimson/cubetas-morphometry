#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_orientation.py   (manuscript Table 2, Section 4.2)
Orientation of the cubetas' long axis: is there a directional imprint?

What this script computes: the axial resultant length R of the cubeta long-axis
azimuth, per basin and for several subsets, against a Monte-Carlo isotropy band at
the same n. No significance test is applied: this is a census, so with 3210 bodies
any negligible departure from isotropy would be 'significant'. The criterion is the
MAGNITUDE of R judged against what isotropy yields at that n (Section 3.3).

Two estimators are reported: the second-moment axis, which the manuscript uses as
its primary estimator, and the maximum Feret diameter, which matches the criterion
of the regional aeolian literature and makes the published values comparable. The
minimum bounding rectangle is computed in script 01 and used here only as an
internal check of agreement between estimators.

Subsets computed: all cubetas; the 50 largest (matching the sampling protocol of the
regional aeolian literature); E >= 1.3 (shape sensitivity); and the upper area decile.
Results are written to results/orientation_R.csv and interpreted in the manuscript,
not here.

Inputs : data/cubetas_morphometry.gpkg (layer cubetas_morphometry)
Outputs: results/orientation_R.csv, figures/orientation_rose.{png,pdf}
Project: cubetas-morphometry | Python 3.11
"""
import sys
import numpy as np
import pandas as pd
import geopandas as gpd

from lib import config as cfg
IN_GPKG   = cfg.DATA / "cubetas_morphometry.gpkg"
IN_LAYER  = "cubetas_morphometry"
OUT_CSV   = cfg.RESULTS / "orientation_R.csv"

THETA_MBR   = "theta_mbr_geo"        # long-axis azimuth from the MBR (in the layer)
THETA_FERET = "theta_feret_geo"  # maximum Feret diameter (in the layer, script 01)
LARGEST_N = 50                # mirrors the N=50 large-lake aeolian sample
ISO_SIMS  = 5000              # Monte-Carlo resamples for the isotropy band
SECTOR    = 10                # rose sector width (deg)


# --------------------------------------------------------------------------


def axial_R(theta_deg):
    """Axial resultant: n, R and the mean axial direction (0-180).

    No Rayleigh p is computed. The inventory is a census, so with 3210 bodies any
    negligible departure from isotropy is 'significant'; what is interpretable is
    the magnitude of R against the isotropy band at the same n (Section 3.3).
    """
    th = np.asarray(theta_deg, float); th = th[np.isfinite(th)]
    n = len(th)
    if n < 3:
        return n, np.nan, np.nan
    ang = np.deg2rad(2.0 * th)
    C, S = np.cos(ang).mean(), np.sin(ang).mean()
    R = float(np.hypot(C, S))
    mean = (np.degrees(np.arctan2(S, C)) % 360) / 2.0
    return n, R, float(mean)


def iso_band(n, rng, sims=ISO_SIMS):
    """R under isotropy at this n: median and 95th percentile."""
    if n < 3:
        return np.nan, np.nan
    ang = rng.uniform(0, np.pi, size=(sims, n)) * 2.0
    R = np.hypot(np.cos(ang).mean(1), np.sin(ang).mean(1))
    return float(np.median(R)), float(np.percentile(R, 95))


def verdict(R, iso_p95):
    if not np.isfinite(R) or not np.isfinite(iso_p95):
        return "n/a"
    if R <= iso_p95:
        return "indistinguishable from isotropy"
    return "above isotropy band" if R > 1.3 * iso_p95 else "marginal"


def rose(ax, theta, color, title):
    th = np.asarray(theta, float); th = th[np.isfinite(th)] % 180.0
    edges = np.arange(0, 180 + SECTOR, SECTOR)
    cnt, _ = np.histogram(th, bins=edges)
    dens = cnt / cnt.sum() if cnt.sum() else cnt
    dens2 = np.concatenate([dens, dens])
    centers = np.deg2rad(np.arange(0, 360, SECTOR) + SECTOR / 2)
    ax.bar(centers, dens2, width=np.deg2rad(SECTOR) * 0.95, color=color,
           edgecolor="white", linewidth=0.4, alpha=0.85)
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["N", "E", "S", "W"], fontsize=9)
    ax.set_yticklabels([]); ax.set_title(title, fontsize=10, pad=12)


def main():
    cfg.ensure_dirs()
    rng = np.random.default_rng(cfg.SEED)

    if not IN_GPKG.exists():
        sys.exit(f"[error] not found: {IN_GPKG}. Run 01_morphometry.py first.")
    cub = gpd.read_file(IN_GPKG, layer=IN_LAYER)
    if cub.crs.to_epsg() != cfg.EPSG:
        cub = cub.to_crs(epsg=cfg.EPSG)
    # theta_mom_geo comes from 01_morphometry.py, corrected for meridian
    # convergence like every other azimuth in this pipeline.
    assert "theta_mom_geo" in cub.columns, (
        "theta_mom_geo missing: run 01_morphometry.py first")
    print(f"[info] {len(cub)} cubetas | {cub['cuenca'].value_counts().to_dict()}")


    # subsets: 'all' and 'largest50' are the paper-facing ones; the rest are
    # internal controls kept in the CSV.
    def subsets(g):
        yield "all", g
        yield f"largest{LARGEST_N}", g.nlargest(min(LARGEST_N, len(g)), "S_ha")
        yield "E>=1.3", g[g["E"] >= 1.3]           # sensitivity (Appendix A.2)
        yield "top_decile", g.nlargest(max(1, len(g)//10), "S_ha")  # internal

    rows = []
    for bname in cfg.BASINS + ["ALL"]:
        g = cub if bname == "ALL" else cub[cub["cuenca"] == bname]
        for sname, sub in subsets(g):
            # The minimum-bounding-rectangle axis is computed in script 01 but
            # not reported: the manuscript uses the second-moment axis as the
            # primary estimator and the maximum Feret diameter for comparability.
            for method, col in [("feret", THETA_FERET),
                                ("moments", "theta_mom_geo")]:
                if col not in sub.columns:
                    continue
                n, R, mean = axial_R(sub[col].values)
                iso_med, iso_p95 = iso_band(n, rng)
                rows.append(dict(basin=bname, subset=sname, method=method,
                                 n=n, R=round(R, 4) if np.isfinite(R) else np.nan,
                                 mean_dir=round(mean, 1) if np.isfinite(mean) else np.nan,
                                 iso_median=round(iso_med, 4) if np.isfinite(iso_med) else np.nan,
                                 iso_p95=round(iso_p95, 4) if np.isfinite(iso_p95) else np.nan,
                                 verdict=verdict(R, iso_p95)))
    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    print(f"[ok] {OUT_CSV}")

    # --- console: the paper-facing view (all + largest50, moments primary) ---
    pd.set_option("display.width", 170, "display.float_format", lambda v: f"{v:.4f}")
    show = res[res.subset.isin(["all", f"largest{LARGEST_N}"]) & (res.method == "feret")]
    print("\n" + "=" * 72)
    print("ORIENTATION — R vs isotropy band (Feret; paper-facing subsets)")
    print("=" * 72)
    print(show[["basin", "subset", "n", "R", "mean_dir", "iso_p95", "verdict"]]
          .to_string(index=False))

    print("\nReading:")
    for _, r in show.iterrows():
        print(f"  {r['basin']:>4} / {r['subset']:<10} n={int(r['n']):4d}  "
              f"R={r['R']:.3f}  (isotropy p95={r['iso_p95']:.3f}) -> {r['verdict']}")

    # MBR vs moments agreement (internal validation, not for the paper)
    d = np.abs(cub[THETA_MBR] - cub["theta_mom_geo"]) % 180.0
    d = np.minimum(d, 180 - d)
    print(f"\n[internal] MBR vs moments axial agreement: median={np.nanmedian(d):.1f} deg "
          f"(E>=1.3: {np.nanmedian(d[cub['E']>=1.3]):.1f} deg)")

    # --- rose figure (moments, all cubetas) per basin + ALL ---
    plt = cfg.setup_matplotlib()
    fig = plt.figure(figsize=(11, 4))
    for k, bname in enumerate(cfg.BASINS + ["ALL"]):
        g = cub if bname == "ALL" else cub[cub["cuenca"] == bname]
        _, R, _ = axial_R(g["theta_mom_geo"].values)
        color = cfg.BASIN_COLORS.get(bname, "#888888")
        ax = fig.add_subplot(1, 3, k + 1, projection="polar")
        rose(ax, g["theta_mom_geo"].values, color, f"{bname}  (n={len(g)}, R={R:.2f})")
    fig.tight_layout()
    cfg.save_figure(fig, "orientation_rose")
    plt.close(fig)
    print(f"\n[ok] figures/orientation_rose.png (+ .pdf)")


if __name__ == "__main__":
    main()
