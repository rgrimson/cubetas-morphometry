#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_ripley_inhom.py   (manuscript Section 4.5)
Is the cubeta clustering (from the homogeneous L, script 04) first-order intensity
(terrain control) or second-order interaction (cubeta-cubeta)?

Inhomogeneous L (Baddeley, Moller & Waagepetersen 2000). The intensity lambda(u)
is modelled from covariates — regional slope (sigma=200 m, non-circular) and
distance to the Salado divide — by logistic regression of cubetas (presence) vs the
10k control points (background/quadrature): rho(u) ∝ exp(eta(u)). The null envelope
is an INHOMOGENEOUS Poisson process with that same lambda (n points drawn from the
control points with probability ∝ rho). Observed and simulated use the same
edge-uncorrected estimator; the Monte-Carlo envelope absorbs the edge bias.

Interpretation:
  - observed L_inhom INSIDE the inhomogeneous envelope  => the clustering seen in the
    homogeneous L is explained by first-order intensity (terrain). No interaction.
  - observed L_inhom ABOVE the envelope                 => residual interaction.

Note: the lambda normalisation (which involves |W|) enters observed and simulated
identically, so it cancels in the observed-vs-envelope comparison that drives the
conclusion — the verdict is robust to normalisation error.

Inputs : data/cubetas.gpkg, data/covariates_cubetas.csv,
         data/control_points.gpkg, data/area_de_estudio.gpkg
Outputs: results/ripley_inhom.csv, figures/ripley_inhom.{png,pdf}
Project: cubetas-morphometry | Python 3.11 | geopandas, numpy, scipy, scikit-learn
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import union_all
from scipy.spatial.distance import pdist
from sklearn.linear_model import LogisticRegression

from lib import config as cfg
CUB_GPKG   = cfg.DATA / "cubetas.gpkg"
CUB_LAYER  = cfg.CUBETAS_LAYER
CUB_COV    = cfg.DATA / "covariates_cubetas.csv"
CTRL_GPKG  = cfg.DATA / "control_points.gpkg"
CTRL_LAYER = "control_points"
OUT_CSV    = cfg.RESULTS / "ripley_inhom.csv"

COVARIATES  = ["slope_s200", "dist_divide_km"]   # regional slope + distance to divide
N_SIM_INHOM = 999        # Monte-Carlo simulations for the inhomogeneous envelope


def basin_field(gdf):
    return next((c for c in gdf.columns if c.lower() == cfg.BASIN_FIELD.lower()), None)


def L_inhom(coords, inv_lam, radii, area):
    """Edge-uncorrected inhomogeneous L. inv_lam = 1/lambda at each point.
    K(r) = (2/|W|) * sum_{pairs, d<=r} inv_lam_i*inv_lam_j ; L = sqrt(K/pi)."""
    n = len(coords)
    if n < 3:
        return np.full(len(radii), np.nan)
    d = pdist(coords)                                  # condensed, order matches triu(n,1)
    iu = np.triu_indices(n, 1)
    wp = inv_lam[iu[0]] * inv_lam[iu[1]]               # weight product per pair
    order = np.argsort(d)
    d_sorted = d[order]
    w_cum = np.cumsum(wp[order])
    idx = np.searchsorted(d_sorted, radii, side="right")
    cum_at_r = np.where(idx > 0, w_cum[np.clip(idx - 1, 0, len(w_cum) - 1)], 0.0)
    K = (2.0 / area) * cum_at_r
    return np.sqrt(np.maximum(K, 0) / np.pi)


def build_window(area_gpkg, basin):
    sector = union_all(gpd.read_file(area_gpkg, layer=cfg.SECTOR_LAYER)
                       .to_crs(epsg=cfg.EPSG).geometry.values)
    cuencas = gpd.read_file(area_gpkg, layer=cfg.CUENCAS_LAYER).to_crs(epsg=cfg.EPSG)
    bf = basin_field(cuencas)
    cu = union_all(cuencas[cuencas[bf] == basin].geometry.values)
    return sector.intersection(cu)


def run_max(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def main():
    cfg.ensure_dirs()
    rng = np.random.default_rng(cfg.SEED)

    cub = gpd.read_file(CUB_GPKG, layer=CUB_LAYER).to_crs(epsg=cfg.EPSG)
    bf = basin_field(cub) or "cuenca"
    cov = pd.read_csv(CUB_COV)
    cov["dist_divide_km"] = cov["dist_divide_m"] / 1000.0
    cub = cub.merge(cov[[cfg.ID_FIELD] + COVARIATES], on=cfg.ID_FIELD, how="left")
    cen = cub.geometry.centroid
    cub["_x"], cub["_y"] = cen.x.values, cen.y.values

    ctrl = gpd.read_file(CTRL_GPKG, layer=CTRL_LAYER).to_crs(epsg=cfg.EPSG)
    ctrl["dist_divide_km"] = ctrl["dist_divide_m"] / 1000.0
    ctrl["_x"], ctrl["_y"] = ctrl.geometry.x.values, ctrl.geometry.y.values

    rows, curves = [], {}
    for basin in cfg.BASINS:
        cq = cub[(cub[bf] == basin)].dropna(subset=COVARIATES).reset_index(drop=True)
        cc = ctrl[(ctrl["cuenca"] == basin)].dropna(subset=COVARIATES).reset_index(drop=True)
        win = build_window(cfg.AREA_GPKG, basin)
        area = float(win.area)
        n = len(cq)

        # --- intensity model: logistic cubetas(1) vs control(0) ---
        Xc = cq[COVARIATES].values
        Xk = cc[COVARIATES].values
        X = np.vstack([Xc, Xk])
        mu, sd = X.mean(0), X.std(0)
        Xs = (X - mu) / sd
        y = np.r_[np.ones(len(Xc)), np.zeros(len(Xk))]
        clf = LogisticRegression(penalty=None, max_iter=1000).fit(Xs, y)
        eta_cub  = clf.decision_function((Xc - mu) / sd)
        eta_ctrl = clf.decision_function((Xk - mu) / sd)
        rho_cub, rho_ctrl = np.exp(eta_cub), np.exp(eta_ctrl)

        # normalise lambda so it integrates to n over W (control pts as quadrature)
        M = len(cc)
        scale = n * M / (area * rho_ctrl.sum())
        lam_cub  = rho_cub * scale
        lam_ctrl = rho_ctrl * scale
        inv_lam_cub  = 1.0 / lam_cub
        inv_lam_ctrl = 1.0 / lam_ctrl

        # radii (same convention as script 04)
        minx, miny, maxx, maxy = win.bounds
        rmax = cfg.RMAX_FRAC * min(maxx - minx, maxy - miny)
        radii = np.linspace(rmax / cfg.N_RADII, rmax, cfg.N_RADII)

        coords = np.c_[cq["_x"].values, cq["_y"].values]
        obs = L_inhom(coords, inv_lam_cub, radii, area)

        # homogeneous observed (constant lambda) for reference in the figure
        inv_lam_hom = np.full(n, area / n)
        obs_hom = L_inhom(coords, inv_lam_hom, radii, area)

        # inhomogeneous Poisson null: draw n control pts with prob ∝ rho
        p = rho_ctrl / rho_ctrl.sum()
        ctrl_xy = np.c_[cc["_x"].values, cc["_y"].values]
        sims = np.empty((N_SIM_INHOM, len(radii)))
        print(f"[{basin}] n={n} |W|={area/1e6:.1f} km2  coef={dict(zip(COVARIATES, clf.coef_[0].round(3)))}"
              f"  ({N_SIM_INHOM} inhom sims)...", flush=True)
        for s in range(N_SIM_INHOM):
            idx = rng.choice(M, size=n, replace=False, p=p)
            sims[s] = L_inhom(ctrl_xy[idx], inv_lam_ctrl[idx], radii, area)
        lo, hi = sims.min(0), sims.max(0)
        mean = sims.mean(0)
        p025, p975 = np.percentile(sims, [2.5, 97.5], axis=0)

        above = obs > hi
        diag = (f"RESIDUAL CLUSTERING (L_inhom>env, run {run_max(above)}, "
                f"from r~{radii[np.argmax(above)]/1000:.1f} km)" if run_max(above) >= 3
                else "L_inhom within envelope -> clustering explained by intensity (first-order)")
        print(f"   {diag}")

        curves[basin] = dict(radii=radii, obs=obs, obs_hom=obs_hom,
                             lo=lo, hi=hi, mean=mean, p025=p025, p975=p975)
        for i, r in enumerate(radii):
            rows.append(dict(basin=basin, r_m=r, Linhom_obs=obs[i], Lhom_obs=obs_hom[i],
                             env_lo=lo[i], env_hi=hi[i], env_mean=mean[i],
                             env_p025=p025[i], env_p975=p975[i]))

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\n[ok] {OUT_CSV}")

    # figure: L_inhom(r)-r with inhom envelope; homogeneous excess as reference
    plt = cfg.setup_matplotlib()
    fig, axes = plt.subplots(1, len(cfg.BASINS), figsize=(5.2 * len(cfg.BASINS), 4.4),
                             squeeze=False)
    for ax, basin in zip(axes[0], cfg.BASINS):
        c = curves[basin]; r = c["radii"]; rk = r / 1000
        ax.axhline(0, color="0.5", ls=":", lw=1.0)
        ax.fill_between(rk, c["lo"] - r, c["hi"] - r, color="0.82",
                        label="inhom. Poisson env. (min-max)")
        ax.fill_between(rk, c["p025"] - r, c["p975"] - r, color="0.66", label="inhom. 95%")
        ax.plot(rk, c["obs_hom"] - r, color="0.45", ls="--", lw=1.4,
                label="L observed (homogeneous)")
        ax.plot(rk, c["obs"] - r, color="#C44E52", lw=2.2, label="L_inhom observed")
        ax.set_xlabel("r (km)"); ax.set_ylabel("L(r) - r (m)"); ax.set_title(basin)
    axes[0][0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    cfg.save_figure(fig, "ripley_inhom")
    plt.close(fig)
    print("[ok] figures/ripley_inhom.png (+ .pdf)")


if __name__ == "__main__":
    main()
