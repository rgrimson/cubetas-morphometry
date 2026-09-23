#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_modality.py   (manuscript Table A1, Appendix A.1 and A.2)

Directional structure of cubeta orientation: three competing descriptions of the
axial azimuth distribution, fitted by expectation-maximisation and compared by BIC.

    VM1   one preferred direction shared by the whole population      (2 par)
    VM1U  one preferred direction over a directionless background     (3 par)
    VM2   two preferred directions                                    (5 par)

The modality decision is taken on the second-moment axis, because the maximum
Feret diameter is depleted near the grid axes and creates a spurious two-component
fit on the grid diagonals (Appendix A.3). Both estimators are reported, with and
without the E >= 1.3 shape filter, so that the sensitivity check the manuscript
announces is verifiable for the estimator that carries the decision.

All three BIC values and all three AIC values are exported, not only the winner's:
the argument of the paper is a model-selection argument, so the alternatives have
to be auditable. Appendix A.1 states why BIC is preferred.

Inputs : results/morphometry.csv  (theta_mom_geo, theta_feret_geo, S_ha, E, cuenca)
Outputs: results/TableA1_modality.csv
Project: cubetas-morphometry | Python 3.11
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import config as cfg
from lib import vonmises as vm

OUT_CSV = cfg.RESULTS / "TableA1_modality.csv"

ESTIMATORS = [("moments", "theta_mom_geo"), ("feret", "theta_feret_geo")]
FILTERS    = [("no filter", None), (f"E>={cfg.SHAPE_FILTER_E}", cfg.SHAPE_FILTER_E)]

# The 61 deg reference: mean long-axis azimuth reported for the lakes of the
# adjoining loessic sector LLM-2 (Tripaldi et al. 2025, Table 6.1). It is a
# morphometric descriptor of those lakes, never a wind direction.
AZ_REF = 61.0


def axial_dist(a, b=AZ_REF):
    """Smallest axial separation between two azimuths, in degrees (0-90)."""
    d = abs(float(a) - b) % 180.0
    return min(d, 180.0 - d)


def iso_band(n, rng, ndraws=None):
    """Median and 95th percentile of R under isotropy, by Monte Carlo.

    The band is simulated rather than taken from the asymptotic sqrt(-ln 0.05 / n),
    which is only accurate for large n; the analytical value is reported alongside.
    """
    ndraws = ndraws or cfg.N_ISO
    ang = rng.uniform(0, np.pi, size=(ndraws, n))       # doubled angles
    R = np.abs(np.exp(2j * ang).mean(axis=1))
    return float(np.median(R)), float(np.percentile(R, 95))


def axial_R(theta_deg):
    """Resultant length and mean direction of axial azimuths in degrees."""
    x = np.asarray(theta_deg, float)
    x = x[np.isfinite(x)]
    z = np.exp(2j * np.deg2rad(x)).mean()
    return len(x), float(abs(z)), float((np.rad2deg(np.angle(z)) / 2.0) % 180.0)


def size_classes(g):
    """The five rows of Table A1 for one basin group."""
    yield "all", g
    for lo, hi, name in cfg.AREA_CLASSES:
        sub = g
        if lo is not None:
            sub = sub[sub["S_ha"] >= lo]
        if hi is not None:
            sub = sub[sub["S_ha"] < hi]
        yield name, sub
    yield "top_decile", g.nlargest(max(1, len(g) // 10), "S_ha")


def strength(d):
    """BIC difference read as in Appendix A.1."""
    return "strong" if d > 6 else ("weak" if d > 2 else "marginal")


def main():
    cfg.ensure_dirs()
    m = pd.read_csv(cfg.RESULTS / "morphometry.csv")
    need = {"cuenca", "S_ha", "E", "theta_mom_geo", "theta_feret_geo"}
    missing = need - set(m.columns)
    assert not missing, f"morphometry.csv is missing {missing}"
    print(f"[info] {len(m)} cubetas | {m['cuenca'].value_counts().to_dict()}")

    rng = np.random.default_rng(cfg.ISO_SEED)
    rows = []

    for est, col in ESTIMATORS:
        for fname, fmin in FILTERS:
            base = m if fmin is None else m[m["E"] >= fmin]
            for bname in ["ALL"] + cfg.BASINS:
                g = base if bname == "ALL" else base[base["cuenca"] == bname]
                for cname, sub in size_classes(g):
                    theta = sub[col].dropna().values
                    fits = vm.fit_all(theta, seed=cfg.VM_SEED)
                    if fits is None:
                        print(f"[skip] {est}/{fname}/{bname}/{cname}: n={len(theta)} < 30")
                        continue
                    sel = vm.select_bic(fits)
                    n, R, mean_dir = axial_R(theta)
                    iso_med, iso_p95 = iso_band(n, rng)
                    win = fits[sel["bic_model"]]
                    comps = win["comps"]                 # dominant component first

                    rec = dict(
                        estimator=est, filter=fname, basin=bname, size_class=cname,
                        n=n, R=round(R, 4), mean_dir=round(mean_dir, 1),
                        iso_p95=round(iso_p95, 4), iso_median=round(iso_med, 4),
                        iso_p95_asymptotic=round(float(np.sqrt(-np.log(0.05) / n)), 4),
                        model=sel["bic_model"], dBIC_next=round(sel["dBIC_next"], 2),
                        strength=strength(sel["dBIC_next"]),
                        model_aic=sel["aic_model"], dAIC_next=round(sel["dAIC_next"], 2),
                        criteria_agree=(sel["bic_model"] == sel["aic_model"]),
                        BIC_VM1=round(sel["BIC"]["VM1"], 1),
                        BIC_VM1U=round(sel["BIC"]["VM1U"], 1),
                        BIC_VM2=round(sel["BIC"]["VM2"], 1),
                        AIC_VM1=round(sel["AIC"]["VM1"], 1),
                        AIC_VM1U=round(sel["AIC"]["VM1U"], 1),
                        AIC_VM2=round(sel["AIC"]["VM2"], 1),
                        kappa_at_bound=win["kappa_at_bound"],
                    )
                    for j, c in enumerate(comps, start=1):
                        rec[f"mu{j}"] = round(c["mu_deg"], 1)
                        rec[f"kappa{j}"] = round(c["kappa"], 2)
                        rec[f"w{j}"] = round(c["w"], 3)
                    rec["min_dist_to_ref_deg"] = round(
                        min(axial_dist(c["mu_deg"]) for c in comps), 1)
                    rows.append(rec)

    res = pd.DataFrame(rows)
    ORDER = ["estimator", "filter", "basin", "size_class", "n", "R", "mean_dir",
             "iso_median", "iso_p95", "iso_p95_asymptotic",
             "model", "dBIC_next", "strength", "model_aic", "dAIC_next",
             "criteria_agree", "BIC_VM1", "BIC_VM1U", "BIC_VM2",
             "AIC_VM1", "AIC_VM1U", "AIC_VM2",
             "mu1", "kappa1", "w1", "mu2", "kappa2", "w2",
             "min_dist_to_ref_deg", "kappa_at_bound"]
    res = res[[c for c in ORDER if c in res.columns]]
    res.to_csv(OUT_CSV, index=False)
    print(f"[ok] {OUT_CSV}  ({len(res)} rows)")

    # ---------------------------------------------------------------- checks
    pd.set_option("display.width", 200)
    key = res[(res.estimator == "moments") & (res["filter"] == "no filter")
              & (res.size_class != "top_decile")]
    print("\n" + "=" * 78)
    print("TABLE A1 - modality on the second-moment axis, unfiltered")
    print("=" * 78)
    print(key[["basin", "size_class", "n", "R", "iso_p95", "model", "dBIC_next",
               "mu1", "kappa1", "w1", "model_aic", "criteria_agree"]]
          .to_string(index=False))

    print("\nVM2 selected by BIC in", int((key.model == "VM2").sum()), "of", len(key))
    print("VM2 selected by AIC in", int((key.model_aic == "VM2").sum()), "of", len(key))
    print("criteria agree in     ", int(key.criteria_agree.sum()), "of", len(key))
    print(f"closest fitted component to {AZ_REF:.0f} deg:",
          key.min_dist_to_ref_deg.min(), "deg")

    if res.kappa_at_bound.any():
        print("\n[!] fits with kappa at the optimiser bound:")
        print(res.loc[res.kappa_at_bound,
                      ["estimator", "filter", "basin", "size_class", "kappa1", "kappa2"]]
              .to_string(index=False))



if __name__ == "__main__":
    main()
