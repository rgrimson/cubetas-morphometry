#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
09_coupling.py   (manuscript Figure 7, Section 4.4)

Is each cubeta oriented along its own cañada? The three directional signals share
a regional NNE axis, but sharing a mean direction does not imply agreeing body by
body: two populations can have the same mean and be uncorrelated case by case. The
question is therefore asked on the axial difference theta_C - theta_V, which
cancels the regional axis and retains only the local agreement.

Null. The agreement is read against a permutation null, obtained by permuting
theta_V among the cubetas OF EACH STRATUM. Permuting within the stratum is what
makes the null conservative: for two independent axial signals the resultant of
their difference converges to R_C * R_V, not to zero, so where the cañada is well
expressed and theta_V is more concentrated, agreement by chance is higher. The
band rises with the quartile and the isotropy band does not, which is why the
isotropy reference is anti-conservative here and is reported only for comparison.

The valley orientation field is read with the cubetas masked out of the DEM and
sampled on a ring entirely outside each body (Appendix A.4), so the coupling
cannot be an echo of the landforms being measured.

No consistency threshold is applied. The threshold of cfg.CONSISTENCY_TH belongs
to the descriptive summary of the field (Figure 6); here local consistency is the
stratifying variable, so filtering on it would remove the stratum that carries the
result.

Inputs : data/tensor_cubetas.gpkg        (az_canada, ring_consistency)
         data/cubetas_morphometry.gpkg   (theta_mom_geo, theta_feret_geo, S_ha, a_m)
Outputs: results/valley_coupling.csv
         figures/Figure_7.{png,pdf}

Run after the tensor layer has been produced (see prep/).
Project: cubetas-morphometry | Python 3.11
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import config as cfg

OUT_CSV = cfg.RESULTS / "valley_coupling.csv"
FIG_NAME = "Figure_7"

MIN_N = 50                      # a stratum below this is not evaluated
QLABELS = ["Q1", "Q2", "Q3", "Q4"]

# Panels of Figure 7: the pooled census and the smallest class, which is where a
# reader would suspect the gradient is really a size effect.
FIG_CLASSES = ["all", "<1 ha"]
FIG_BASINS = ["ALL", "CMR", "CRR"]
FIG_COLORS = {"ALL": "#333333", **cfg.BASIN_COLORS}
FIG_MARKERS = {"ALL": "o", "CMR": "s", "CRR": "^"}


# --------------------------------------------------------------------- stats
def Rdiff(tc, tv):
    """Resultant of the axial difference, via the doubled-angle transformation."""
    zc = np.exp(2j * np.deg2rad(np.asarray(tc, float)))
    zv = np.exp(2j * np.deg2rad(np.asarray(tv, float)))
    return float(np.abs((zc * np.conj(zv)).mean()))


def med_absdiff(tc, tv):
    """Median absolute axial departure, in degrees (0-90)."""
    d = np.abs(np.asarray(tc, float) - np.asarray(tv, float)) % 180.0
    return float(np.median(np.minimum(d, 180.0 - d)))


def perm_null(tc, tv, rng, n_perm=None, chunk=100):
    """95th percentile and mean of R under permutation of theta_V within the stratum."""
    n_perm = n_perm or cfg.N_PERM
    zc = np.exp(2j * np.deg2rad(np.asarray(tc, float)))
    zv = np.exp(2j * np.deg2rad(np.asarray(tv, float)))
    n, vals, done = zc.size, [], 0
    while done < n_perm:
        k = min(chunk, n_perm - done)
        idx = np.argsort(rng.random((k, n)), axis=1)
        vals.append(np.abs((zc[None, :] * np.conj(zv[idx])).mean(axis=1)))
        done += k
    v = np.concatenate(vals)
    return float(np.percentile(v, 95)), float(v.mean())


def iso_p95(n):
    """Asymptotic isotropy band, reported for comparison only."""
    return float(np.sqrt(-np.log(0.05) / n))


# ---------------------------------------------------------------------- data
def load():
    ten = gpd.read_file(cfg.TENSOR_GPKG, layer=cfg.TENSOR_LAYER)
    mor = gpd.read_file(cfg.MORPH_GPKG, layer=cfg.MORPH_LAYER)
    print(f"[info] tensor {len(ten)} | morphometry {len(mor)}")

    # Columns present on both sides are dropped before the merge: left in place
    # they come back suffixed _x and _y, and a later lookup by bare name fails.
    dup = [c for c in ten.columns
           if c in mor.columns and c not in (cfg.ID_FIELD, "geometry")]
    if dup:
        print("[info] dropped from tensor (already in morphometry):", dup)
        ten = ten.drop(columns=dup)

    d = mor.drop(columns="geometry").merge(
        ten.drop(columns="geometry"), on=cfg.ID_FIELD, how="inner", validate="1:1")
    bad = [c for c in d.columns if c.endswith(("_x", "_y"))]
    assert not bad, f"merge produced suffixed columns: {bad}"

    for c in ["az_canada", "ring_consistency", "theta_mom_geo",
              "theta_feret_geo", "S_ha", "a_m", cfg.BASIN_FIELD]:
        assert c in d.columns, f"missing column {c}; have {list(d.columns)}"

    n0 = len(d)
    d = d[d["az_canada"].notna() & d["ring_consistency"].notna()].copy()
    print(f"[info] with a defined valley field: {len(d)} of {n0}")

    d["cls"] = pd.cut(d["S_ha"], [-np.inf, 1, 2, np.inf],
                      labels=["<1 ha", "1-2 ha", ">=2 ha"])
    edges = [b[0] for b in cfg.LENGTH_BINS] + [np.inf]
    labels = [f"{lo}-{hi} m" if hi else f">{lo} m" for lo, hi in cfg.LENGTH_BINS]
    d["lenbin"] = pd.cut(d["a_m"], edges, labels=labels, right=False)
    return d


# ---------------------------------------------------------------------- main
def main():
    cfg.ensure_dirs()
    d = load()
    rng = np.random.default_rng(cfg.PERM_SEED)
    rows = []

    def add(sub, group, kind, stratum, est, col):
        if len(sub) < MIN_N:
            print(f"[skip] {group}/{stratum}/{est}: n={len(sub)} < {MIN_N}")
            return
        tc, tv = sub[col].values, sub["az_canada"].values
        R = Rdiff(tc, tv)
        p95, mu = perm_null(tc, tv, rng)
        rows.append(dict(group=group, kind=kind, stratum=stratum, estimator=est,
                         n=len(sub), R=round(R, 4), perm_p95=round(p95, 4),
                         perm_mean=round(mu, 4), iso_p95=round(iso_p95(len(sub)), 4),
                         medabs=round(med_absdiff(tc, tv), 1), above=bool(R > p95)))

    for est, col in [("moments", "theta_mom_geo"), ("feret", "theta_feret_geo")]:
        print(f"[run] {est}")

        # (a) length bins, pooled census: the size gradient of Section 4.4
        for lab in d["lenbin"].cat.categories[1:]:          # the 0-50 m bin is not reported
            add(d[d["lenbin"] == lab], "ALL/all", "length_bin", lab, est, col)

        # (b) the twelve basin-by-area-class cells, by quartile of local consistency
        for bsn in ["ALL"] + cfg.BASINS:
            for cl in ["all", "<1 ha", "1-2 ha", ">=2 ha"]:
                cell = d if bsn == "ALL" else d[d[cfg.BASIN_FIELD] == bsn]
                if cl != "all":
                    cell = cell[cell["cls"] == cl]
                if len(cell) < 4 * MIN_N:
                    print(f"[skip] cell {bsn}/{cl}: n={len(cell)}")
                    continue
                q = pd.qcut(cell["ring_consistency"], 4, labels=QLABELS,
                            duplicates="drop")
                for lab in QLABELS:
                    add(cell[q == lab], f"{bsn}/{cl}", "consistency_quartile",
                        lab, est, col)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    print(f"\n[ok] {OUT_CSV}  ({len(res)} rows)")

    figure7(res)
    report(res)


# -------------------------------------------------------------------- figure
def figure7(res):
    plt = cfg.setup_matplotlib()
    q = res[(res.kind == "consistency_quartile") & (res.estimator == "moments")]
    x = np.arange(1, 5)

    fig, axes = plt.subplots(1, len(FIG_CLASSES), figsize=(9.5, 4.0), sharey=True)
    for ax, cl in zip(np.atleast_1d(axes), FIG_CLASSES):
        for b in FIG_BASINS:
            s = q[q.group == f"{b}/{cl}"].set_index("stratum").reindex(QLABELS)
            if s.R.isna().all():
                continue
            c = FIG_COLORS[b]
            ax.plot(x, s.R.values, marker=FIG_MARKERS[b], color=c, lw=1.6, ms=5,
                    label=cfg.basin_label(b))
            ax.plot(x, s.perm_p95.values, color=c, lw=1.0, ls="--", alpha=0.65)
        ax.set_xticks(x); ax.set_xticklabels(QLABELS)
        ax.set_xlabel("Quartile of local consistency of the valley field")
        ax.set_title({"all": "All cubetas", "<1 ha": "Cubetas < 1 ha"}.get(cl, cl))
        ax.set_ylim(bottom=0)
    np.atleast_1d(axes)[0].set_ylabel(r"$R(\theta_C-\theta_V)$")
    np.atleast_1d(axes)[0].legend(loc="upper left")
    fig.tight_layout()
    for p in cfg.save_figure(fig, FIG_NAME):
        print(f"[ok] {p}")
    plt.close(fig)


# -------------------------------------------------------------------- report
def report(res):
    pd.set_option("display.width", 200)
    pool = res[(res.group == "ALL/all") & (res.estimator == "moments")]

    print("\n" + "=" * 78)
    print("COUPLING theta_C - theta_V, pooled census, second-moment axis")
    print("=" * 78)
    q = pool[pool.kind == "consistency_quartile"]
    print("\nBy quartile of local consistency:")
    print(q[["stratum", "n", "R", "perm_p95", "iso_p95", "medabs", "above"]]
          .to_string(index=False))
    if len(q) == 4:
        ratio = (q.R.values / q.perm_p95.values)
        print("  R / permutation band:", np.round(ratio, 2),
              "  monotone" if np.all(np.diff(ratio) > 0) else "  NOT monotone")

    print("\nBy rectangle length:")
    print(pool[pool.kind == "length_bin"]
          [["stratum", "n", "R", "perm_p95", "medabs"]].to_string(index=False))

    print("\nCells above the permutation band, of the twelve:")
    cells = res[res.kind == "consistency_quartile"]
    print(cells.groupby(["estimator", "stratum"])
               .agg(above=("above", "sum"), cells=("above", "count"))
               .reset_index().to_string(index=False))



if __name__ == "__main__":
    main()
