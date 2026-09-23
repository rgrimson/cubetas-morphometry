"""
vonmises.py — von Mises mixture fitting for axial orientation data.

Three competing descriptions of an axial azimuth distribution, fitted by
expectation-maximisation on doubled angles and compared by BIC:

    VM1   one preferred direction shared by the whole population      (2 par)
    VM1U  one preferred direction over a uniform background           (3 par)
    VM2   two preferred directions                                    (5 par)

All fitting functions take DOUBLED ANGLES IN RADIANS: use dbl() to convert axial
azimuths in degrees and undbl() to convert a fitted mean back. This module is
imported, never executed; run it directly to check it against the published fit.

Reference: Section 3.5 and Appendix A.1 of the manuscript.
"""
import numpy as np
from scipy import special, optimize

__all__ = ["dbl", "undbl", "fit_vm1", "fit_vm1u", "fit_vm2", "fit_all",
           "select_bic", "KAPPA_MAX"]

KAPPA_MAX = 1e4     # fits that hit this bound are flagged, not reported as fitted


def dbl(theta_deg):
    """Axial azimuth in degrees (0-180) -> doubled angle in radians (-pi, pi]."""
    return np.angle(np.exp(2j * np.deg2rad(np.asarray(theta_deg, float))))


def undbl(x_rad):
    """Doubled angle in radians -> axial azimuth in degrees (0-180)."""
    return float((np.rad2deg(x_rad) / 2.0) % 180.0)


def _kappa(R):
    """ML concentration for a given mean resultant length (Mardia & Jupp 2000)."""
    R = float(np.clip(R, 1e-12, 1 - 1e-12))
    f = lambda k: special.i1e(k) / special.i0e(k) - R
    if f(KAPPA_MAX) < 0:
        return KAPPA_MAX
    return float(optimize.brentq(f, 1e-9, KAPPA_MAX))


def _lpdf(x, mu, k):
    k = max(k, 1e-9)
    return k * (np.cos(x - mu) - 1) - np.log(2 * np.pi * special.i0e(k))


def fit_vm1(x):
    C, S = np.cos(x).mean(), np.sin(x).mean()
    mu, k = np.arctan2(S, C), _kappa(np.hypot(C, S))
    return dict(model="VM1", ll=float(_lpdf(x, mu, k).sum()), npar=2,
                comps=[dict(mu=float(mu), kappa=k, w=1.0)])


def fit_vm1u(x, iters=500, tol=1e-8):
    lu = -np.log(2 * np.pi)
    C, S = np.cos(x).mean(), np.sin(x).mean()
    mu = np.arctan2(S, C)
    k, w, ll_old, ll = max(_kappa(np.hypot(C, S)), .1), .5, -np.inf, -np.inf
    for _ in range(iters):
        la = np.log(max(w, 1e-12)) + _lpdf(x, mu, k)
        lb = np.log(max(1 - w, 1e-12)) + lu
        m = np.maximum(la, lb)
        ll = float((m + np.log(np.exp(la - m) + np.exp(lb - m))).sum())
        r = np.exp(la - m) / (np.exp(la - m) + np.exp(lb - m))
        w = float(r.mean())
        Cw, Sw = (r * np.cos(x)).sum(), (r * np.sin(x)).sum()
        mu = np.arctan2(Sw, Cw)
        k = _kappa(np.hypot(Cw, Sw) / max(r.sum(), 1e-12))
        if abs(ll - ll_old) < tol * max(1, abs(ll)):
            break
        ll_old = ll
    return dict(model="VM1U", ll=ll, npar=3,
                comps=[dict(mu=float(mu), kappa=k, w=w)])


def fit_vm2(x, iters=500, tol=1e-8, restarts=12, seed=None):
    """Two-component fit. `seed` makes the random restarts reproducible."""
    rng = np.random.default_rng(seed)
    best, mu0 = None, np.arctan2(np.sin(x).mean(), np.cos(x).mean())
    for t in range(restarts):
        if t == 0:
            mu = np.array([mu0, mu0 + np.pi / 2]) % (2 * np.pi)
        elif t == 1:
            mu = np.array([mu0, mu0 + np.pi]) % (2 * np.pi)
        else:
            mu = rng.uniform(0, 2 * np.pi, 2)
        k = np.array([2.0, 2.0]); w = np.array([.5, .5])
        ll_old = ll = -np.inf
        for _ in range(iters):
            L = np.vstack([np.log(max(w[j], 1e-12)) + _lpdf(x, mu[j], k[j])
                           for j in range(2)])
            m = L.max(0)
            ll = float((m + np.log(np.exp(L - m).sum(0))).sum())
            r = np.exp(L - m); r /= r.sum(0)
            w = r.mean(1)
            for j in range(2):
                Cw, Sw = (r[j] * np.cos(x)).sum(), (r[j] * np.sin(x)).sum()
                mu[j] = np.arctan2(Sw, Cw)
                k[j] = _kappa(np.hypot(Cw, Sw) / max(r[j].sum(), 1e-12))
            if abs(ll - ll_old) < tol * max(1, abs(ll)):
                break
            ll_old = ll
        if best is None or ll > best["ll"]:
            best = dict(model="VM2", ll=ll, npar=5,
                        comps=[dict(mu=float(mu[j]), kappa=float(k[j]),
                                    w=float(w[j])) for j in range(2)])
    # report the dominant component first
    best["comps"].sort(key=lambda c: -c["w"])
    return best


def fit_all(theta_deg, seed=None, min_n=30):
    """Fit the three models to axial azimuths in degrees. Returns {name: fit} or None."""
    x = dbl(theta_deg)
    x = x[np.isfinite(x)]
    if x.size < min_n:
        return None
    out = {"VM1": fit_vm1(x), "VM1U": fit_vm1u(x), "VM2": fit_vm2(x, seed=seed)}
    for f in out.values():
        f["n"] = int(x.size)
        f["kappa_at_bound"] = any(c["kappa"] >= KAPPA_MAX * 0.999 for c in f["comps"])
        for c in f["comps"]:
            c["mu_deg"] = undbl(c["mu"])
    return out


def select_bic(fits):
    """BIC and AIC for every model, plus the winner and the gap to the runner-up.

    All three criteria values are returned, not only the winner's: the argument of
    the paper is a model-selection argument, so the alternatives must be auditable.
    """
    n = fits["VM1"]["n"]
    bic = {m: -2 * f["ll"] + f["npar"] * np.log(n) for m, f in fits.items()}
    aic = {m: -2 * f["ll"] + 2 * f["npar"] for m, f in fits.items()}
    sb, sa = sorted(bic.values()), sorted(aic.values())
    return dict(n=n, BIC=bic, AIC=aic,
                bic_model=min(bic, key=bic.get), dBIC_next=sb[1] - sb[0],
                aic_model=min(aic, key=aic.get), dAIC_next=sa[1] - sa[0])


if __name__ == "__main__":
    # Self-test against the published fit of the pooled census (theta_mom_geo, n = 3210):
    # log L = -5835.93 / -5823.39 / -5821.27  ->  BIC = 11688.0 / 11671.0 / 11682.9
    import sys, geopandas as gpd
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as cfg

    g = gpd.read_file(cfg.CUBETAS_GPKG.parent / "cubetas_morphometry.gpkg")
    fits = fit_all(g["theta_mom_geo"].dropna().values, seed=cfg.VM_SEED)
    sel = select_bic(fits)
    TARGET = {"VM1": 11688.0, "VM1U": 11671.0, "VM2": 11682.9}
    for m in ["VM1", "VM1U", "VM2"]:
        ok = abs(sel["BIC"][m] - TARGET[m]) < 0.5
        print(f"{m:5s} BIC={sel['BIC'][m]:9.1f} (target {TARGET[m]:9.1f}) {'OK' if ok else 'MISMATCH'}")
        for c in fits[m]["comps"]:
            print(f"      mu={c['mu_deg']:6.1f}  kappa={c['kappa']:7.3f}  w={c['w']:.3f}")
    print("BIC winner:", sel["bic_model"], " AIC winner:", sel["aic_model"])
