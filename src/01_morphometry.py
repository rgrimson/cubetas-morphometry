#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
01_morphometry.py   (manuscript Table 1)
Planimetric morphometry of the cubetas (upper CMR and CRR basins).

Per-cubeta variables:
    S_ha        surface area (ha)
    P_m         perimeter (m). No cubeta in this census has interior rings,
                so the exterior and the total perimeter coincide
    a_m, b_m    long / short side of the minimum-area bounding rectangle (m)
    theta_mbr_grid   azimuth of the rectangle long axis, grid north (axial, 0-180)
    theta_mbr_geo    the same, corrected for meridian convergence (geographic)
    theta_feret_grid / theta_feret_geo
                     azimuth of the maximum Feret diameter, grid and geographic
    theta_mom_grid / theta_mom_geo
                     azimuth of the second-moment axis, grid and geographic.
                     This is the primary orientation estimator of the paper
    Dm_m        equivalent circular diameter, 2*sqrt(S/pi) (m)
    IC          circularity index, P / (2*sqrt(pi*S))  (1 = circle)
    E           elongation, a/b

Inputs : config.CUBETAS_GPKG (layer config.CUBETAS_LAYER)
Outputs: results/morphometry.csv           (per-cubeta table)
         results/morphometry_summary.csv    (descriptive statistics, tidy long)
         data/cubetas_morphometry.gpkg    (derived layer for downstream scripts)

Project: cubetas-morphometry | Python 3.11 | GeoPandas, Shapely 2.x, NumPy, pandas
"""
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import minimum_rotated_rectangle, make_valid, convex_hull, get_coordinates

from lib import config as cfg
DERIVED_GPKG  = cfg.DATA / "cubetas_morphometry.gpkg"
DERIVED_LAYER = "cubetas_morphometry"
OUT_CSV       = cfg.RESULTS / "morphometry.csv"
OUT_SUMMARY   = cfg.RESULTS / "morphometry_summary.csv"


def az_moments_local(geom):
    """Long-axis azimuth (0-180) by 2nd-order area moments, local coords."""
    if geom is None or geom.is_empty:
        return np.nan
    minx, miny, _, _ = geom.bounds
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    A = Mx = My = Ix = Iy = Ixy = 0.0
    for p in polys:
        xs, ys = p.exterior.coords.xy
        x = np.asarray(xs) - minx; y = np.asarray(ys) - miny
        cr = x[:-1]*y[1:] - x[1:]*y[:-1]
        A  += cr.sum()/2.0
        Mx += ((x[:-1]+x[1:])*cr).sum()/6.0
        My += ((y[:-1]+y[1:])*cr).sum()/6.0
        Iy += ((x[:-1]**2 + x[:-1]*x[1:] + x[1:]**2)*cr).sum()/12.0
        Ix += ((y[:-1]**2 + y[:-1]*y[1:] + y[1:]**2)*cr).sum()/12.0
        Ixy += ((x[:-1]*y[1:] + 2*x[:-1]*y[:-1] + 2*x[1:]*y[1:] + x[1:]*y[:-1])*cr).sum()/24.0
    if A == 0:
        return np.nan
    Cx, Cy = Mx/A, My/A
    m20 = Iy/A - Cx**2; m02 = Ix/A - Cy**2; m11 = Ixy/A - Cx*Cy
    w, v = np.linalg.eigh(np.array([[m20, m11], [m11, m02]]))
    vx, vy = v[:, int(np.argmax(w))]
    return float(np.degrees(np.arctan2(vx, vy)) % 180.0)


def feret_max(geom):
    """Maximum Feret diameter (m) and azimuth of the longest chord (grid north, 0-180).
    Tripaldi's 'azimuth of maximum length' criterion, over the convex-hull vertices."""
    if geom is None or geom.is_empty:
        return np.nan, np.nan
    pts = get_coordinates(convex_hull(geom))          # (k,2); maneja Multi* también
    if len(pts) < 2:
        return np.nan, np.nan
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    dx, dy = pts[j] - pts[i]
    return float(np.sqrt(d2[i, j])), float(np.degrees(np.arctan2(dx, dy)) % 180.0)

# --------------------------------------------------------------------------
# Geometry / morphometry
# --------------------------------------------------------------------------
def fix_validity(gdf):
    invalid = ~gdf.geometry.is_valid
    n = int(invalid.sum())
    if n:
        print(f"[warn] {n} invalid geometry(ies) repaired with make_valid.")
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].apply(make_valid)
    return gdf


def mbr_metrics(geom):
    """(a, b, theta_mbr_grid): long side, short side and azimuth of the long side
    (clockwise from grid north, 0-180) of the minimum-area bounding rectangle."""
    rect = minimum_rotated_rectangle(geom)
    if rect.geom_type != "Polygon":
        return (np.nan, np.nan, np.nan)
    x, y = rect.exterior.coords.xy
    pts = np.column_stack([np.asarray(x[:-1]), np.asarray(y[:-1])])
    e0, e1 = pts[1] - pts[0], pts[2] - pts[1]
    L0, L1 = float(np.hypot(*e0)), float(np.hypot(*e1))
    a, b, vlong = (L0, L1, e0) if L0 >= L1 else (L1, L0, e1)
    dx, dy = vlong
    az = float(np.degrees(np.arctan2(dx, dy)) % 180.0)
    return (a, b, az)


def meridian_convergence_deg(lon, lat, lon0):
    """gamma ~= (lon-lon0)*sin(lat) in degrees; azimuth_geo ~= azimuth_grid + gamma."""
    return (lon - lon0) * np.sin(np.radians(lat))


def ensure_id(gdf, id_field, basin_col):
    for cand in [id_field, "id", "ID", "fid", "FID", "OBJECTID"]:
        if cand in gdf.columns and gdf[cand].is_unique and gdf[cand].notna().all():
            if cand != id_field:
                gdf[id_field] = gdf[cand].astype(str)
            print(f"[info] Using existing ID field: '{cand}'.")
            return gdf
    ids, counters = [], {}
    for b in gdf[basin_col].fillna("NA"):
        counters[b] = counters.get(b, 0) + 1
        ids.append(f"{b}_{counters[b]:04d}")
    gdf[id_field] = ids
    print(f"[info] Generated ID field '{id_field}' (basin prefix + counter). "
          f"NOTE: stable only if row order is stable.")
    return gdf


def descriptive_table(df, group_col, value_cols):
    """Tidy long descriptive table by basin and for the whole set."""
    pcts = [0.25, 0.50, 0.75]
    frames = []
    for name, g in list(df.groupby(group_col)) + [("ALL", df)]:
        d = g[value_cols].describe(percentiles=pcts).T
        d = d[["mean", "std", "min", "25%", "50%", "75%", "max"]]
        d = d.rename(columns={"25%": "q25", "50%": "median", "75%": "q75"})
        d.insert(0, "n", int(len(g)))
        d.insert(0, "group", name)
        d.index.name = "variable"
        frames.append(d.reset_index())
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    cfg.ensure_dirs()
    print("=" * 72)
    print("MORPHOMETRY OF CUBETAS  |  CMR + CRR")
    print("=" * 72)

    if not cfg.CUBETAS_GPKG.exists():
        sys.exit(f"[error] not found: {cfg.CUBETAS_GPKG}")
    cub = gpd.read_file(cfg.CUBETAS_GPKG, layer=cfg.CUBETAS_LAYER)
    if cub.crs is None:
        sys.exit("[error] input layer has no CRS.")
    if cub.crs.to_epsg() != cfg.EPSG:
        print(f"[info] reprojecting {cub.crs.to_epsg()} -> EPSG:{cfg.EPSG}")
        cub = cub.to_crs(epsg=cfg.EPSG)

    cub = fix_validity(cub)

    # Basin column: use existing attribute if present
    if cfg.BASIN_FIELD in cub.columns:
        cub["cuenca"] = cub[cfg.BASIN_FIELD]
    else:
        sys.exit(f"[error] no basin field '{cfg.BASIN_FIELD}' in the layer. "
                 f"Columns: {list(cub.columns)}")
    print(f"[info] {len(cub)} cubetas | by basin: "
          f"{cub['cuenca'].value_counts().to_dict()}")

    cub = ensure_id(cub, cfg.ID_FIELD, "cuenca")

    # --- metrics ---
    geoms = cub.geometry
    cub["S_ha"] = geoms.area / 10_000.0

    perim_total = geoms.length
    perim_ext = geoms.exterior.length
    perim_ext = perim_ext.fillna(perim_total)   # MultiPolygon -> exterior is None
    n_holes = int((perim_total - perim_ext > 1e-6).sum())
    # No cubeta in this census has interior rings, so the exterior and the total
    # perimeter coincide and the circularity index is the same either way. The
    # total perimeter is used, and the check below reports any polygon that would
    # make the distinction matter.
    if n_holes:
        print(f"[warn] {n_holes} cubeta(s) with interior rings: the exterior and " 
              "total perimeters differ, and IC uses the total one.")
    cub["P_m"] = perim_total

    abtheta = geoms.apply(lambda g: pd.Series(mbr_metrics(g),
                                              index=["a_m", "b_m", "theta_mbr_grid"]))
    cub[["a_m", "b_m", "theta_mbr_grid"]] = abtheta

    cen_ll = cub.representative_point().to_crs(epsg=4326)
    gamma = meridian_convergence_deg(cen_ll.x.values, cen_ll.y.values, cfg.CENTRAL_MERIDIAN)
    cub["theta_mbr_geo"] = (cub["theta_mbr_grid"].values + gamma) % 180.0

    S_m2 = cub["S_ha"] * 10_000.0
    cub["Dm_m"] = 2.0 * np.sqrt(S_m2 / np.pi)
    cub["IC"]   = cub["P_m"] / (2.0 * np.sqrt(np.pi * S_m2))
    cub["E"]    = cub["a_m"] / cub["b_m"]
    
    fer = geoms.apply(lambda g: pd.Series(feret_max(g),
                                      index=["feret_max_m", "theta_feret_grid"]))
    cub[["feret_max_m", "theta_feret_grid"]] = fer
    # same meridian convergence (gamma) already computed for theta_mbr_geo:
    cub["theta_feret_geo"] = (cub["theta_feret_grid"].values + gamma) % 180.0

    # Second-moment axis, the primary orientation estimator of the paper. It is
    # computed here, with the other two, so that the three receive the same
    # meridian-convergence correction: az_moments_local works on the projected
    # coordinates of the polygon, so its raw output is a GRID azimuth.
    cub["theta_mom_grid"] = geoms.apply(az_moments_local)
    cub["theta_mom_geo"] = (cub["theta_mom_grid"].values + gamma) % 180.0

    # --- outputs: per-cubeta table + derived vector ---
    # Keep only the morphometric products (plus id and basin); never carry over
    # inventory/process columns that may be present in the input layer.
    keep = [cfg.ID_FIELD, "cuenca"] + cfg.MORPH_VARS + ["P_m", "theta_feret_grid"]
    keep = [c for c in dict.fromkeys(keep) if c in cub.columns]
    cub = cub[keep + ["geometry"]]

    cub.drop(columns="geometry").to_csv(OUT_CSV, index=False)
    cub.to_file(DERIVED_GPKG, layer=DERIVED_LAYER, driver="GPKG")
    print(f"\n[ok] {OUT_CSV}")
    print(f"[ok] {DERIVED_GPKG} (layer '{DERIVED_LAYER}')")

    # --- descriptive statistics ---
    tab = descriptive_table(cub.drop(columns="geometry"), "cuenca", cfg.MORPH_VARS)
    tab.to_csv(OUT_SUMMARY, index=False)
    print(f"[ok] {OUT_SUMMARY}")

    pd.set_option("display.max_rows", None, "display.width", 160,
                  "display.float_format", lambda v: f"{v:,.3f}")
    print("\n" + "=" * 72)
    print("DESCRIPTIVE STATISTICS (per basin and whole set)")
    print("=" * 72)
    for grp in tab["group"].unique():
        sub = tab[tab["group"] == grp].drop(columns="group").set_index("variable")
        print(f"\n--- {grp}  (n={int(sub['n'].iloc[0])}) ---")
        print(sub.drop(columns="n").to_string())

    # medians table (the paper reports medians): quick cross-check vs manuscript
    print("\n" + "-" * 72)
    print("MEDIANS by basin (paper table)")
    print("-" * 72)
    med = (cub.drop(columns="geometry")
              .groupby("cuenca")[cfg.MORPH_VARS].median())
    med.loc["ALL"] = cub[cfg.MORPH_VARS].median()
    print(med.to_string())


if __name__ == "__main__":
    main()
