#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13_fig5_rosarios.py   (manuscript Figure 5)

Bead-chain arrangements of cubetas ("rosarios de cubetas") in the RRB detail
window: the CAÑADA ORIENTATION (structure tensor of the relief) drawn as a short
axial segment at each cubeta centroid, optionally over a topographic-position (TPI)
background that makes the cañadas visible as threads of negative TPI.

The orientation is NOT recomputed here: it is read from data/tensor_cubetas.gpkg
(az_canada, coh) — the same product the Figure 6 roses use — and matched to each
cubeta centroid. A segment is drawn only where the tensor coherence exceeds COH_MIN
(elsewhere the cubeta is shown as a bare ring).

TPI BACKGROUND. The full raster is ACUMAR/COMIREC-derived from the LiDAR DEM and
is not distributed, but two crops of it are: data/tpi_fig5.tif for the detail
window and data/tpi_rrb_inset.tif, coarser, for the locator inset. The figure is
therefore fully reproducible. If either file is missing, the corresponding panel
is drawn without relief shading and the orientations are unaffected. The
background needs `rasterio`.

Inputs : data/cubetas.gpkg, data/tensor_cubetas.gpkg, data/area_de_estudio.gpkg
         data/tpi_fig5.tif       (TPI of the detail window, 10 m, EPSG:32721)
         data/tpi_rrb_inset.tif  (TPI of the whole basin, 200 m, locator inset)
Output : figures/Figure_5.{png,pdf}
Project: cubetas-morphometry | Python 3.11
"""
import sys
from pathlib import Path
import numpy as np
import geopandas as gpd
from pyproj import Transformer
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

# config.py lives in the same src/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import config as cfg
# Relief background: topographic position index, EPSG:32721, ACUMAR/COMIREC-derived.
# Only the two crops below are distributed; the full raster is not.
BG_RASTER   = cfg.TPI_FIG5_TIF       # detail window, 10 m, main map
INSET_RASTER = cfg.TPI_RRB_INSET     # whole basin, 200 m, locator inset
RASTER_EPSG = 32721

CUB_GPKG     = cfg.DATA / "cubetas.gpkg"
CUB_LAYER    = cfg.CUBETAS_LAYER
TENSOR_GPKG  = cfg.DATA / "tensor_cubetas.gpkg"
TENSOR_LAYER = "tensor_cubetas"

COH_MIN = 0.3               # minimum tensor coherence to draw a reliable orientation
# detail crop (EPSG:32721): SW, NE
X0, Y0, X1, Y1 = 291597, 6145379, 298700, 6151641

# background TPI — terrain-semantic colormap; TwoSlopeNorm pins neutral at TPI = 0
BG_MIN, BG_MAX = -0.4, 0.3
_TPI_STOPS = [
    (0.00, "#0B3C49"),   # deep petrol   (deepest cañada / cubeta)
    (0.22, "#19617A"),   # petrol
    (0.38, "#6FA8B8"),   # light teal
    (0.48, "#D9E6E3"),   # pale aqua-gray
    (0.50, "#F6F3EB"),   # warm paper    (TPI = 0)
    (0.58, "#EDE0C4"),   # pale sand
    (0.78, "#CFAE7F"),   # sand/tan
    (1.00, "#8C5A33"),   # muted umber   (highest interfluve)
]

# marker aesthetics
SEG_LEN_M   = 330
MARK_COLOR  = "#FFB000"
LINE_W      = 2.2
RING_S      = 55
SCALE_RINGS = False
INSET_RECT  = "#D62728"
RING_S_MIN, RING_S_MAX = 25, 150
N_TICKS     = 4
# alpha=1.0 rather than 0.75: a semi-transparent stroke forces PDF transparency
# groups on every one of the 200-odd markers, which costs megabytes. At this
# line width the halo reads the same opaque.
_OUT = [pe.withStroke(linewidth=2.6, foreground="#1A1A1A")]

def basin_field(gdf):
    return next((c for c in gdf.columns if c.lower() == cfg.BASIN_FIELD.lower()), None)


def latlon_ticks(ax):
    tr = Transformer.from_crs(RASTER_EPSG, 4326, always_xy=True)
    xt = np.linspace(X0, X1, N_TICKS)[:-1]      # drop rightmost (collides with colorbar)
    yt = np.linspace(Y0, Y1, N_TICKS)
    lon, _ = tr.transform(xt, np.full_like(xt, (Y0 + Y1) / 2))
    _, lat = tr.transform(np.full_like(yt, (X0 + X1) / 2), yt)
    ax.set_xticks(xt); ax.set_xticklabels([f"{v:.3f}\u00b0" for v in lon], fontsize=8)
    ax.set_yticks(yt); ax.set_yticklabels([f"{v:.3f}\u00b0" for v in lat], fontsize=8)


def main():
    has_tpi = BG_RASTER.exists()
    if not has_tpi:
        print(f"[info] TPI background not found ({BG_RASTER.name}); drawing without "
              f"relief shading. The orientations are unaffected.")

    # --- cubetas of the CRR crop; centroids in the raster CRS ---
    cub = gpd.read_file(CUB_GPKG, layer=CUB_LAYER).to_crs(epsg=RASTER_EPSG)
    bf = basin_field(cub) or "cuenca"
    cub = cub[cub[bf] == "CRR"]
    cen = cub.geometry.centroid
    cx_all, cy_all = cen.x.values, cen.y.values
    inside = (cx_all >= X0) & (cx_all <= X1) & (cy_all >= Y0) & (cy_all <= Y1)
    cx, cy = cx_all[inside], cy_all[inside]
    area_ha = cub.geometry.area.values[inside] / 1e4

    # --- cañada orientation & coherence per cubeta, from tensor_cubetas.gpkg ---
    ten = gpd.read_file(TENSOR_GPKG, layer=TENSOR_LAYER).to_crs(epsg=RASTER_EPSG)
    tbf = basin_field(ten) or "cuenca"
    ten = ten[ten[tbf] == "CRR"][["az_canada", "coh", "geometry"]]
    cub_pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(cx, cy),
                               crs=f"EPSG:{RASTER_EPSG}")
    j = gpd.sjoin_nearest(cub_pts, ten, how="left")
    j = j[~j.index.duplicated(keep="first")]
    az   = j["az_canada"].values
    cohc = j["coh"].values
    has = np.isfinite(az) & (cohc > COH_MIN)
    print(f"[CRR crop] cubetas inside={len(cx)}  with reliable tensor (coh>{COH_MIN})="
          f"{int(has.sum())} ({100*has.mean():.0f}%)")

    # orientation segments (no arrowhead), only where the tensor is reliable
    half = SEG_LEN_M / 2
    dx = half * np.sin(np.deg2rad(az[has])); dy = half * np.cos(np.deg2rad(az[has]))
    xs, ys = cx[has], cy[has]
    segs = [[(x - ddx, y - ddy), (x + ddx, y + ddy)]
            for x, y, ddx, ddy in zip(xs, ys, dx, dy)]

    # --- figure ---
    plt = cfg.setup_matplotlib()
    cmap = LinearSegmentedColormap.from_list("tpi_terrain", _TPI_STOPS)
    norm = TwoSlopeNorm(vmin=BG_MIN, vcenter=0.0, vmax=BG_MAX)
    aspect = (X1 - X0) / (Y1 - Y0)
    fig, ax = plt.subplots(figsize=(10.5, 10.5 / aspect))

    if has_tpi:
        import rasterio
        from rasterio.windows import from_bounds
        with rasterio.open(BG_RASTER) as src:
            win = from_bounds(X0, Y0, X1, Y1, src.transform)
            vd = src.read(1, window=win).astype("float64"); nod = src.nodata
        if nod is not None:
            vd[np.isclose(vd, nod)] = np.nan
        vd = np.ma.masked_invalid(vd)
        # rasterized: the TPI background is stored as an image instead of one
        # vector object per pixel. Markers, segments and text stay vector.
        im = ax.imshow(vd, extent=[X0, X1, Y0, Y1], origin="upper", cmap=cmap,
                       rasterized=True,
                       norm=norm, aspect="equal", interpolation="nearest")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.12)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("TPI (R = 500 m)", fontsize=9)
        cb.ax.text(0.5, 1.015, "high position", transform=cb.ax.transAxes,
                   ha="center", va="bottom", fontsize=8, color="#6B4A2B")
        cb.ax.text(0.5, -0.015, "low position", transform=cb.ax.transAxes,
                   ha="center", va="top", fontsize=8, color="#0B3C49")
    else:
        ax.set_aspect("equal")

    if SCALE_RINGS:
        s = np.clip(RING_S_MIN + 28.0 * np.sqrt(area_ha), RING_S_MIN, RING_S_MAX)
        ring_label = "cubeta centroids (size \u221d area)"
    else:
        s = RING_S
        ring_label = "cubeta centroids"
    rings = ax.scatter(cx, cy, s=s, facecolors="none", edgecolors=MARK_COLOR,
                       linewidths=1.3, zorder=3)
    rings.set_path_effects(_OUT)
    lc = LineCollection(segs, colors=MARK_COLOR, linewidths=LINE_W, zorder=4,
                        capstyle="round")
    lc.set_path_effects(_OUT)
    ax.add_collection(lc)
    ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1)
    latlon_ticks(ax)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor=MARK_COLOR, markersize=8, label=ring_label),
        Line2D([0], [0], color=MARK_COLOR, lw=2.2,
               label="valley orientation field ($\\theta_V$)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.75)

    # north arrow — white over the dark TPI, dark when there is no background
    na_color = "white" if has_tpi else "#333333"
    ax.annotate("N", xy=(0.95, 0.95), xytext=(0.95, 0.88), xycoords="axes fraction",
                ha="center", va="center", color=na_color, fontsize=14, fontweight="bold",
                path_effects=_OUT,
                arrowprops=dict(arrowstyle="-|>", color=na_color, lw=2))

    # scale bar (1 km)
    bar_len = 1000.0
    pad_x = (X1 - X0) * 0.04; pad_y = (Y1 - Y0) * 0.04
    xb = X0 + (X1 - X0) * 0.34; yb = Y0 + (Y1 - Y0) * 0.075
    box = Rectangle((xb - pad_x, yb - pad_y), bar_len + 2 * pad_x, (Y1 - Y0) * 0.085,
                    facecolor="black", alpha=0.45, edgecolor="white",
                    linewidth=0.8, zorder=5)
    ax.add_patch(box)
    tick = (Y1 - Y0) * 0.010
    ax.plot([xb, xb + bar_len], [yb, yb], color="white", lw=3.5,
            solid_capstyle="butt", zorder=6)
    ax.plot([xb, xb], [yb - tick, yb + tick], color="white", lw=3.5, zorder=6)
    ax.plot([xb + bar_len, xb + bar_len], [yb - tick, yb + tick],
            color="white", lw=3.5, zorder=6)
    ax.text(xb + bar_len / 2, yb + (Y1 - Y0) * 0.018, "1 km", color="white",
            ha="center", va="bottom", fontsize=11, fontweight="bold", zorder=6)

    # location inset (bottom-left): RRB outline + crop rectangle (+ TPI if available)
    try:
        bas = gpd.read_file(cfg.AREA_GPKG, layer=cfg.CUENCAS_LAYER).to_crs(epsg=RASTER_EPSG)
        ccol = next(c for c in bas.columns if c.lower() == "cuenca")
        rrb = bas.dissolve(by=ccol).loc["CRR"].geometry

        ax_in = ax.inset_axes([0.015, 0.015, 0.26, 0.26])
        if has_tpi:
            import rasterio
            from rasterio.enums import Resampling
            with rasterio.open(INSET_RASTER) as src:
                sh = src.shape
                f = max(1, int(max(sh) / 700))
                tpi_s = src.read(1, out_shape=(sh[0] // f, sh[1] // f),
                                 resampling=Resampling.average).astype("float64")
                nod_s = src.nodata
                bb = src.bounds
            if nod_s is not None:
                tpi_s[np.isclose(tpi_s, nod_s)] = np.nan
            tpi_s = np.ma.masked_invalid(tpi_s)
            ax_in.imshow(tpi_s, rasterized=True,
                         extent=[bb.left, bb.right, bb.bottom, bb.top],
                         origin="upper", cmap=cmap, norm=norm,
                         interpolation="bilinear", zorder=1)
        gpd.GeoSeries([rrb], crs=f"EPSG:{RASTER_EPSG}").boundary.plot(
            ax=ax_in, color="#444444", linewidth=0.9, zorder=3)
        ax_in.add_patch(Rectangle((X0, Y0), X1 - X0, Y1 - Y0, facecolor="none",
                                  edgecolor=INSET_RECT, linewidth=1.8, zorder=5))
        minx, miny, maxx, maxy = rrb.bounds
        mx = 0.04 * (maxx - minx); my = 0.04 * (maxy - miny)
        ax_in.set_xlim(minx - mx, maxx + mx); ax_in.set_ylim(miny - my, maxy + my)
        ax_in.set_xticks([]); ax_in.set_yticks([])
        ax_in.set_aspect("equal")
        ax_in.set_facecolor("white"); ax_in.patch.set_alpha(0.92)
        for sp in ax_in.spines.values():
            sp.set_edgecolor("#444444"); sp.set_linewidth(0.9)
        ax_in.text(0.06, 0.94, "Reconquista River Basin", transform=ax_in.transAxes, fontsize=8,
                   va="top", ha="left", color="#333333", fontweight="bold",
                   path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    except Exception as e:
        print(f"[warn] location inset skipped ({e})")

    # no burned-in title (caption in the manuscript)
    fig.tight_layout()
    cfg.ensure_dirs()
    for ext in ("png", "pdf"):
        fig.savefig(cfg.FIGURES / f"Figure_5.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"\n[ok] {cfg.FIGURES / 'Figure_5.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
