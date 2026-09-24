#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_fig2_schematic.py   (manuscript Figure 2)

Schematic of the three axial directional signals analysed in the paper:

    theta_C   cubeta orientation          long axis of an individual cubeta
    theta_N   cubeta alignment            azimuth to the nearest neighbour
    theta_V   valley orientation field    local direction of the cañada

What the drawing has to convey is not the geometric definition of each angle but
that the three are measured on DIFFERENT SOURCES OF DATA: theta_C on the polygon
of the body, theta_N on the centroids alone, and theta_V on the relief with the
cubetas removed from the elevation model and the field read on a ring entirely
outside the body. That is why the three agreeing on NNE is a result and not a
tautology.

Three idealised cubetas sit in a bead-chain arrangement along a cañada, the
configuration Section 4.4 quantifies. Purely illustrative: nothing here is
measured from data, the positions are invented and the drawing is not to scale.

No burned-in title: the caption goes in the manuscript, like the other figures.

Output : figures/Figure_2.{png,pdf}
Project: cubetas-morphometry | Python 3.11 | matplotlib, scipy
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Wedge, Patch, FancyArrow
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import config as cfg

# ---- colours ---------------------------------------------------------------
# The three signals take their colours from the configuration, so that a signal
# is the same colour here and in the roses of Figure 6.
C_THC = cfg.SIGNAL_COLORS["theta_C"]
C_THN = cfg.SIGNAL_COLORS["theta_N"]
C_THV = cfg.SIGNAL_COLORS["theta_V"]

C_BAND = "#D6E3E8"      # the cañada, as a soft band
C_AXIS = "#8C9CA3"      # its axis, dashed
C_CUB = "#F5C85C"       # cubeta fill
C_CUB_E = "#C08A16"     # cubeta outline
C_NORTH = "#6B7280"     # grid north arrows
C_INK = "#2B2B2B"

# ---- geometry (arbitrary units, 0-1) ---------------------------------------
# Nodes the cañada axis passes through. A spline through them gives a sinuous
# valley, so that the local direction differs from the regional one, which is
# the point of separating theta_V from a single regional axis.
AXIS_NODES = np.array([[0.10, 0.05], [0.26, 0.20], [0.50, 0.26],
                       [0.74, 0.34], [0.90, 0.51]])

# (position along the axis 0-1, width, height, departure from the axis in deg)
CUBETAS = [(0.17, 0.125, 0.088, -17),
           (0.50, 0.109, 0.078, +13),
           (0.83, 0.136, 0.094, -25)]

BAND_LW = 118            # thickness of the cañada band, in points
RING_R = 0.103          # radius of the theta_V sampling ring
NORTH_LEN = 0.140        # length of the grid-north arrows
SECTOR_R = 0.058        # radius of the filled angular sectors


# ---- helpers ---------------------------------------------------------------
def spline(nodes, t):
    """Smooth interpolation of the nodes, evaluated at t in [0, 1]."""
    s = np.linspace(0, 1, len(nodes))
    return np.c_[CubicSpline(s, nodes[:, 0])(t), CubicSpline(s, nodes[:, 1])(t)]


def tangent_az(nodes, t, h=1e-3):
    """Axial azimuth of the tangent to the axis at t, clockwise from north."""
    p0 = spline(nodes, np.array([max(0.0, t - h)]))[0]
    p1 = spline(nodes, np.array([min(1.0, t + h)]))[0]
    d = p1 - p0
    return float(np.degrees(np.arctan2(d[0], d[1]))) % 180.0


def sector(ax, centre, az, radius, colour, label, lab_scale=2.35):
    """Filled sector from grid north to the azimuth, with its label.

    Matplotlib measures degrees anticlockwise from the +x axis, so an azimuth
    measured clockwise from north becomes 90 - az.
    """
    t1, t2 = 90.0 - az, 90.0
    ax.add_patch(Wedge(centre, radius, t1, t2, facecolor=colour, alpha=0.20,
                       edgecolor="none", zorder=5))
    ax.add_patch(Wedge(centre, radius, t1, t2, facecolor="none",
                       edgecolor=colour, lw=2.2, zorder=6))
    am = np.deg2rad(90.0 - az / 2.0)
    ax.text(centre[0] + radius * lab_scale * np.cos(am),
            centre[1] + radius * lab_scale * np.sin(am), label,
            color=colour, fontsize=19, fontweight="bold",
            ha="center", va="center", zorder=8)


def segment(ax, centre, az, length, colour, lw=4.0, z=6):
    """Axial segment: a direction has no head, so neither does the line."""
    dx = length / 2.0 * np.sin(np.deg2rad(az))
    dy = length / 2.0 * np.cos(np.deg2rad(az))
    ax.plot([centre[0] - dx, centre[0] + dx], [centre[1] - dy, centre[1] + dy],
            color=colour, lw=lw, solid_capstyle="round", zorder=z)


def main():
    cfg.ensure_dirs()
    fig = plt.figure(figsize=(11.2, 6.2))
    ax = fig.add_axes([0.0, 0.26, 1.0, 0.74])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 0.60)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # ---- the cañada: a soft band along a sinuous axis ----------------------
    # Drawn slightly beyond the nodes so that the band is cut by the frame
    # rather than ending in a visible cap: a cañada does not begin or end here.
    tt = np.linspace(-0.13, 1.13, 500)
    curve = spline(AXIS_NODES, tt)
    ax.plot(curve[:, 0], curve[:, 1], color=C_BAND, lw=BAND_LW,
            solid_capstyle="round", zorder=1)
    ax.plot(curve[:, 0], curve[:, 1], color=C_AXIS, lw=1.5, ls=(0, (7, 5)),
            zorder=2)

    centres = [spline(AXIS_NODES, np.array([t]))[0] for t, *_ in CUBETAS]
    az_axis = [tangent_az(AXIS_NODES, t) for t, *_ in CUBETAS]
    az_cub = [(a + dev) % 180.0 for (t, w, h, dev), a in zip(CUBETAS, az_axis)]

    # ---- the three bodies, each departing from the local axis --------------
    for (t, w, h, dev), c, azc in zip(CUBETAS, centres, az_cub):
        ax.add_patch(Ellipse(c, w, h, angle=90.0 - azc, facecolor=C_CUB,
                             edgecolor=C_CUB_E, lw=1.6, alpha=0.95, zorder=4))

    # ---- every body carries both directions, drawn thin --------------------
    # theta_V and theta_C are defined at every cubeta, not only at the one being
    # explained. Drawing them thin on all three, and thick on one, says that the
    # thick stroke is an example and not a special case.
    for c, aze, azc in zip(centres, az_axis, az_cub):
        segment(ax, c, aze, 0.110, C_THV, lw=1.6, z=5)
        segment(ax, c, azc, 0.106, C_THC, lw=1.6, z=5)

    # ---- the sampling ring, drawn on every body ----------------------------
    # theta_V is read on a ring of points entirely outside the cubeta, on a DEM
    # from which the body has been removed. Drawing the ring on all three makes
    # clear that this is the rule and not a special case.
    for c in centres:
        ax.add_patch(Circle(c, RING_R, facecolor="none", edgecolor=C_THV,
                            lw=1.7, ls=(0, (6, 3, 1, 3)), zorder=5))

    # ---- grid north at each body -------------------------------------------
    for c in centres:
        ax.annotate("", xy=(c[0], c[1] + NORTH_LEN), xytext=(c[0], c[1]),
                    zorder=3,
                    arrowprops=dict(arrowstyle="-|>", color=C_NORTH, lw=1.4,
                                    shrinkA=0, shrinkB=0))

    # ---- theta_V on the left-hand body -------------------------------------
    c1, az1 = centres[0], az_axis[0]
    segment(ax, c1, az1, 0.165, C_THV, lw=4.0, z=6)
    sector(ax, c1, az1, SECTOR_R, C_THV, r"$\theta_V$")

    # ---- theta_N from the middle body to its neighbour ---------------------
    c2, c3 = centres[1], centres[2]
    v = c3 - c2
    az_n = float(np.degrees(np.arctan2(v[0], v[1]))) % 180.0
    # theta_N is axial, like the other two, so the line carries no head: it
    # runs centroid to centroid and the two ends are equivalent.
    ax.plot([c2[0], c3[0]], [c2[1], c3[1]], color=C_THN, lw=3.4,
            solid_capstyle="round", zorder=6)
    ax.scatter([c2[0], c3[0]], [c2[1], c3[1]], s=30, color=C_THN,
               edgecolors="white", linewidths=1.0, zorder=7)
    sector(ax, c2, az_n, SECTOR_R * 1.10, C_THN, r"$\theta_N$", lab_scale=2.15)

    # ---- theta_C on the right-hand body ------------------------------------
    segment(ax, c3, az_cub[2], 0.134, C_THC, lw=4.2, z=6)
    sector(ax, c3, az_cub[2], SECTOR_R, C_THC, r"$\theta_C$")

    # ---- key ----------------------------------------------------------------
    left = [
        Line2D([0], [0], color=C_THV, lw=4.0,
               label=r"$\theta_V$   valley orientation field"),
        Line2D([0], [0], color=C_THN, lw=3.2,
               label=r"$\theta_N$   cubeta alignment"),
        Line2D([0], [0], color=C_THC, lw=4.2,
               label=r"$\theta_C$   cubeta orientation"),
    ]
    right = [
        Line2D([0], [0], color=C_NORTH, lw=0, marker=r"$\uparrow$",
               markersize=15, label="grid north"),
        Line2D([0], [0], color=C_THV, lw=1.7, ls=(0, (6, 3, 1, 3)),
               label=r"sampling ring for $\theta_V$"),
        Patch(facecolor=C_BAND, edgecolor="none", label="cañada"),
        Line2D([0], [0], color=C_AXIS, lw=1.5, ls=(0, (7, 5)),
               label="cañada axis"),
    ]
    lg1 = fig.legend(handles=left, loc="lower left",
                     bbox_to_anchor=(0.09, 0.005), frameon=False,
                     fontsize=12.5, handlelength=2.3, labelspacing=0.85)
    fig.add_artist(lg1)
    fig.legend(handles=right, loc="lower left", bbox_to_anchor=(0.54, 0.005),
               frameon=False, fontsize=12.5, handlelength=2.3,
               labelspacing=0.85)

    print(f"[schematic] theta_V = {az1:.0f}deg  theta_N = {az_n:.0f}deg  "
          f"theta_C = {az_cub[2]:.0f}deg   (drawn, not measured)")

    for ext in ("png", "pdf"):
        fig.savefig(cfg.FIGURES / f"Figure_2.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {cfg.FIGURES / 'Figure_2.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
