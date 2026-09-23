#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_fig2_schematic.py   (manuscript Figure 2)

Schematic of the three axial directional signals analysed in the paper:

    theta_C   cubeta orientation      long axis of an individual cubeta
    theta_N   cubeta alignment        azimuth to the nearest neighbour
    theta_V   valley orientation field  local direction of the cañada

Purely illustrative: nothing here is measured from data. Two idealised cubetas
sit inside a cañada whose axis runs NNE, the arrangement the analysis quantifies.
Angles are drawn from grid north, clockwise, as axial quantities (0-180 deg).

No burned-in title: the caption goes in the manuscript, like the other figures.

Output : figures/Figure_2.{png,pdf}
Project: cubetas-morphometry | Python 3.11 | matplotlib only
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Arc, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import config as cfg
FIG_DIR = cfg.FIGURES

# ---- colours: same family as the other figures, colourblind-safe -----------
C_CUB   = "#4477AA"     # cubeta fill
C_CUBE  = "#2b4c6f"     # cubeta outline
C_THC   = cfg.SIGNAL_COLORS["theta_C"]   # cubeta orientation
C_THN   = cfg.SIGNAL_COLORS["theta_N"]   # cubeta alignment
C_THV   = cfg.SIGNAL_COLORS["theta_V"]   # valley orientation field
C_VALL  = "#c8d8e4"     # cañada band
C_NORTH = "#444444"

# ---- geometry (map units, arbitrary) --------------------------------------
AZ_VALLEY = 22.0        # cañada axis, NNE
AZ_C1     = 44.0        # orientation of cubeta 1: departs from the cañada
AZ_C2     = 8.0         # orientation of cubeta 2
P3_OFF    = 1.35        # theta_V is drawn up-valley of the ring, still inside the cañada
P1 = np.array([-0.90, -1.25])
P2 = np.array([0.85, 1.15])


def unit(az_deg):
    """Unit vector for an azimuth measured clockwise from north."""
    a = np.deg2rad(az_deg)
    return np.array([np.sin(a), np.cos(a)])


def axial_segment(ax, centre, az, half, color, lw=2.4, z=5, ls="-"):
    v = unit(az) * half
    ax.plot([centre[0] - v[0], centre[0] + v[0]],
            [centre[1] - v[1], centre[1] + v[1]],
            color=color, lw=lw, zorder=z, ls=ls, solid_capstyle="round")


def north_dashes(ax, centre, length, color=C_NORTH):
    ax.plot([centre[0], centre[0]], [centre[1], centre[1] + length],
            color=color, lw=1.0, ls=(0, (4, 3)), zorder=4)


def angle_arc(ax, centre, az, radius, color, label, lab_r=None, lab_az=None):
    """Arc from grid north to the azimuth, with its label."""
    # matplotlib Arc measures degrees anticlockwise from +x; north is 90
    th1, th2 = 90 - az, 90
    ax.add_patch(Arc(centre, 2 * radius, 2 * radius, angle=0,
                     theta1=th1, theta2=th2, color=color, lw=1.6, zorder=6))
    lr = lab_r if lab_r is not None else radius * 1.30
    la = lab_az if lab_az is not None else az / 2.0
    q = centre + unit(la) * lr
    ax.text(q[0], q[1], label, color=color, fontsize=12, ha="center",
            va="center", zorder=8,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.85))


def main():
    fig, ax = plt.subplots(figsize=(6.2, 5.7))

    # ---- cañada: a soft band along AZ_VALLEY -----------------------------
    v = unit(AZ_VALLEY)
    n = np.array([-v[1], v[0]])
    L, W = 4.8, 0.95
    for w, alpha in [(W * 1.7, 0.32), (W, 0.62)]:
        c = np.array([-v * L + n * w, v * L + n * w, v * L - n * w, -v * L - n * w])
        ax.fill(c[:, 0], c[:, 1], color=C_VALL, alpha=alpha, lw=0, zorder=0)
    # axis of the cañada, drawn along its whole length in the theta_V colour:
    # theta_V is a continuous field, not a single local reading
    ax.plot([-v[0] * L, v[0] * L], [-v[1] * L, v[1] * L], color=C_THV,
            lw=1.3, ls=(0, (9, 6)), alpha=0.55, zorder=1,
            solid_capstyle="round")
    lab = -v * 2.55
    ax.text(lab[0], lab[1], "cañada", color="#5d7a8c", fontsize=12,
            style="italic", ha="center", va="center", rotation=-AZ_VALLEY, zorder=3)

    # ---- two cubetas -----------------------------------------------------
    E1, E2 = (0.78, 1.42), (0.80, 0.98)
    ax.add_patch(Ellipse(P1, *E1, angle=-AZ_C1, facecolor=C_CUB,
                         edgecolor=C_CUBE, lw=1.4, alpha=0.85, zorder=3))
    ax.add_patch(Ellipse(P2, *E2, angle=-AZ_C2, facecolor=C_CUB,
                         edgecolor=C_CUBE, lw=1.4, alpha=0.85, zorder=3))

    # ---- theta_N: from cubeta 1 to its nearest neighbour -----------------
    d = P2 - P1
    az_n = (np.degrees(np.arctan2(d[0], d[1]))) % 180
    ax.add_patch(FancyArrowPatch(tuple(P1), tuple(P2), arrowstyle="-|>",
                                 mutation_scale=14, color=C_THN, lw=1.9,
                                 shrinkA=0, shrinkB=0, zorder=4))
    # centroids made explicit: theta_N runs centroid to centroid
    ax.scatter([P1[0], P2[0]], [P1[1], P2[1]], s=26, color=C_THN,
               edgecolors="white", linewidths=0.9, zorder=7)

    # ---- theta_C on cubeta 1: the long axis of the ellipse itself --------
    north_dashes(ax, P1, 1.30)
    axial_segment(ax, P1, AZ_C1, E1[1] / 2 * 1.06, C_THC, lw=2.8, z=6)
    angle_arc(ax, P1, AZ_C1, 0.88, C_THC, r"$\theta_C$", lab_r=1.16, lab_az=AZ_C1 * 0.52)

    # ---- theta_N arc, on the same centre but wider -----------------------
    angle_arc(ax, P1, az_n, 1.62, C_THN, r"$\theta_N$", lab_r=1.88, lab_az=az_n * 0.60)

    # ---- theta_V: measured OUTSIDE the cubeta, on a ring around it -------
    # (Section 3.5: the cubetas are masked out of the DEM and the field is
    #  sampled on a ring entirely outside each body)
    ring_r = 0.95
    ang = np.linspace(0, 2 * np.pi, 13)[:-1]
    ax.plot(P2[0] + ring_r * np.cos(ang), P2[1] + ring_r * np.sin(ang),
            ls=":", color="#8899a6", lw=1.0, zorder=2)
    ax.scatter(P2[0] + ring_r * np.cos(ang), P2[1] + ring_r * np.sin(ang),
               s=9, facecolors="white", edgecolors="#6b7f8c", linewidths=0.9,
               zorder=5)
    P3 = P2 + unit(AZ_VALLEY) * P3_OFF
    north_dashes(ax, P3, 0.95)
    axial_segment(ax, P3, AZ_VALLEY, 0.68, C_THV, lw=2.8, z=6)
    angle_arc(ax, P3, AZ_VALLEY, 0.74, C_THV, r"$\theta_V$", lab_r=1.02,
              lab_az=AZ_VALLEY * 0.5)

    # ---- north arrow -----------------------------------------------------
    ax.annotate("", xy=(-2.45, 2.90), xytext=(-2.45, 2.15),
                arrowprops=dict(arrowstyle="-|>", color=C_NORTH, lw=1.8))
    ax.text(-2.45, 3.02, "N", color=C_NORTH, fontsize=13, fontweight="bold",
            ha="center", va="bottom")

    # ---- key, below the drawing so it covers nothing ---------------------
    handles = [
        plt.Line2D([0], [0], color=C_THC, lw=2.8,
                   label=r"$\theta_C$   cubeta orientation"),
        plt.Line2D([0], [0], color=C_THN, lw=1.9,
                   label=r"$\theta_N$   cubeta alignment"),
        plt.Line2D([0], [0], color=C_THV, lw=2.8,
                   label=r"$\theta_V$   valley orientation field"),
        plt.Line2D([0], [0], color=C_NORTH, lw=1.0, ls=(0, (4, 3)),
                   label="grid north"),
        plt.Line2D([0], [0], marker="o", ls="none", markerfacecolor="white",
                   markeredgecolor="#6b7f8c", markersize=5,
                   label=r"sampling ring for $\theta_V$"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=2, fontsize=9.5, frameon=False, columnspacing=2.0,
              labelspacing=0.7, handlelength=2.1)

    ax.set_xlim(-3.0, 3.0); ax.set_ylim(-3.3, 3.4)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"Figure_2.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] {FIG_DIR / 'Figure_2.png'} (+ .pdf)")


if __name__ == "__main__":
    main()
