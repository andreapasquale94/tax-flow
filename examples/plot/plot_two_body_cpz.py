#!/usr/bin/env python3
"""Render the constrained-polynomial-zonotope figure from cpz.json.

Usage:
    python3 plot_two_body_cpz.py [--data DIR] [--out DIR]

Produces two_body_cpz.png with three panels:
    (a) the (y, vy) box, the fixed-energy curve, and the feasible band of
        sub-boxes that survives constraint pruning;
    (b) leaf count vs subdivision depth — unpruned (area) vs feasible (length);
    (c) the propagated set at t = T/2: the 2-D ADS tiling of the whole box vs
        the 1-D constrained image, over 10000 Monte-Carlo samples.
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

plt.rcParams.update({"figure.facecolor": "white", "font.size": 11,
                     "xtick.direction": "in", "ytick.direction": "in"})

BLOB_COLOR = "#9ecae1"
CPZ_COLOR = "#d62728"
MC_COLOR = "0.35"
CURVE_COLOR = "#1f77b4"


def load(path):
    with open(path) as f:
        return json.load(f)


def plot_ic(ax, d):
    cy, cv = d["center_yv"]
    hy, hv = d["hw_yv"]
    ax.add_patch(Rectangle((cy - hy, cv - hv), 2 * hy, 2 * hv, fill=False,
                           edgecolor="0.4", lw=1.4, label="uncertainty box"))
    for (bcy, bcv, bhy, bhv) in d["band"]:
        ax.add_patch(Rectangle((bcy - bhy, bcv - bhv), 2 * bhy, 2 * bhv,
                               facecolor=CPZ_COLOR, alpha=0.35, edgecolor="none"))
    ax.plot(d["curve_yv"]["y"], d["curve_yv"]["v"], color=CURVE_COLOR, lw=2.0,
            label="energy curve $E=E_0$")
    ax.add_patch(Rectangle((0, 0), 0, 0, facecolor=CPZ_COLOR, alpha=0.5,
                           label="feasible band"))
    ax.set_xlim(cy - 1.12 * hy, cy + 1.12 * hy)
    ax.set_ylim(cv - 1.12 * hv, cv + 1.12 * hv)
    ax.set_xlabel("$y$")
    ax.set_ylabel("$v_y$")
    ax.set_title("(a) constrained IC set in $(y, v_y)$")
    ax.legend(loc="upper center", fontsize=8.5)


def plot_convergence(ax, d):
    c = d["convergence"]
    ax.semilogy(c["depth"], c["full"], "-o", color="0.5", ms=3,
                label="unpruned box ($2^\\mathrm{depth}$)")
    ax.semilogy(c["depth"], c["feasible"], "-o", color=CPZ_COLOR, ms=3,
                label="feasible (on constraint)")
    ax.set_xlabel("subdivision depth")
    ax.set_ylabel("leaves")
    ax.set_title("(b) constraint pruning: area → length")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)


def plot_image(ax, d):
    ax.plot(d["reference_orbit"]["x0"], d["reference_orbit"]["x1"], color="0.8", lw=0.8, zorder=1)
    for leaf in d["blob"]:
        ax.fill(leaf["x"], leaf["y"], facecolor=BLOB_COLOR, alpha=0.5,
                edgecolor="#6baed6", lw=0.4, zorder=2)
    ax.scatter(d["mc"]["x"], d["mc"]["y"], s=3, color=MC_COLOR, alpha=0.6, zorder=3,
               rasterized=True)
    for seg in d["cpz_curves"]:
        ax.plot(seg["x"], seg["y"], color=CPZ_COLOR, lw=2.2, zorder=4)
    ax.fill([], [], facecolor=BLOB_COLOR, alpha=0.7,
            label=f"2-D box ADS ({d['n_full_ads']} leaves)")
    ax.plot([], [], color=CPZ_COLOR, lw=2.2, label=f"constrained ({d['n_cpz']} leaves)")
    ax.scatter([], [], s=8, color=MC_COLOR, label="Monte Carlo (10000)")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    ax.set_title(f"(c) propagated set at $t=T/2$   (RMS {d['rms']:.1e})")
    ax.legend(loc="best", fontsize=8.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=pathlib.Path, default=".")
    ap.add_argument("--out", type=pathlib.Path, default=".")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = load(args.data / "cpz.json")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), layout="constrained")
    plot_ic(axes[0], d)
    plot_convergence(axes[1], d)
    plot_image(axes[2], d)
    fig.suptitle("Constrained polynomial zonotope: fixed-energy initial set on the two-body problem",
                 fontsize=13)
    fig.savefig(args.out / "two_body_cpz.png", dpi=150)
    plt.close(fig)
    print(f"wrote two_body_cpz.png -> {args.out}")


if __name__ == "__main__":
    main()
