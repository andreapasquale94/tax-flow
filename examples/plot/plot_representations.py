#!/usr/bin/env python3
"""Render the set-representation hierarchy from representations.json.

Usage:
    python3 plot_representations.py [--data DIR] [--out DIR]

Produces representations.png with three panels:
    (a) box ⊇ zonotope ⊇ polynomial zonotope = true set (MC cloud);
    (b) the union of 4 per-sub-box linear zonotopes;
    (c) the union of 16 — converging onto the curved set.
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.facecolor": "white", "font.size": 11,
                     "xtick.direction": "in", "ytick.direction": "in"})

MC_COLOR = "0.7"
BOX_COLOR = "#d62728"
ZONO_COLOR = "#ff7f0e"
PZ_COLOR = "#1f77b4"


def load(path):
    with open(path) as f:
        return json.load(f)


def parallelogram(c, J):
    c = np.array(c)
    J = np.array(J).reshape(2, 2)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=float)
    return c[None, :] + corners @ J.T


def mc_underlay(ax, d):
    ax.scatter(d["mc"]["x"], d["mc"]["y"], s=2, color=MC_COLOR, alpha=0.5,
               label="Monte Carlo (true set)", zorder=1, rasterized=True)


def plot_hierarchy(ax, d):
    mc_underlay(ax, d)

    x0, x1, y0, y1 = d["box"]
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
            color=BOX_COLOR, lw=1.6, ls="--", label="box", zorder=4)

    z = d["zonotope"]
    pz = parallelogram(z["c"], z["J"])
    ax.plot(pz[:, 0], pz[:, 1], color=ZONO_COLOR, lw=1.8, ls="-",
            label="linear image (1st order)", zorder=5)

    ax.plot(d["poly_zonotope"]["x"], d["poly_zonotope"]["y"], color=PZ_COLOR, lw=2.0,
            label="polynomial zonotope (= true set)", zorder=6)

    ax.set_title("(a) box enclosure · linear image · polynomial zonotope")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def plot_split(ax, d, key, label):
    mc_underlay(ax, d)
    for f in d[key]:
        pg = parallelogram(f["c"], f["J"])
        ax.fill(pg[:, 0], pg[:, 1], facecolor=PZ_COLOR, alpha=0.25, edgecolor=PZ_COLOR,
                lw=0.8, zorder=3)
    ax.plot(d["poly_zonotope"]["x"], d["poly_zonotope"]["y"], color="0.3", lw=1.0, ls=":",
            zorder=6)
    ax.set_title(label)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=pathlib.Path, default=".")
    ap.add_argument("--out", type=pathlib.Path, default=".")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = load(args.data / "representations.json")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), layout="constrained")
    plot_hierarchy(axes[0], d)
    plot_split(axes[1], d, "split2", "(b) piecewise linear images, 4 sub-boxes")
    plot_split(axes[2], d, "split4", "(c) piecewise linear images, 16 sub-boxes")
    fig.suptitle("Set representations: box, linear image, polynomial zonotope, and ADS splitting",
                 fontsize=13)
    fig.savefig(args.out / "representations.png", dpi=150)
    plt.close(fig)
    print(f"wrote representations.png -> {args.out}")


if __name__ == "__main__":
    main()
