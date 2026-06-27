#!/usr/bin/env python3
"""Render the oriented-domain (Zonotope) two-body figure from zonotope.json.

Usage:
    python3 plot_two_body_zonotope.py [--data DIR] [--out DIR]

Reads zonotope.json (and zonotope_box.json for the overlay) written by
two_body_zonotope (run it first; it writes to its working directory) and
produces:

    two_body_zonotope.png — three panels:
      (a) the IC set in the (y, vy) plane: oriented parallelotope vs the
          axis-aligned box that bounds it;
      (b) leaf count per snapshot, oriented Zonotope vs axis-aligned Box;
      (c) the leaf tiling in (x, y) at the final snapshot, both domains.
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "font.size": 11,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.8",
    }
)

ZONO_COLOR = "#1f77b4"
BOX_COLOR = "#d62728"


def load(path: pathlib.Path):
    with open(path) as f:
        return json.load(f)


def plot_ic_sets(ax, params):
    """Oriented parallelotope (G·ξ) vs its axis-aligned bounding box, (y, vy)."""
    cy, cv = params["ic_center_yv"]
    g11, g13, g31, g33 = params["ic_gen_yv"]
    hy, hv = params["bbox_hw_yv"]
    G = np.array([[g11, g13], [g31, g33]])
    c = np.array([cy, cv])

    # Parallelotope corners: image of the unit-square corners under ξ ↦ c + Gξ.
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=float)
    poly = (c[None, :] + corners @ G.T)
    ax.fill(poly[:, 0], poly[:, 1], facecolor=ZONO_COLOR, alpha=0.30,
            edgecolor=ZONO_COLOR, lw=1.8, label="oriented Zonotope", zorder=3)

    # Axis-aligned bounding box.
    bx = [cy - hy, cy + hy, cy + hy, cy - hy, cy - hy]
    by = [cv - hv, cv - hv, cv + hv, cv + hv, cv - hv]
    ax.plot(bx, by, color=BOX_COLOR, lw=1.8, ls="--", label="bounding Box", zorder=2)

    ax.scatter([cy], [cv], marker="+", s=80, color="0.2", zorder=5)
    ax.set_xlabel("$y$")
    ax.set_ylabel("$v_y$")
    ax.set_aspect("equal")
    ax.set_title("(a) initial set, $(y, v_y)$ plane")
    ax.legend(loc="upper left", fontsize=9)


def plot_leaf_counts(ax, params, t_final):
    """Leaf count per snapshot for both domains."""
    zc = params["zono_leaf_counts"]
    bc = params["box_leaf_counts"]
    tau = np.linspace(0.0, 1.0, len(zc))
    ax.plot(tau, bc, "-o", color=BOX_COLOR, lw=1.6, ms=4, label="axis-aligned Box")
    ax.plot(tau, zc, "-o", color=ZONO_COLOR, lw=1.6, ms=4, label="oriented Zonotope")
    ax.set_xlabel("$t\\,/\\,t_\\mathrm{final}$")
    ax.set_ylabel("leaves")
    ax.set_title("(b) domains needed vs time")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.25)


def plot_tiling(ax, zono, box):
    """Leaf tiling in (x, y) at the final snapshot, both domains overlaid."""
    ref = zono["reference_orbit"]
    ax.plot(ref["x0"], ref["x1"], color="0.6", lw=1.0, zorder=1, label="reference orbit")
    ax.scatter([0], [0], marker="o", s=90, color="#f2b134", edgecolor="0.3", zorder=6)

    for leaf in box["snapshots"][-1]["leaves"]:
        ax.fill(leaf["x"], leaf["y"], facecolor="none", edgecolor=BOX_COLOR,
                lw=0.8, alpha=0.8, zorder=2)
    for leaf in zono["snapshots"][-1]["leaves"]:
        ax.fill(leaf["x"], leaf["y"], facecolor=ZONO_COLOR, alpha=0.25,
                edgecolor=ZONO_COLOR, lw=0.7, zorder=3)

    nz = len(zono["snapshots"][-1]["leaves"])
    nb = len(box["snapshots"][-1]["leaves"])
    ax.plot([], [], color=BOX_COLOR, lw=1.2, label=f"Box leaves ({nb})")
    ax.fill([], [], color=ZONO_COLOR, alpha=0.4, label=f"Zonotope leaves ({nz})")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    ax.set_title("(c) leaf tiling at final time")
    ax.legend(loc="upper left", fontsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=pathlib.Path, default=".",
                    help="directory containing zonotope.json / zonotope_box.json")
    ap.add_argument("--out", type=pathlib.Path, default=".",
                    help="output directory for the PNG figure")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    zono = load(args.data / "zonotope.json")
    box = load(args.data / "zonotope_box.json")
    params = zono["params"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), layout="constrained")
    plot_ic_sets(axes[0], params)
    plot_leaf_counts(axes[1], params, params["t_final"])
    plot_tiling(axes[2], zono, box)

    fig.suptitle("Two-body ADS: oriented Zonotope domain vs axis-aligned Box", fontsize=13)
    fig.savefig(args.out / "two_body_zonotope.png", dpi=150)
    plt.close(fig)
    print(f"wrote two_body_zonotope.png -> {args.out}")


if __name__ == "__main__":
    main()
