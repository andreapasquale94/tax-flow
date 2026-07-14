#!/usr/bin/env python3
"""Render the adaptive oriented-domain figure from zonotope_adaptive.json.

Usage:
    python3 plot_two_body_zonotope_adaptive.py [--data DIR] [--out DIR]

Reads zonotope_adaptive.json written by two_body_zonotope_adaptive and
produces two_body_zonotope_adaptive.png with three panels:

    (a) the same ellipsoidal IC covered three ways in the (y, vy) plane —
        axis-aligned box, fixed Cholesky parallelotope, flow-aligned
        parallelotope;
    (b) leaf count per snapshot for the three coverings over a full period;
    (c) the flow-aligned leaf tiling in (x, y) at the final time.
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

BOX_COLOR = "#d62728"
CHOL_COLOR = "#7f7f7f"
ALIGN_COLOR = "#1f77b4"


def load(path: pathlib.Path):
    with open(path) as f:
        return json.load(f)


def parallelogram(center, block):
    """Corners of {center + G·ξ : ξ ∈ unit-square}, closed."""
    G = np.array(block, dtype=float).reshape(2, 2)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=float)
    return np.array(center)[None, :] + corners @ G.T


def ellipse(center, cov, n=200):
    C = np.array(cov, dtype=float).reshape(2, 2)
    L = np.linalg.cholesky(C)
    th = np.linspace(0, 2 * np.pi, n)
    circ = np.stack([np.cos(th), np.sin(th)], axis=1)
    return np.array(center)[None, :] + circ @ L.T


def plot_ic(ax, p):
    c = p["ic_center_yv"]
    el = ellipse(c, p["cov"])
    ax.plot(el[:, 0], el[:, 1], color="0.2", lw=1.2, ls=":", label="1σ ellipse", zorder=4)

    hy, hv = p["box_hw_yv"]
    bx = [c[0] - hy, c[0] + hy, c[0] + hy, c[0] - hy, c[0] - hy]
    by = [c[1] - hv, c[1] - hv, c[1] + hv, c[1] + hv, c[1] - hv]
    ax.plot(bx, by, color=BOX_COLOR, lw=1.6, ls="--", label="box", zorder=2)

    pc = parallelogram(c, p["chol_yv"])
    ax.plot(pc[:, 0], pc[:, 1], color=CHOL_COLOR, lw=1.6, label="Cholesky", zorder=2)

    pa = parallelogram(c, p["aligned_yv"])
    ax.fill(pa[:, 0], pa[:, 1], facecolor=ALIGN_COLOR, alpha=0.25,
            edgecolor=ALIGN_COLOR, lw=1.8, label="flow-aligned", zorder=3)

    ax.set_xlabel("$y$")
    ax.set_ylabel("$v_y$")
    ax.set_aspect("equal")
    ax.set_title("(a) one ellipsoid, three coverings")
    ax.legend(loc="upper left", fontsize=8)


def plot_counts(ax, p):
    box, chol, al = p["box_leaf_counts"], p["chol_leaf_counts"], p["aligned_leaf_counts"]
    tau = np.linspace(0.0, 1.0, len(box))
    ax.plot(tau, chol, "-o", color=CHOL_COLOR, lw=1.5, ms=4, label=f"Cholesky ({int(chol[-1])})")
    ax.plot(tau, box, "-o", color=BOX_COLOR, lw=1.5, ms=4, label=f"box ({int(box[-1])})")
    ax.plot(tau, al, "-o", color=ALIGN_COLOR, lw=1.8, ms=4, label=f"flow-aligned ({int(al[-1])})")
    ax.set_yscale("log")
    ax.set_xlabel("$t\\,/\\,T$")
    ax.set_ylabel("leaves")
    ax.set_title("(b) domains needed over a full period")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)


def plot_tiling(ax, data):
    ref = data["reference_orbit"]
    ax.plot(ref["x0"], ref["x1"], color="0.6", lw=1.0, zorder=1, label="reference orbit")
    ax.scatter([0], [0], marker="o", s=90, color="#f2b134", edgecolor="0.3", zorder=6)
    leaves = data["snapshots"][-1]["leaves"]
    for leaf in leaves:
        ax.fill(leaf["x"], leaf["y"], facecolor=ALIGN_COLOR, alpha=0.30,
                edgecolor=ALIGN_COLOR, lw=0.7, zorder=3)
    ax.fill([], [], color=ALIGN_COLOR, alpha=0.4, label=f"aligned leaves ({len(leaves)})")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    ax.set_title("(c) flow-aligned tiling at $t = T$")
    ax.legend(loc="upper left", fontsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=pathlib.Path, default=".",
                    help="directory containing zonotope_adaptive.json")
    ap.add_argument("--out", type=pathlib.Path, default=".",
                    help="output directory for the PNG figure")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = load(args.data / "zonotope_adaptive.json")
    p = data["params"]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), layout="constrained")
    plot_ic(axes[0], p)
    plot_counts(axes[1], p)
    plot_tiling(axes[2], data)

    fig.suptitle("Two-body ADS: flow-aligned covering vs box and fixed Cholesky frame",
                 fontsize=13)
    fig.savefig(args.out / "two_body_zonotope_adaptive.png", dpi=150)
    plt.close(fig)
    print(f"wrote two_body_zonotope_adaptive.png -> {args.out}")


if __name__ == "__main__":
    main()
