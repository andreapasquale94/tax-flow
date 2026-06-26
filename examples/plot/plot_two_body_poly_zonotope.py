#!/usr/bin/env python3
"""Render the curved polynomial-zonotope figure from poly_zonotope.json.

Usage:
    python3 plot_two_body_poly_zonotope.py [--data DIR] [--out DIR]

Produces two_body_poly_zonotope.png with three panels:
    (a) the initial set in (x, y): the curved polynomial zonotope vs its tangent
        (linear) approximation, over the 10000-sample cloud;
    (b) propagated to T/2 — curved leaves over the cloud (tracks it);
    (c) propagated to T/2 — linear leaves over the cloud (misses the tails).
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt

plt.rcParams.update({"figure.facecolor": "white", "font.size": 11,
                     "xtick.direction": "in", "ytick.direction": "in"})

CURVED = "#1f77b4"
LINEAR = "#d62728"
MC = "0.4"


def load(path):
    with open(path) as f:
        return json.load(f)


def plot_ic(ax, d):
    ax.scatter(d["ic_mc"]["x"], d["ic_mc"]["y"], s=2, color=MC, alpha=0.5, zorder=1,
               label="Monte Carlo", rasterized=True)
    ax.plot(d["ic_linear"]["x"], d["ic_linear"]["y"], color=LINEAR, lw=2.0, ls="--",
            label="linear (tangent)", zorder=3)
    ax.plot(d["ic_curved"]["x"], d["ic_curved"]["y"], color=CURVED, lw=2.0,
            label="polynomial zonotope", zorder=4)
    ax.set_title("(a) initial set $(x, y)$")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.legend(loc="best", fontsize=8.5)


def plot_prop(ax, d, key, color, title):
    ax.plot(d["reference_orbit"]["x0"], d["reference_orbit"]["x1"], color="0.8", lw=0.8, zorder=1)
    xs, ys = list(d["mc"]["x"]), list(d["mc"]["y"])
    for leaf in d[key]:
        ax.fill(leaf["x"], leaf["y"], facecolor=color, alpha=0.25, edgecolor=color, lw=0.6,
                zorder=2)
        xs += leaf["x"]
        ys += leaf["y"]
    ax.scatter(d["mc"]["x"], d["mc"]["y"], s=2, color=MC, alpha=0.5, zorder=3, rasterized=True)
    px = 0.1 * (max(xs) - min(xs) + 1e-9)
    py = 0.1 * (max(ys) - min(ys) + 1e-9)
    ax.set_xlim(min(xs) - px, max(xs) + px)
    ax.set_ylim(min(ys) - py, max(ys) + py)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=pathlib.Path, default=".")
    ap.add_argument("--out", type=pathlib.Path, default=".")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = load(args.data / "poly_zonotope.json")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.3), layout="constrained")
    plot_ic(axes[0], d)
    plot_prop(axes[1], d, "leaves_curved", CURVED,
              f"(b) polynomial zonotope at $T/2$\n{d['n_curved']} leaves, "
              f"RMS {d['rms_curved']:.1e}, max {d['max_curved']:.1e}")
    plot_prop(axes[2], d, "leaves_linear", LINEAR,
              f"(c) linear (tangent) at $T/2$\n{d['n_linear']} leaves, "
              f"RMS {d['rms_linear']:.1e}, max {d['max_linear']:.1e}")
    fig.suptitle("Curved initial set: polynomial zonotope vs linear approximation (3σ in ν, e)",
                 fontsize=13)
    fig.savefig(args.out / "two_body_poly_zonotope.png", dpi=150)
    plt.close(fig)
    print(f"wrote two_body_poly_zonotope.png -> {args.out}")


if __name__ == "__main__":
    main()
