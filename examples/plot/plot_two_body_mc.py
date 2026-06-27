#!/usr/bin/env python3
"""Render the two-body Monte-Carlo validation figures.

Usage:
    python3 plot_two_body_mc.py [--data DIR] [--out DIR]

Reads two_body_mc_parallelotope.json and two_body_mc_ellipse.json written by
zonotope_two_body_mc and produces, per scenario, one panel per covering domain
showing the 10000-sample Monte-Carlo cloud under that domain's leaf tiling,
titled with the leaf count and the RMS prediction error:

    two_body_mc_parallelotope.png
    two_body_mc_ellipse.png
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt

plt.rcParams.update({"figure.facecolor": "white", "font.size": 11,
                     "xtick.direction": "in", "ytick.direction": "in"})

LEAF_COLOR = "#1f77b4"
MC_COLOR = "0.55"


def load(path):
    with open(path) as f:
        return json.load(f)


def plot_scenario(data, out_path, title):
    doms = data["domains"]
    fig, axes = plt.subplots(1, len(doms), figsize=(5.6 * len(doms), 5.4), layout="constrained")
    if len(doms) == 1:
        axes = [axes]

    # Zoom to the final-time set (the MC cloud), with a small margin, so the
    # leaf tiling vs cloud match is visible instead of the whole orbit.
    mx, my = data["mc"]["x"], data["mc"]["y"]
    pad_x = 0.08 * (max(mx) - min(mx) + 1e-9)
    pad_y = 0.08 * (max(my) - min(my) + 1e-9)
    xlim = (min(mx) - pad_x, max(mx) + pad_x)
    ylim = (min(my) - pad_y, max(my) + pad_y)

    for ax, dom in zip(axes, doms):
        ax.plot(data["reference_orbit"]["x0"], data["reference_orbit"]["x1"],
                color="0.8", lw=0.8, zorder=1)
        for leaf in dom["leaves"]:
            ax.fill(leaf["x"], leaf["y"], facecolor=LEAF_COLOR, alpha=0.20,
                    edgecolor=LEAF_COLOR, lw=0.4, zorder=2)
        ax.scatter(mx, my, s=2, color=MC_COLOR, alpha=0.45, zorder=4, rasterized=True)
        ax.set_title(f"{dom['name']}\n{dom['n_leaves']} leaves, "
                     f"RMS {dom['rms']:.1e}, max {dom['max_err']:.1e}")
        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

    fig.suptitle(title, fontsize=13)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=pathlib.Path, default=".")
    ap.add_argument("--out", type=pathlib.Path, default=".")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    par = load(args.data / "two_body_mc_parallelotope.json")
    plot_scenario(par, args.out / "two_body_mc_parallelotope.png",
                  "Two-body Monte-Carlo validation — oriented parallelotope (0.8 orbit): "
                  "leaf tiling over 10000 samples")

    ell = load(args.data / "two_body_mc_ellipse.json")
    plot_scenario(ell, args.out / "two_body_mc_ellipse.png",
                  "Two-body Monte-Carlo validation — covariance ellipsoid (full orbit): "
                  "leaf tiling over 10000 samples")

    print(f"wrote two_body_mc_parallelotope.png, two_body_mc_ellipse.png -> {args.out}")


if __name__ == "__main__":
    main()
