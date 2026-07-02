#!/usr/bin/env python3
"""Render the hyperbolic-flyby B-plane / closest-approach figures.

Usage:
    python3 plot_hyperbola_bplane.py [--data DIR] [--out DIR]

Reads:
    hyperbola_bplane.json      (written by hyperbola_bplane_taylor)
    hyperbola_bplane_ads.json  (written by hyperbola_bplane_ads, optional)

Produces:
    hyperbola_bplane.png       (a) the B-plane image with its first-order
                               zonotope (nearly exact — the impact parameter is
                               a near-linear function of the IC); (b) the
                               closest-approach POSITION with the box / zonotope
                               / polynomial-zonotope enclosure hierarchy, where
                               the flyby is genuinely nonlinear.
    hyperbola_event_tiling.png the ADS event surface (closest-approach position)
                               tiled by the box and by the oriented zonotope.
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
LEAF_COLOR = "#1f77b4"


def load(path):
    with open(path) as f:
        return json.load(f)


def parallelogram(c, J):
    c = np.array(c)
    J = np.array(J).reshape(2, 2)
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], dtype=float)
    return c[None, :] + corners @ J.T


def plot_bplane(ax, d):
    bp = d["bplane"]
    ax.scatter(bp["mc"]["bt"], bp["mc"]["br"], s=2, color=MC_COLOR, alpha=0.5,
               label="Monte Carlo (true set)", zorder=1, rasterized=True)
    pg = parallelogram(bp["c"], bp["J"])
    ax.plot(pg[:, 0], pg[:, 1], color=ZONO_COLOR, lw=2.0,
            label="first-order zonotope", zorder=5)
    ax.set_title("(a) B-plane image at closest approach\n"
                 f"e = {d['ecc']:.3f}, $v_\\infty$ = {d['vinf']:.3f} "
                 "(map is nearly linear)")
    ax.legend(loc="best", fontsize=8.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$B \cdot T$")
    ax.set_ylabel(r"$B \cdot R$")


def plot_capos(ax, d):
    c = d["capos"]
    ax.scatter(c["mc"]["x"], c["mc"]["y"], s=2, color=MC_COLOR, alpha=0.5,
               label="Monte Carlo (true set)", zorder=1, rasterized=True)
    x0, x1, y0, y1 = c["box"]
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
            color=BOX_COLOR, lw=1.6, ls="--", label="box (interval hull)", zorder=4)
    z = c["zonotope"]
    pg = parallelogram(z["c"], z["J"])
    ax.plot(pg[:, 0], pg[:, 1], color=ZONO_COLOR, lw=1.8,
            label="zonotope (1st-order image)", zorder=5)
    ax.plot(c["poly_zonotope"]["x"], c["poly_zonotope"]["y"], color=PZ_COLOR, lw=2.0,
            label="polynomial zonotope", zorder=6)
    ax.set_title(f"(b) closest-approach position\n$r_{{ca}}$ = {d['r_ca_const']:.4f} "
                 "(box $\\supseteq$ zonotope $\\supseteq$ poly)")
    ax.legend(loc="best", fontsize=8.5)
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def plot_tiling(ax, ads, dom, title):
    mc = ads["mc"]
    ax.scatter(mc["x"], mc["y"], s=2, color=MC_COLOR, alpha=0.45, zorder=1, rasterized=True)
    for leaf in dom["leaves"]:
        ax.fill(leaf["x"], leaf["y"], facecolor=LEAF_COLOR, alpha=0.20,
                edgecolor=LEAF_COLOR, lw=0.5, zorder=3)
    cx, cy = ads["center"]
    ax.plot([cx], [cy], "k+", ms=9, zorder=5)
    ax.set_title(f"{title}\n{dom['name']}, {dom['n_leaves']} leaves")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=pathlib.Path, default=".")
    ap.add_argument("--out", type=pathlib.Path, default=".")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = load(args.data / "hyperbola_bplane.json")
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4), layout="constrained")
    plot_bplane(axes[0], d)
    plot_capos(axes[1], d)
    fig.suptitle("Hyperbolic flyby: B-plane and closest-approach enclosures", fontsize=13)
    fig.savefig(args.out / "hyperbola_bplane.png", dpi=150)
    plt.close(fig)
    print(f"wrote hyperbola_bplane.png -> {args.out}")

    ads_path = args.data / "hyperbola_bplane_ads.json"
    if ads_path.exists():
        ads = load(ads_path)
        by_name = {dm["name"]: dm for dm in ads["domains"]}
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.4), layout="constrained")
        plot_tiling(axes[0], ads, by_name["bounding box"], "(a) box domain")
        plot_tiling(axes[1], ads, by_name["oriented zonotope"], "(b) oriented zonotope domain")
        fig.suptitle("Hyperbolic flyby: ADS tiles the closest-approach event surface", fontsize=13)
        fig.savefig(args.out / "hyperbola_event_tiling.png", dpi=150)
        plt.close(fig)
        print(f"wrote hyperbola_event_tiling.png -> {args.out}")


if __name__ == "__main__":
    main()
