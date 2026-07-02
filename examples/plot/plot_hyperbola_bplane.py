#!/usr/bin/env python3
"""Render the hyperbolic-flyby B-plane figures.

Usage:
    python3 plot_hyperbola_bplane.py [--data DIR] [--out DIR]

Reads:
    hyperbola_bplane.json      (written by hyperbola_bplane_taylor)
    hyperbola_bplane_ads.json  (written by hyperbola_bplane_ads, optional)

Produces hyperbola_bplane.png:
    (a) enclosures of the (B.T, B.R) image of one Taylor flow map — the box
        interval hull, the linear zonotope (first-order image), and the
        polynomial-zonotope outline — over the Monte-Carlo truth cloud;
    (b) the box-domain ADS event tiling over the same cloud;
    (c) the oriented-zonotope-domain ADS event tiling.
Panels (b)/(c) are omitted if hyperbola_bplane_ads.json is absent.
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


def plot_hierarchy(ax, d):
    ax.scatter(d["mc"]["bt"], d["mc"]["br"], s=2, color=MC_COLOR, alpha=0.5,
               label="Monte Carlo (true set)", zorder=1, rasterized=True)

    x0, x1, y0, y1 = d["box"]
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
            color=BOX_COLOR, lw=1.6, ls="--", label="box (interval hull)", zorder=4)

    z = d["zonotope"]
    pg = parallelogram(z["c"], z["J"])
    ax.plot(pg[:, 0], pg[:, 1], color=ZONO_COLOR, lw=1.8,
            label="zonotope (1st-order image)", zorder=5)

    ax.plot(d["poly_zonotope"]["bt"], d["poly_zonotope"]["br"], color=PZ_COLOR, lw=2.0,
            label="polynomial zonotope", zorder=6)

    ax.set_title("(a) B-plane enclosures of one Taylor flow map\n"
                 f"e = {d['ecc']:.3f}, $v_\\infty$ = {d['vinf']:.3f}, "
                 f"$r_{{ca}}$ = {d['r_ca_const']:.4f}")
    ax.legend(loc="best", fontsize=8.5)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$B \cdot T$")
    ax.set_ylabel(r"$B \cdot R$")


def plot_tiling(ax, ads, dom, mc, title):
    ax.scatter(mc["bt"], mc["br"], s=2, color=MC_COLOR, alpha=0.45, zorder=1, rasterized=True)
    for leaf in dom["leaves"]:
        ax.fill(leaf["x"], leaf["y"], facecolor=LEAF_COLOR, alpha=0.20,
                edgecolor=LEAF_COLOR, lw=0.4, zorder=3)
    bt0, br0 = ads["b_center"]
    ax.plot([bt0], [br0], "k+", ms=9, zorder=5)
    ax.set_title(f"{title}\n{dom['name']}, {dom['n_leaves']} leaves")
    ax.set_aspect("equal")
    ax.set_xlabel(r"$B \cdot T$")
    ax.set_ylabel(r"$B \cdot R$")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=pathlib.Path, default=".")
    ap.add_argument("--out", type=pathlib.Path, default=".")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = load(args.data / "hyperbola_bplane.json")
    ads_path = args.data / "hyperbola_bplane_ads.json"
    ads = load(ads_path) if ads_path.exists() else None

    npanels = 1 if ads is None else 3
    fig, axes = plt.subplots(1, npanels, figsize=(5.6 * npanels, 5.4), layout="constrained")
    if npanels == 1:
        axes = [axes]

    plot_hierarchy(axes[0], d)
    if ads is not None:
        by_name = {dm["name"]: dm for dm in ads["domains"]}
        plot_tiling(axes[1], ads, by_name["bounding box"], ads["mc"],
                    "(b) ADS event tiling — box")
        plot_tiling(axes[2], ads, by_name["oriented zonotope"], ads["mc"],
                    "(c) ADS event tiling — zonotope")

    fig.suptitle("Hyperbolic flyby: B-plane enclosures at closest approach", fontsize=13)
    fig.savefig(args.out / "hyperbola_bplane.png", dpi=150)
    plt.close(fig)
    print(f"wrote hyperbola_bplane.png -> {args.out}")


if __name__ == "__main__":
    main()
