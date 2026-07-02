#!/usr/bin/env python3
"""Render the hyperbolic-flyby enclosure snapshots along the trajectory.

Usage:
    python3 plot_hyperbola_snapshots.py [--data DIR] [--out DIR]

Reads hyperbola_snapshots_zono.json and hyperbola_snapshots_pz.json (written by
hyperbola_snapshots) and produces hyperbola_snapshots.png: one panel per domain
(Zonotope, PolynomialZonotope) showing the (x, y) projection of the leaf tilings
at every snapshot time, coloured by time, over the reference flyby orbit. The
partition refines as the flow stretches through periapsis.
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt

plt.rcParams.update({"figure.facecolor": "white", "font.size": 11,
                     "xtick.direction": "in", "ytick.direction": "in"})


def load(path):
    with open(path) as f:
        return json.load(f)


def plot_domain(ax, data, title):
    ref = data["reference_orbit"]
    ax.plot(ref["x0"], ref["x1"], color="0.8", lw=1.0, zorder=1)
    ax.plot([0.0], [0.0], marker="*", color="0.4", ms=12, zorder=2)  # attractor

    snaps = data["snapshots"]
    cmap = plt.get_cmap("viridis")
    n = max(len(snaps) - 1, 1)
    counts = []
    for i, snap in enumerate(snaps):
        color = cmap(i / n)
        for leaf in snap["leaves"]:
            ax.fill(leaf["x"], leaf["y"], facecolor=color, alpha=0.35,
                    edgecolor=color, lw=0.4, zorder=3)
        counts.append(len(snap["leaves"]))

    ax.set_title(f"{title}\nleaves/snap: {', '.join(str(c) for c in counts)}")
    ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=pathlib.Path, default=".")
    ap.add_argument("--out", type=pathlib.Path, default=".")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    zono = load(args.data / "hyperbola_snapshots_zono.json")
    pz = load(args.data / "hyperbola_snapshots_pz.json")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6), layout="constrained")
    plot_domain(axes[0], zono, "Zonotope domain")
    plot_domain(axes[1], pz, "Polynomial-zonotope domain")

    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array([0.0, 1.0])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("snapshot (t = 0 .. t_final, periapsis mid-arc)")

    fig.suptitle("Hyperbolic flyby: enclosure tilings along the trajectory", fontsize=13)
    fig.savefig(args.out / "hyperbola_snapshots.png", dpi=150)
    plt.close(fig)
    print(f"wrote hyperbola_snapshots.png -> {args.out}")


if __name__ == "__main__":
    main()
