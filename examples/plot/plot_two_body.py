#!/usr/bin/env python3
"""Render the two-body example figures from taylor.json / ads.json / loads.json.

Usage:
    python3 plot_two_body.py [--data DIR] [--out DIR]

Reads the JSON files written by two_body_taylor / two_body_ads /
two_body_loads (run them first; they write to their working directory)
and produces:

    two_body_flow.png    — IC box image under one flow polynomial, one orbit
    two_body_ads.png     — single polynomial vs ADS leaves near periapsis
    two_body_leaves.png  — leaf counts per snapshot, ADS vs LOADS
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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


def load(path: pathlib.Path):
    with open(path) as f:
        return json.load(f)


def plot_flow(taylor, out_dir):
    """Reference orbit + the IC-box image under the single flow polynomial."""
    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    ref = taylor["reference_orbit"]
    ax.plot(ref["x0"], ref["x1"], color="0.55", lw=1.0, zorder=1, label="reference orbit")
    ax.scatter([0], [0], marker="o", s=120, color="#f2b134", edgecolor="0.3",
               zorder=5, label="primary")

    snaps = taylor["snapshots"]
    t_final = taylor["params"]["t_final"]
    norm = Normalize(0.0, 1.0)
    cmap = plt.get_cmap("viridis")
    for snap in snaps:
        color = cmap(norm(snap["t"] / t_final))
        for leaf in snap["leaves"]:
            ax.fill(leaf["x"], leaf["y"], facecolor=color, alpha=0.45,
                    edgecolor=color, lw=1.4, zorder=3)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("$t\\,/\\,T$")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "two_body_flow.png", dpi=150)
    plt.close(fig)


def plot_ads(taylor, ads, out_dir):
    """Single polynomial vs ADS leaves at the last two snapshots."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6), layout="constrained",
                             width_ratios=[1.45, 1.0])
    picks = [-2, -1]  # the two final snapshots
    cmap = plt.get_cmap("viridis")

    for ax, k in zip(axes, picks):
        st, sa = taylor["snapshots"][k], ads["snapshots"][k]
        ref = taylor["reference_orbit"]
        ax.plot(ref["x0"], ref["x1"], color="0.55", lw=1.0, zorder=1)
        ax.scatter([0], [0], marker="o", s=100, color="#f2b134", edgecolor="0.3", zorder=5)

        leaf = st["leaves"][0]
        ax.plot(leaf["x"], leaf["y"], color="#d1495b", lw=1.6, ls="--", zorder=4,
                label="TTE")

        n = len(sa["leaves"])
        for i, lf in enumerate(sa["leaves"]):
            c = cmap(0.15 + 0.7 * i / max(n - 1, 1))
            ax.fill(lf["x"], lf["y"], facecolor=c, alpha=0.55, edgecolor=c, lw=1.0,
                    zorder=3, label="ADS" if i == 0 else None)

        # Zoom to the ADS set (the trustworthy one); the single-polynomial
        # tail running out of frame is the point of the figure.
        ax_all = np.concatenate([lf["x"] for lf in sa["leaves"]])
        ay_all = np.concatenate([lf["y"] for lf in sa["leaves"]])
        pad_x = 0.22 * (ax_all.max() - ax_all.min())
        pad_y = 0.22 * (ay_all.max() - ay_all.min())
        ax.set_xlim(ax_all.min() - pad_x, ax_all.max() + pad_x)
        ax.set_ylim(ay_all.min() - pad_y, ay_all.max() + pad_y)

        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="upper left", title=f"$t = {st['t']:.2f}$   ({n} leaves)")

    fig.savefig(out_dir / "two_body_ads.png", dpi=150)
    plt.close(fig)


def plot_ads_orbit(ads, out_dir):
    """All ADS snapshots overlaid on the full orbit."""
    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    ref = ads["reference_orbit"]
    ax.plot(ref["x0"], ref["x1"], color="0.55", lw=1.0, zorder=1, label="reference orbit")
    ax.scatter([0], [0], marker="o", s=120, color="#f2b134", edgecolor="0.3",
               zorder=5, label="primary")

    snaps = ads["snapshots"]
    t_final = ads["params"]["t_final"]
    norm = Normalize(0.0, 1.0)
    cmap = plt.get_cmap("viridis")
    for snap in snaps:
        color = cmap(norm(snap["t"] / t_final))
        for leaf in snap["leaves"]:
            ax.fill(leaf["x"], leaf["y"], facecolor=color, alpha=0.45,
                    edgecolor=color, lw=1.0, zorder=3)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("$t\\,/\\,T$")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_dir / "two_body_ads_orbit.png", dpi=150)
    plt.close(fig)


def plot_leaves(ads, loads, out_dir):
    """Leaf count growth per snapshot, ADS vs LOADS."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for run, color, marker, label in [
        (ads, "#1f77b4", "o", f"ADS (truncation, $P={ads['params']['P']}$)"),
        (loads, "#d1495b", "s", f"LOADS (NLI, $P={loads['params']['P']}$)"),
    ]:
        t = [s["t"] for s in run["snapshots"]]
        n = [len(s["leaves"]) for s in run["snapshots"]]
        ax.plot(t, n, marker=marker, color=color, lw=1.6, label=label)

    ax.set_xlabel("snapshot time  $t$")
    ax.set_ylabel("number of leaves")
    ax.set_yscale("log", base=2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "two_body_leaves.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=pathlib.Path, default=".",
                    help="directory containing taylor.json / ads.json / loads.json")
    ap.add_argument("--out", type=pathlib.Path, default=".",
                    help="output directory for the PNG figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    taylor = load(args.data / "taylor.json")
    ads = load(args.data / "ads.json")
    loads = load(args.data / "loads.json")

    plot_flow(taylor, args.out)
    plot_ads(taylor, ads, args.out)
    plot_ads_orbit(ads, args.out)
    plot_leaves(ads, loads, args.out)
    print(f"wrote two_body_flow.png, two_body_ads.png, two_body_ads_orbit.png, two_body_leaves.png -> {args.out}")


if __name__ == "__main__":
    main()
