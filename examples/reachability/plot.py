#!/usr/bin/env python3
"""examples/reachability/plot.py

Render the low-thrust reachable set from reachability.json (written by the
`reachability` example). Each snapshot's ADS leaves are drawn as NON-FILLED
outlines, coloured by t / T, overlaid on the ballistic reference orbit.

Usage:
    python3 plot.py [--data DIR] [--out DIR]

Output: reachability.png
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
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
    }
)


def load(path: pathlib.Path):
    with open(path) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=pathlib.Path, default=".",
                    help="directory containing reachability.json")
    ap.add_argument("--out", type=pathlib.Path, default=".",
                    help="output directory for the PNG")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    d = load(args.data / "reachability.json")
    snaps = d["snapshots"]
    ref = d["reference_orbit"]
    t_final = d["params"]["t_final"]

    fig, ax = plt.subplots(figsize=(7.4, 6.8))

    # Reference (ballistic) orbit: position is state components x2, x3.
    ax.plot(ref["x2"], ref["x3"], color="0.55", lw=1.0, zorder=1,
            label="ballistic orbit")
    ax.scatter([0], [0], marker="o", s=120, color="#f2b134", edgecolor="0.3",
               zorder=5, label="Sun")

    norm = Normalize(0.0, 1.0)
    cmap = plt.get_cmap("viridis")
    for snap in snaps:
        color = cmap(norm(snap["t"] / t_final))
        for leaf in snap["leaves"]:
            # NON-FILLED: outline only.
            ax.plot(leaf["x"], leaf["y"], color=color, lw=0.9, alpha=0.9, zorder=3)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("$t\\,/\\,T$")

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal")
    ax.legend(loc="upper left")
    ax.set_title("Low-thrust reachable set over one orbit")

    fig.tight_layout()
    out_path = args.out / "reachability.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
