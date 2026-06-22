#!/usr/bin/env python3
"""examples/reachability/plot.py

Render the low-thrust reachable set from one or more JSON files (written by the
`reachability` example). Each file produces one panel in a side-by-side figure.
Snapshot envelopes are drawn as NON-FILLED outlines, coloured by t / T, overlaid
on the ballistic reference orbit.  A single shared Blues colour bar encodes t/T.

Usage:
    python3 plot.py [files ...] [--out PATH]

    files   one or more reachability JSON paths  (default: reachability.json)
    --out   output PNG path                       (default: reachability.png)
"""

import argparse
import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize

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
    ap.add_argument(
        "files",
        nargs="*",
        type=pathlib.Path,
        default=[pathlib.Path("reachability.json")],
        help="one or more reachability JSON files",
    )
    ap.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path("reachability.png"),
        help="output PNG path",
    )
    args = ap.parse_args()

    datasets = [load(f) for f in args.files]

    # Shared colour mapping across all panels. Blues_r, sampled over [0.0, 0.8]
    # so the latest snapshots stay visible rather than fading into white.
    norm = Normalize(0.0, 1.0)
    cmap = LinearSegmentedColormap.from_list(
        "blues_r_hi", plt.get_cmap("Blues_r")(np.linspace(0.0, 0.8, 256))
    )

    # Compute shared x/y limits from all reference orbits + all envelope points.
    all_x, all_y = [], []
    for d in datasets:
        ref = d["reference_orbit"]
        all_x.extend(ref["x2"])
        all_y.extend(ref["x3"])
        for snap in d["snapshots"]:
            for leaf in snap["leaves"]:
                all_x.extend(leaf["x"])
                all_y.extend(leaf["y"])
    # Add Sun origin.
    all_x.append(0.0)
    all_y.append(0.0)

    margin = 0.05
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    xlim = (min(all_x) - margin * x_range, max(all_x) + margin * x_range)
    ylim = (min(all_y) - margin * y_range, max(all_y) + margin * y_range)

    n = len(datasets)
    fig_w = 7.0 * n + 0.8  # extra width for colour bar
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 6.8), squeeze=False)
    axes = axes[0]  # list of Axes

    for ax, d in zip(axes, datasets):
        params = d["params"]
        snaps = d["snapshots"]
        ref = d["reference_orbit"]
        t_final = params["t_final"]

        # Ballistic reference orbit.
        ax.plot(ref["x2"], ref["x3"], color="0.55", lw=1.0, zorder=1,
                label="ballistic orbit")
        ax.scatter([0], [0], marker="o", s=120, color="#f2b134", edgecolor="0.3",
                   zorder=5, label="Sun")

        # Envelope outlines coloured by t/T; final snapshot highlighted in red.
        for idx, snap in enumerate(snaps):
            is_last = idx == len(snaps) - 1
            if is_last:
                color = "red"
                lw = 1.5
                zo = 4
            else:
                color = cmap(norm(snap["t"] / t_final))
                lw = 0.9
                zo = 3
            for leaf in snap["leaves"]:
                # NON-FILLED: outline only.
                ax.plot(leaf["x"], leaf["y"], color=color, lw=lw, alpha=0.9, zorder=zo)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

        # Title from params.case (already JSON-quoted in file; strip quotes).
        case_label = params.get("case", "")
        if case_label.startswith('"') and case_label.endswith('"'):
            case_label = case_label[1:-1]
        ax.set_title(case_label)

    # Single shared colour bar on the right of the last panel.
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes[-1], fraction=0.04, pad=0.02)
    cbar.set_label("$t\\,/\\,T$")

    fig.tight_layout()
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
