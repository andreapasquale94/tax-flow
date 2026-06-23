#!/usr/bin/env python3
"""examples/transfer_dispersion/plot.py

Dispersion sets of the low-thrust Earth -> NEA transfer. For each thrust level
(columns) and each uncertainty case (rows: initial / thrust / both) the DA box
is propagated through the nominal transfer and sampled at snapshots; this plots
the convex hull of each dispersion set RELATIVE to the nominal trajectory (in
km), coloured by t/T (final set in red) so the growth and shape are visible.

Usage:
    python3 plot.py [files ...] [--out transfer_dispersion.png]
"""
import argparse
import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.spatial import ConvexHull

AU_KM = 1.495978707e8
CASES = ["initial", "thrust", "both"]
CASE_LABEL = {"initial": "initial dispersion\n(±1000 km, ±1 m/s)",
              "thrust": "thrust dispersion\n(±2%, ±5°)",
              "both": "both combined"}
DEFAULT = [pathlib.Path(f"transfer_dispersion_{l}.json") for l in ("low", "med", "high")]

plt.rcParams.update({"figure.facecolor": "white", "font.size": 10,
                     "xtick.direction": "in", "ytick.direction": "in"})


def load(p):
    with open(p) as f:
        return json.load(f)


def hull_xy(snap):
    """Dispersion-set hull for one snapshot, in the nominal radial-transverse
    (RTN) frame (km): x = transverse (along-track), y = radial."""
    cx, cy = snap["c"]
    cn = np.hypot(cx, cy)
    rh = np.array([cx, cy]) / cn          # radial unit
    th = np.array([-cy, cx]) / cn         # transverse unit (perp, prograde)
    dx = (np.asarray(snap["x"]) - cx) * AU_KM
    dy = (np.asarray(snap["y"]) - cy) * AU_KM
    dt = dx * th[0] + dy * th[1]          # transverse component
    dr = dx * rh[0] + dy * rh[1]          # radial component
    pts = np.column_stack([dt, dr])
    if np.ptp(dt) < 1e-9 and np.ptp(dr) < 1e-9:
        return None
    try:
        h = ConvexHull(pts)
        v = np.r_[h.vertices, h.vertices[0]]
        return pts[v, 0], pts[v, 1]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=pathlib.Path, default=DEFAULT)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("transfer_dispersion.png"))
    args = ap.parse_args()
    files = [f for f in args.files if f.exists()]
    data = [load(f) for f in files]
    levels = [d["params"]["level"] for d in data]
    n = len(data)

    cmap = LinearSegmentedColormap.from_list(
        "b", plt.get_cmap("Blues_r")(np.linspace(0.0, 0.8, 256)))
    norm = Normalize(0.0, 1.0)

    fig, axes = plt.subplots(3, n, figsize=(4.8 * n, 9.0), squeeze=False)
    for r, case in enumerate(CASES):
        for c, d in enumerate(data):
            ax = axes[r][c]
            cd = next(cc for cc in d["cases"] if cc["name"] == case)
            snaps = cd["snapshots"]
            tf = snaps[-1]["t"]
            sel = sorted(set(np.linspace(1, len(snaps) - 1, 6).astype(int)))
            for i in sel:
                sn = snaps[i]
                hx = hull_xy(sn)
                if hx is None:
                    continue
                last = i == len(snaps) - 1
                col = cmap(norm(sn["t"] / tf))
                ax.fill(hx[0], hx[1], facecolor=col, edgecolor="none",
                        alpha=0.45, zorder=2 + i)
                if last:
                    ax.plot(hx[0], hx[1], color="red", lw=1.8, zorder=40)
            ax.scatter([0], [0], c="k", s=10, zorder=5)
            ax.grid(alpha=0.3)  # axes auto-scaled (sets are ~100:1 along-track)
            ax.axhline(0, color="0.85", lw=0.6, zorder=0)
            ax.axvline(0, color="0.85", lw=0.6, zorder=0)
            if r == 0:
                ax.set_title(f"{levels[c]}", fontsize=12)
            if c == 0:
                ax.set_ylabel(CASE_LABEL[case] + "\n\nradial [km]", fontsize=9)
            if r == 2:
                ax.set_xlabel("transverse (along-track) [km]")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("$t\\,/\\,T$  (final set in red)")
    fig.suptitle("Low-thrust Earth → NEA transfer: dispersion sets (relative to nominal)",
                 fontsize=14, y=0.995)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
