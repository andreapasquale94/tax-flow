#!/usr/bin/env python3
"""examples/transfer_dispersion/plot.py

Dispersion sets of the low-thrust Earth -> NEA transfer, shown ON the transfer
in the Sun-Earth ROTATING frame (Earth fixed at (1,0)). For each thrust level
(columns) and uncertainty case (rows: initial / thrust / both) the DA box is
propagated and sampled at snapshots; the convex hull of each dispersion set is
drawn at its actual position and scale along the nominal trajectory, coloured
by t/T (final set in red).

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


def rot(tabs, x, y):
    """Inertial -> Sun-Earth rotating frame (rotate by -tabs)."""
    c, s = np.cos(tabs), np.sin(tabs)
    return c * x + s * y, -s * x + c * y


def nea_state(nea, t):
    a, e, w, M0 = nea["a"], nea["e"], nea["w"], nea["M0"]
    n = a ** -1.5
    M = M0 + n * t
    E = M
    for _ in range(60):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    b = np.sqrt(1 - e * e)
    xo, yo = a * (np.cos(E) - e), a * b * np.sin(E)
    cw, sw = np.cos(w), np.sin(w)
    return cw * xo - sw * yo, sw * xo + cw * yo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=pathlib.Path, default=DEFAULT)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("transfer_dispersion.png"))
    args = ap.parse_args()
    data = [load(f) for f in args.files if f.exists()]
    levels = [d["params"]["level"] for d in data]
    n = len(data)

    cmap = LinearSegmentedColormap.from_list(
        "b", plt.get_cmap("Blues_r")(np.linspace(0.0, 0.8, 256)))
    norm = Normalize(0.0, 1.0)

    fig, axes = plt.subplots(3, n, figsize=(4.6 * n, 13.0), squeeze=False)
    for r, case in enumerate(CASES):
        for c, d in enumerate(data):
            ax = axes[r][c]
            td, T, nea = d["params"]["td"], d["params"]["T"], d["nea"]
            # nominal transfer (rotating)
            nom = d["nominal"]
            nt = np.asarray(nom["t"])
            nx, ny = rot(td + nt, np.asarray(nom["x"]), np.asarray(nom["y"]))
            ax.plot(nx, ny, color="0.45", lw=1.3, zorder=2)
            # NEA arc (+-90 d) and Earth orbit, rotating
            ts = np.linspace(td - 90 * 2 * np.pi / 365.25, td + T + 90 * 2 * np.pi / 365.25, 400)
            nax, nay = [], []
            for t in ts:
                xx, yy = nea_state(nea, t)
                a, b = rot(t, xx, yy)
                nax.append(a); nay.append(b)
            ax.plot(nax, nay, color="0.78", lw=1.1, zorder=1)
            ax.scatter(0, 0, c="#f2b134", s=130, ec="0.3", zorder=6)        # Sun
            ax.plot(1, 0, marker="o", ms=7, color="k", zorder=7)            # Earth (departure)
            # dispersion sets
            cd = next(cc for cc in d["cases"] if cc["name"] == case)
            snaps = cd["snapshots"]
            tf = snaps[-1]["t"]
            for i, sn in enumerate(snaps):
                xr, yr = rot(td + sn["t"], np.asarray(sn["x"]), np.asarray(sn["y"]))
                pts = np.column_stack([xr, yr])
                if np.ptp(xr) < 1e-12 and np.ptp(yr) < 1e-12:
                    continue
                try:
                    h = ConvexHull(pts); v = np.r_[h.vertices, h.vertices[0]]
                except Exception:
                    continue
                last = i == len(snaps) - 1
                col = cmap(norm(sn["t"] / tf))
                ax.fill(pts[v, 0], pts[v, 1], facecolor=col, edgecolor="none",
                        alpha=0.55, zorder=3 + i)
                if last:
                    ax.plot(pts[v, 0], pts[v, 1], color="red", lw=1.6, zorder=40)
            cax, cay = rot(td + tf, snaps[-1]["c"][0], snaps[-1]["c"][1])
            ax.plot(cax, cay, marker="D", ms=7, color="k", zorder=8)        # arrival
            ax.set_aspect("equal"); ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(levels[c], fontsize=12)
            if c == 0:
                ax.set_ylabel(CASE_LABEL[case] + "\n\n$y$ [AU]", fontsize=9)
            if r == 2:
                ax.set_xlabel("$x$ [AU]")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.022, pad=0.02)
    cb.set_label("$t\\,/\\,T$  (final set in red)")
    fig.suptitle("Low-thrust Earth → NEA: dispersion sets on the transfer (Sun–Earth rotating frame)",
                 fontsize=13, y=0.995)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
