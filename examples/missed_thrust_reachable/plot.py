#!/usr/bin/env python3
"""examples/missed_thrust_reachable/plot.py

Render the SET-VALUED missed-thrust reachable set written by the
`missed_thrust_reachable` example. Each snapshot stores a coverage grid: the
number of outage-parameter samples whose flow image lands in each cell. The
support of that grid (cells that are reachable at all) is the reachable-set
envelope; the count itself is a robustness heat — how much of the 5-D outage
box maps into each region.

Pass one or more scenario JSON files (mild / moderate / severe); each becomes
one column in the comparison figures.

Two figures are produced:

  * <out>            one panel per scenario: the reachable-set boundary at
                     every snapshot over the revolution, coloured by t/T
                     (final in red), on the nominal / ballistic orbits.
  * <out>_final.png  per scenario: the final reachable set as a coverage-
                     density heat with its outer boundary, zoomed in.

Usage:
    python3 plot.py [files ...] [--out missed_thrust_reachable.png]
"""

import argparse
import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm
from scipy.ndimage import gaussian_filter

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

DEFAULT_FILES = [
    pathlib.Path("missed_thrust_reachable_mild.json"),
    pathlib.Path("missed_thrust_reachable_moderate.json"),
    pathlib.Path("missed_thrust_reachable_severe.json"),
]


def load(path):
    with open(path) as f:
        return json.load(f)


def grid_axes(g):
    """Cell-centre coordinate arrays (x, y) for the stored coverage grid."""
    nx, ny = g["nx"], g["ny"]
    xs = g["xmin"] + (np.arange(nx) + 0.5) * (g["xmax"] - g["xmin"]) / nx
    ys = g["ymin"] + (np.arange(ny) + 0.5) * (g["ymax"] - g["ymin"]) / ny
    return xs, ys


def coverage(snap, g):
    """Return the (ny, nx) coverage-count grid as float."""
    return np.asarray(snap["cov"], dtype=float).reshape(g["ny"], g["nx"])


def outer_level(h, frac=0.995):
    """Density threshold enclosing `frac` of total coverage mass (HDR outer).

    Trims the sparsest cells (under-sampled folds / numerical specks) so the
    boundary follows the bulk reachable region, not isolated outliers.
    """
    flat = np.sort(h.ravel())[::-1]
    csum = np.cumsum(flat)
    if csum[-1] <= 0:
        return 0.5
    idx = min(np.searchsorted(csum, frac * csum[-1]), len(flat) - 1)
    return max(flat[idx], 0.5)


def support_mask(h, frac=0.995, smooth=1.2):
    """Smoothed reachable-support indicator for a clean outer contour."""
    occ = (h >= outer_level(h, frac)).astype(float)
    return gaussian_filter(occ, smooth) if smooth else occ


def support_bbox(h, xs, ys, frac=0.995, pad=0.12):
    """Padded square (xlim, ylim) of the reachable support, for auto-zoom."""
    iy, ix = np.where(h >= outer_level(h, frac))
    x0, x1 = xs[ix.min()], xs[ix.max()]
    y0, y1 = ys[iy.min()], ys[iy.max()]
    s = max(x1 - x0, y1 - y0) * (1.0 + 2 * pad)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    return (cx - s / 2, cx + s / 2), (cy - s / 2, cy + s / 2)


def draw_refs(ax, d, sun=True, label=False):
    ax.plot(d["ballistic_orbit"]["x"], d["ballistic_orbit"]["y"], color="0.55", lw=1.1,
            zorder=2, label="ballistic (no thrust)" if label else None)
    ax.plot(d["nominal_orbit"]["x"], d["nominal_orbit"]["y"], color="#1b9e77", lw=1.4,
            ls="--", zorder=2, label="nominal (full thrust)" if label else None)
    if sun:
        ax.scatter([0], [0], marker="o", s=110, color="#f2b134", edgecolor="0.3",
                   zorder=6, label="Sun" if label else None)


def shared_limits(datasets):
    xmin = min(d["grid"]["xmin"] for d in datasets)
    xmax = max(d["grid"]["xmax"] for d in datasets)
    ymin = min(d["grid"]["ymin"] for d in datasets)
    ymax = max(d["grid"]["ymax"] for d in datasets)
    return (xmin, xmax), (ymin, ymax)


def scenario_label(d):
    return d["params"].get("scenario", "").capitalize()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", type=pathlib.Path, default=DEFAULT_FILES)
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("missed_thrust_reachable.png"))
    args = ap.parse_args()

    files = [f for f in args.files if f.exists()]
    if not files:
        raise SystemExit("no input JSON files found (run ./missed_thrust_reachable first)")
    datasets = [load(f) for f in files]
    n = len(datasets)

    norm = Normalize(0.0, 1.0)
    cmap = LinearSegmentedColormap.from_list(
        "blues_r_hi", plt.get_cmap("Blues_r")(np.linspace(0.0, 0.8, 256)))
    xlim, ylim = shared_limits(datasets)

    # ======================================================================
    # Figure 1: reachable-set growth (outer boundary), one panel per scenario
    # ======================================================================
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n + 0.8, 6.6), squeeze=False)
    axes = axes[0]
    for k, (ax, d) in enumerate(zip(axes, datasets)):
        g = d["grid"]
        xs, ys = grid_axes(g)
        X, Y = np.meshgrid(xs, ys)
        snaps = d["snapshots"]
        t_final = snaps[-1]["t"]
        for i, snap in enumerate(snaps):
            h = coverage(snap, g)
            if not (h > 0).any():
                continue
            last = i == len(snaps) - 1
            mask = support_mask(h)
            if last:
                ax.contourf(X, Y, mask, levels=[0.5, 1.0], colors=["red"], alpha=0.30, zorder=4)
                ax.contour(X, Y, mask, levels=[0.5], colors=["red"], linewidths=1.7,
                           alpha=0.95, zorder=5)
            else:
                ax.contourf(X, Y, mask, levels=[0.5, 1.0],
                            colors=[cmap(norm(snap["t"] / t_final))], alpha=0.32, zorder=3)
        draw_refs(ax, d, label=(k == 0))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ [AU]")
        ax.set_ylabel("$y$ [AU]")
        ax.set_title(scenario_label(d), fontsize=12)
        if k == 0:
            ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes[-1], fraction=0.045, pad=0.02)
    cb.set_label("$t\\,/\\,T$")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    # ======================================================================
    # Figure 2: final reachable set as a coverage-density heat, zoomed
    # ======================================================================
    figf, axesf = plt.subplots(1, n, figsize=(5.4 * n, 5.8), squeeze=False)
    axesf = axesf[0]
    heat = LinearSegmentedColormap.from_list(
        "cov", plt.get_cmap("viridis")(np.linspace(0.12, 1.0, 256)))
    for k, (ax, d) in enumerate(zip(axesf, datasets)):
        g = d["grid"]
        xs, ys = grid_axes(g)
        X, Y = np.meshgrid(xs, ys)
        snap = d["snapshots"][-1]
        h = coverage(snap, g)
        hs = gaussian_filter(h, 1.3)
        vmax = hs.max()
        # Single density floor for BOTH the colour fill and the boundary, so they
        # coincide. Power-spaced levels (matching PowerNorm) grade the heavy
        # peak at the nominal end down through the sparse displaced tail.
        floor = 0.004 * vmax
        u = np.linspace(0.0, 1.0, 24)
        levels = np.unique(floor + (vmax - floor) * u ** (1.0 / 0.45))
        pcm = ax.contourf(X, Y, hs, levels=levels, cmap=heat,
                          norm=PowerNorm(0.45, vmin=floor, vmax=vmax), extend="max", zorder=1)
        ax.contour(X, Y, hs, levels=[floor], colors=["k"], linewidths=1.1, zorder=3)
        first = k == 0
        draw_refs(ax, d, sun=False, label=first)
        ax.plot(d["ballistic_orbit"]["x"][-1], d["ballistic_orbit"]["y"][-1], marker="o",
                ms=9, color="0.85", mec="0.2", zorder=7,
                label="ballistic end" if first else None)
        ax.plot(d["nominal_orbit"]["x"][-1], d["nominal_orbit"]["y"][-1], marker="*",
                ms=16, color="#1b9e77", mec="0.2", zorder=7,
                label="nominal end" if first else None)
        iy, ix = np.where(hs >= floor)
        bx0, bx1 = xs[ix.min()], xs[ix.max()]
        by0, by1 = ys[iy.min()], ys[iy.max()]
        side = max(bx1 - bx0, by1 - by0) * 1.24
        cx, cy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
        bxlim, bylim = (cx - side / 2, cx + side / 2), (cy - side / 2, cy + side / 2)
        ax.set_xlim(bxlim)
        ax.set_ylim(bylim)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ [AU]")
        ax.set_ylabel("$y$ [AU]")
        ax.set_title(scenario_label(d), fontsize=12)
        cb = figf.colorbar(pcm, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("coverage (outage scenarios)")
        cb.set_ticks([])
        if first:
            ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    figf.tight_layout()
    outf = args.out.with_name(args.out.stem + "_final.png")
    figf.savefig(outf, dpi=150)
    print(f"wrote {outf}")


if __name__ == "__main__":
    main()
