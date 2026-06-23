#!/usr/bin/env python3
"""examples/missed_thrust_onoff/plot.py

Render the on/off missed-thrust dispersion set written by the
`missed_thrust_onoff` example. The Monte-Carlo cloud at each 4-day snapshot is
stored as a 2-D histogram on a shared grid; from it we draw highest-density
confidence bands (the smallest region containing a given fraction of samples).

Pass one or more scenario JSON files (reliable / intermittent / unreliable);
each becomes one column in the comparison figures.

Three figures are produced:

  * <out>            one panel per scenario: the 3-sigma reachable band at every
                     4-day snapshot over the revolution, coloured by t/T (final
                     in red), on the ballistic orbit.
  * <out>_zoom.png   the final dispersion set with filled 1/2/3-sigma bands.
  * <out>_time.png   per scenario: the ON/OFF duty distribution (top) and the
                     radial dispersion envelope (bottom) as a function of time.

Usage:
    python3 plot.py [files ...] [--out missed_thrust_onoff.png]
"""

import argparse
import json
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
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

BAND_COLS = ["#fde0c5", "#f3a16e", "#d6491f"]  # 3 sigma, 2 sigma, 1 sigma (light -> dark)
# 2-D Gaussian-equivalent coverage masses for 1 / 2 / 3 sigma.
SIGMA_FRACS = [0.3935, 0.8647, 0.9889]
ONOFF_COLS = ["#c0392b", "#1b9e77"]  # OFF (red), ON (green)
ONOFF_LABELS = ["thruster off", "thruster on"]
DEFAULT_FILES = [
    pathlib.Path("missed_thrust_onoff_reliable.json"),
    pathlib.Path("missed_thrust_onoff_intermittent.json"),
    pathlib.Path("missed_thrust_onoff_unreliable.json"),
]


def load(path):
    with open(path) as f:
        return json.load(f)


def grid_axes(g):
    """Cell-centre coordinate arrays (x, y) for the stored histogram grid."""
    nx, ny = g["nx"], g["ny"]
    xs = g["xmin"] + (np.arange(nx) + 0.5) * (g["xmax"] - g["xmin"]) / nx
    ys = g["ymin"] + (np.arange(ny) + 0.5) * (g["ymax"] - g["ymin"]) / ny
    return xs, ys


def hist2d(snap, g, smooth=1.1):
    """Return the (ny, nx) histogram, lightly smoothed for clean contours."""
    h = np.asarray(snap["hist"], dtype=float).reshape(g["ny"], g["nx"])
    if smooth:
        h = gaussian_filter(h, smooth)
    return h


def hdr_levels(h, fracs):
    """Density thresholds enclosing the given fractions of total mass (HDR)."""
    flat = np.sort(h.ravel())[::-1]
    csum = np.cumsum(flat)
    total = csum[-1]
    out = []
    for f in fracs:
        idx = min(np.searchsorted(csum, f * total), len(flat) - 1)
        out.append(flat[idx])
    return out


def strictly_increasing(levels):
    """Nudge duplicate/decreasing contour levels so contourf accepts them."""
    out = list(levels)
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] * (1.0 + 1e-6) + 1e-12
    return out


def band_bbox(h, xs, ys, level, pad=0.45):
    """Padded square (xlim, ylim) of the region {h >= level}, for auto-zoom."""
    iy, ix = np.where(h >= level)
    x0, x1 = xs[ix.min()], xs[ix.max()]
    y0, y1 = ys[iy.min()], ys[iy.max()]
    s = max(x1 - x0, y1 - y0) * (1.0 + 2 * pad)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    return (cx - s / 2, cx + s / 2), (cy - s / 2, cy + s / 2)


def draw_refs(ax, d, sun=True, label=False):
    ax.plot(d["ballistic_orbit"]["x"], d["ballistic_orbit"]["y"], color="0.55", lw=1.1,
            zorder=2, label="ballistic (always off)" if label else None)
    ax.plot(d["nominal_orbit"]["x"], d["nominal_orbit"]["y"], color="#1b9e77", lw=1.4,
            ls="--", zorder=2, label="nominal (always on)" if label else None)
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


def time_series(d):
    """Return (tT0, level_dist0, r1, r2, r3) padded with the t = 0 state."""
    snaps = d["snapshots"]
    t_final = snaps[-1]["t"]
    tT0 = np.concatenate([[0.0], [s["t"] / t_final for s in snaps]])
    n_lvl = d["params"]["n_levels"]
    ld = np.array([s["level_dist"] for s in snaps])
    ld0 = np.vstack([np.eye(n_lvl)[n_lvl - 1], ld])  # start ON
    r1 = np.concatenate([[0.0], [s["r1s"] for s in snaps]])
    r2 = np.concatenate([[0.0], [s["r2s"] for s in snaps]])
    r3 = np.concatenate([[0.0], [s["r3s"] for s in snaps]])
    return tT0, ld0, r1, r2, r3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="*", type=pathlib.Path, default=DEFAULT_FILES)
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("missed_thrust_onoff.png"))
    args = ap.parse_args()

    files = [f for f in args.files if f.exists()]
    if not files:
        raise SystemExit("no input JSON files found (run ./missed_thrust_onoff first)")
    datasets = [load(f) for f in files]
    n = len(datasets)

    norm = Normalize(0.0, 1.0)
    cmap = LinearSegmentedColormap.from_list(
        "blues_r_hi", plt.get_cmap("Blues_r")(np.linspace(0.0, 0.8, 256)))
    band_handles = [
        Patch(facecolor=BAND_COLS[2], label="1$\\sigma$"),
        Patch(facecolor=BAND_COLS[1], label="2$\\sigma$"),
        Patch(facecolor=BAND_COLS[0], label="3$\\sigma$"),
    ]
    xlim, ylim = shared_limits(datasets)

    # ======================================================================
    # Figure 1: reachable-set growth (3-sigma envelope), one panel per scenario
    # ======================================================================
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n + 0.8, 6.6), squeeze=False)
    axes = axes[0]
    for k, (ax, d) in enumerate(zip(axes, datasets)):
        g = d["grid"]
        xs, ys = grid_axes(g)
        X, Y = np.meshgrid(xs, ys)
        # A histogram contour cannot resolve a cloud smaller than ~1 grid cell:
        # below this the smoothed HDR contour saturates at a fixed "floor" size
        # (~3 cells) and the early sets look spuriously large. Where the true
        # 3-sigma radius is sub-resolution we draw it as a circle at the cloud
        # mean instead, so the set shows its real (tiny) size and growth.
        cell = max((g["xmax"] - g["xmin"]) / g["nx"], (g["ymax"] - g["ymin"]) / g["ny"])
        res_floor = 1.5 * cell
        snaps = d["snapshots"]
        t_final = snaps[-1]["t"]
        theta = np.linspace(0, 2 * np.pi, 80)
        for i, snap in enumerate(snaps):
            last = i == len(snaps) - 1
            col = "red" if last else cmap(norm(snap["t"] / t_final))
            lw = 1.6 if last else 1.0
            alpha = 0.95 if last else 0.85
            zo = 4 if last else 3
            if snap["r3s"] < res_floor:                # sub-resolution: true circle
                cx, cy = snap["mean"]
                ax.plot(cx + snap["r3s"] * np.cos(theta), cy + snap["r3s"] * np.sin(theta),
                        color=col, lw=lw, alpha=alpha, zorder=zo)
                continue
            h = hist2d(snap, g)
            lvl = hdr_levels(h, [SIGMA_FRACS[2]])[0]   # 3-sigma envelope
            ax.contour(X, Y, h, levels=strictly_increasing([lvl, h.max()]),
                       colors=[col], linewidths=lw, alpha=alpha, zorder=zo)
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
    # Figure 2: zoomed final dispersion set with 1/2/3-sigma bands per scenario
    # ======================================================================
    figz, axesz = plt.subplots(1, n, figsize=(5.2 * n, 5.6), squeeze=False)
    axesz = axesz[0]
    for k, (ax, d) in enumerate(zip(axesz, datasets)):
        g = d["grid"]
        xs, ys = grid_axes(g)
        X, Y = np.meshgrid(xs, ys)
        snap = d["snapshots"][-1]
        h = hist2d(snap, g)
        l1, l2, l3 = hdr_levels(h, SIGMA_FRACS)
        ax.contourf(X, Y, h, levels=strictly_increasing([l3, l2, l1, h.max()]),
                    colors=BAND_COLS, alpha=0.95, zorder=1)
        first = k == 0
        draw_refs(ax, d, sun=False, label=first)
        ax.plot(d["ballistic_orbit"]["x"][-1], d["ballistic_orbit"]["y"][-1], marker="o",
                ms=9, color="0.4", mec="0.2", zorder=7,
                label="ballistic end" if first else None)
        ax.plot(d["nominal_orbit"]["x"][-1], d["nominal_orbit"]["y"][-1], marker="*",
                ms=16, color="#1b9e77", mec="0.2", zorder=7,
                label="nominal end" if first else None)
        ax.scatter(snap["mean"][0], snap["mean"][1], marker="x", s=70, color="k",
                   lw=1.8, zorder=7, label="mean" if first else None)
        bxlim, bylim = band_bbox(h, xs, ys, l3, pad=0.45)
        ax.set_xlim(bxlim)
        ax.set_ylim(bylim)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ [AU]")
        ax.set_ylabel("$y$ [AU]")
        ax.set_title(scenario_label(d), fontsize=12)
        if first:
            hh, ll = ax.get_legend_handles_labels()
            ax.legend(band_handles + hh, [p.get_label() for p in band_handles] + ll,
                      loc="best", fontsize=8, framealpha=0.9)
    figz.tight_layout()
    outz = args.out.with_name(args.out.stem + "_zoom.png")
    figz.savefig(outz, dpi=150)
    print(f"wrote {outz}")

    # ======================================================================
    # Figure 3: missed thrust as a function of time, one column per scenario
    #   row 0: ON/OFF duty distribution (Markov marginal, stacked);
    #   row 1: radial dispersion envelope (1 / 2 / 3 sigma).
    # ======================================================================
    rmax = max(max(s["r3s"] for s in d["snapshots"]) for d in datasets) * 1.05

    fig2, axes2 = plt.subplots(2, n, figsize=(5.0 * n, 7.4), squeeze=False, sharex=True)
    for j, d in enumerate(datasets):
        tT0, ld0, r1, r2, r3 = time_series(d)
        axa, axb = axes2[0][j], axes2[1][j]

        axa.stackplot(tT0, ld0[:, 0], ld0[:, 1], colors=ONOFF_COLS, labels=ONOFF_LABELS,
                      alpha=0.95)
        axa.set_ylim(0, 1)
        axa.margins(x=0)
        axa.set_title(scenario_label(d), fontsize=11)
        if j == 0:
            axa.set_ylabel("fraction of trajectories")

        axb.fill_between(tT0, 0, r3, color=BAND_COLS[0], label="3$\\sigma$ envelope")
        axb.fill_between(tT0, 0, r2, color=BAND_COLS[1], label="2$\\sigma$ envelope")
        axb.fill_between(tT0, 0, r1, color=BAND_COLS[2], label="1$\\sigma$ envelope")
        axb.plot(tT0, r3, color="#7a2710", lw=1.0)
        axb.set_ylim(0, rmax)
        axb.margins(x=0)
        axb.set_xlabel("$t\\,/\\,T$")
        if j == 0:
            axb.set_ylabel("dispersion radius |$r-\\bar r$| [AU]")

    axes2[0][-1].legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    axes2[1][-1].legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    fig2.tight_layout()
    out2 = args.out.with_name(args.out.stem + "_time.png")
    fig2.savefig(out2, dpi=150)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
