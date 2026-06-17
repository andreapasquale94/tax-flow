#!/usr/bin/env python3
"""
examples/three_body/plot.py

Render the three CR3BP example JSONs (cr3bp_taylor / cr3bp_ads /
cr3bp_loads) as a 3-panel comparison figure: an IC box at L1 pushed
forward by a single Taylor flow polynomial, by ADS, and by LOADS.

Output: three_body_box_evolution.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

HERE = Path.cwd()

METHODS = ("taylor", "ads", "loads")
TITLES  = {"taylor": "Taylor", "ads": "ADS", "loads": "LOADS"}


# ---- Framed sans-serif style (same as the two-body figures) ---------------
plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":            8.5,
    "axes.labelsize":       8.5,
    "axes.titlesize":       10.0,
    "axes.titlepad":        4.0,
    "axes.linewidth":       0.6,
    "xtick.major.size":     2.5,
    "ytick.major.size":     2.5,
    "xtick.major.width":    0.55,
    "ytick.major.width":    0.55,
    "xtick.labelsize":      7.0,
    "ytick.labelsize":      7.0,
    "xtick.direction":      "in",
    "ytick.direction":      "in",
    "legend.frameon":       False,
    "legend.fontsize":      7.0,
    "axes.grid":            True,
    "grid.linestyle":       ":",
    "grid.linewidth":       0.45,
    "grid.color":           "#9a9a9a",
    "grid.alpha":           0.6,
    "figure.dpi":           120,
    "savefig.dpi":          400,
    "savefig.bbox":         "tight",
})


def load(method: str) -> dict | None:
    p = HERE / f"cr3bp_{method}.json"
    return json.loads(p.read_text()) if p.exists() else None


def panel_xy_limits(*datasets: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    # We only show the Moon-side manifold, so Earth is intentionally
    # excluded from the limit calculation; the panel auto-fits the
    # L1 -> Moon range instead.
    xs, ys = [], []
    for d in datasets:
        cfg = d.get("config", {})
        for k in ("moon_x", "x_L1"):
            if k in cfg:
                xs.append(cfg[k])
        ys.append(0.0)
        for poly in d.get("polygons", []):
            if "leaves" in poly:
                for lf in poly["leaves"]:
                    xs.extend(lf["x"]); ys.extend(lf["y"])
            else:
                xs.extend(poly["x"]); ys.extend(poly["y"])
    if not xs:
        return (-1.5, 1.5), (-1.5, 1.5)
    xa, xb = min(xs), max(xs)
    ya, yb = min(ys), max(ys)
    px = 0.08 * max(xb - xa, 1e-6)
    py = 0.08 * max(yb - ya, 1e-6)
    return (xa - px, xb + px), (ya - py, yb + py)


def draw_panel(ax: plt.Axes, data: dict, *,
               title: str, xlim, ylim, cmap, norm) -> None:
    cfg      = data["config"]
    polygons = data["polygons"]

    # Polygon snapshots coloured by time.
    for snap in polygons:
        color = cmap(norm(snap["t"]))
        if "leaves" in snap:
            for lf in snap["leaves"]:
                ax.fill(lf["x"], lf["y"], color=color, alpha=0.65,
                        edgecolor="black", linewidth=0.25, zorder=2)
        else:
            ax.fill(snap["x"], snap["y"], color=color, alpha=0.65,
                    edgecolor="black", linewidth=0.25, zorder=2)

    # Moon + L1 only (Earth is outside the Moon-branch manifold region).
    moon_x = cfg["moon_x"]; x_L1 = cfg["x_L1"]
    ax.plot(moon_x, 0.0, marker="o", color="#aeaeae", markersize=6,
            markeredgecolor="black", markeredgewidth=0.4, zorder=10)
    ax.plot(x_L1, 0.0, marker="x", color="black", markersize=6,
            markeredgewidth=1.0, zorder=10)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal", "box")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(title, loc="left")
    ax.tick_params(length=2.5, width=0.55)


def main() -> None:
    loaded = [(m, load(m)) for m in METHODS]
    loaded = [(m, d) for m, d in loaded if d is not None]
    if not loaded:
        print("No cr3bp_*.json files found. Run the three CR3BP examples first.")
        return

    t_final = max(d["config"]["t_final"] for _, d in loaded)
    cmap    = plt.cm.viridis
    norm    = Normalize(vmin=0.0, vmax=t_final)
    xlim, ylim = panel_xy_limits(*(d for _, d in loaded))

    n_panels = len(loaded)
    fig      = plt.figure(figsize=(3.0 * n_panels, 3.4),
                          constrained_layout=True)
    gs       = fig.add_gridspec(2, n_panels, height_ratios=[1.0, 0.035])

    for col, (m, d) in enumerate(loaded):
        ax = fig.add_subplot(gs[0, col])
        draw_panel(ax, d, title=TITLES[m], xlim=xlim, ylim=ylim,
                   cmap=cmap, norm=norm)

    cax  = fig.add_subplot(gs[1, :])
    sm   = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(r"$t$", labelpad=1.5)
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(length=2.0, width=0.4, labelsize=6.5)

    out_path = HERE / "three_body_box_evolution.png"
    fig.savefig(out_path)
    print(f"wrote {out_path.name}")

    # Terminal summary.
    print()
    print(f"  {'method':<8}  {'elapsed':>9}   {'snaps':>5}   leaves (per snap)")
    print(f"  {'-'*8:<8}  {'-'*9:>9}   {'-'*5:>5}   {'-'*40}")
    for m, d in loaded:
        elapsed = d.get("timing", {}).get("elapsed_ms", float("nan")) / 1e3
        polys   = d["polygons"]
        if "leaves" in polys[0]:
            leaves = [len(s["leaves"]) for s in polys]
        else:
            leaves = [1] * len(polys)
        print(f"  {m:<8}  {elapsed:>7.2f} s   {len(polys):>5}   {leaves}")


if __name__ == "__main__":
    main()
