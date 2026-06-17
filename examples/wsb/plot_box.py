#!/usr/bin/env python3
"""
examples/wsb/plot_box.py

Render the two WSB box-propagation JSONs (wsb_taylor / wsb_ads) as
a 2-panel comparison figure: 1 km / 1 m/s IC box around the tangent
WSB IC, pushed forward through the Sun-Earth CR3BP up to Moon-orbit
interception (76 days) with snapshots every 5 days.

Output: wsb_box_evolution.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle

HERE = Path.cwd()

METHODS = ("taylor", "ads")
TITLES  = {"taylor": "Taylor", "ads": "ADS"}


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
    p = HERE / f"wsb_{method}.json"
    return json.loads(p.read_text()) if p.exists() else None


def panel_xy_limits(*datasets: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """Union x/y over reference orbits + polygons. All in km, Earth-centred."""
    xs, ys = [], []
    for d in datasets:
        cfg = d["config"]
        AU_km = cfg["AU_km"]
        ex    = cfg["earth_x"]
        ref   = d.get("reference_orbit", {})
        # Reference orbit (synodic), shifted Earth-centred and converted to km.
        if ref:
            xs.extend([(v - ex) * AU_km for v in ref.get("x0", [])])
            ys.extend([v * AU_km for v in ref.get("x1", [])])
        for poly in d.get("polygons", []):
            if "leaves" in poly:
                for lf in poly["leaves"]:
                    xs.extend([(v - ex) * AU_km for v in lf["x"]])
                    ys.extend([v * AU_km for v in lf["y"]])
            else:
                xs.extend([(v - ex) * AU_km for v in poly["x"]])
                ys.extend([v * AU_km for v in poly["y"]])
    if not xs:
        return (-1e6, 1e6), (-1e6, 1e6)
    xa, xb = min(xs), max(xs)
    ya, yb = min(ys), max(ys)
    px = 0.06 * max(xb - xa, 1.0)
    py = 0.06 * max(yb - ya, 1.0)
    return (xa - px, xb + px), (ya - py, yb + py)


def draw_panel(ax, data: dict, *, title: str, xlim, ylim, cmap, norm) -> None:
    cfg   = data["config"]
    AU_km = cfg["AU_km"]
    ex    = cfg["earth_x"]

    # Reference orbit (faint grey).
    ref = data.get("reference_orbit", {})
    if ref:
        rx = [(v - ex) * AU_km for v in ref["x0"]]
        ry = [v * AU_km        for v in ref["x1"]]
        ax.plot(rx, ry, color="#9a9a9a", lw=0.45, alpha=0.85, zorder=1)

    # Polygon snapshots.
    for snap in data["polygons"]:
        color = cmap(norm(snap["t_days"]))
        if "leaves" in snap:
            for lf in snap["leaves"]:
                ax.fill([(v - ex) * AU_km for v in lf["x"]],
                        [v * AU_km        for v in lf["y"]],
                        color=color, alpha=0.65,
                        edgecolor="black", linewidth=0.25, zorder=2)
        else:
            ax.fill([(v - ex) * AU_km for v in snap["x"]],
                    [v * AU_km        for v in snap["y"]],
                    color=color, alpha=0.65,
                    edgecolor="black", linewidth=0.25, zorder=2)

    # Earth + L1 + L2 + Moon orbit reference.
    moon_r = cfg["moon_orbit_AU"] * AU_km
    L1_dx  = (cfg["L1_x"] - ex) * AU_km
    L2_dx  = (cfg["L2_x"] - ex) * AU_km
    hill_r = cfg["earth_hill_radius"] * AU_km

    ax.add_patch(Circle((0, 0), hill_r, fill=False, edgecolor="#1f77b4",
                        linewidth=0.45, linestyle="--", alpha=0.45, zorder=1))
    ax.add_patch(Circle((0, 0), moon_r, fill=False, edgecolor="#aeaeae",
                        linewidth=0.45, linestyle=":", alpha=0.65, zorder=1))
    ax.plot(0, 0, marker="o", color="#1f77b4", markersize=6,
            markeredgecolor="black", markeredgewidth=0.4, zorder=10)
    ax.plot(L1_dx, 0, marker="x", color="black", markersize=4,
            markeredgewidth=0.8, zorder=10)
    ax.plot(L2_dx, 0, marker="x", color="black", markersize=4,
            markeredgewidth=0.8, zorder=10)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Earth-centred $x$  (km)")
    ax.set_ylabel("Earth-centred $y$  (km)")
    ax.set_title(title, loc="left")


def main() -> None:
    loaded = [(m, load(m)) for m in METHODS]
    loaded = [(m, d) for m, d in loaded if d is not None]
    if not loaded:
        print("No wsb_*.json files found. Run wsb_taylor / wsb_ads / wsb_loads first.")
        return

    t_final = max(d["config"]["t_final_days"] for _, d in loaded)
    cmap    = plt.cm.viridis
    norm    = Normalize(vmin=0.0, vmax=t_final)
    xlim, ylim = panel_xy_limits(*(d for _, d in loaded))

    n = len(loaded)
    fig = plt.figure(figsize=(3.1 * n, 3.6), constrained_layout=True)
    gs  = fig.add_gridspec(2, n, height_ratios=[1.0, 0.035])

    for col, (m, d) in enumerate(loaded):
        ax = fig.add_subplot(gs[0, col])
        draw_panel(ax, d, title=TITLES[m], xlim=xlim, ylim=ylim,
                   cmap=cmap, norm=norm)

    cax  = fig.add_subplot(gs[1, :])
    sm   = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("days since launch")
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(length=2.0, width=0.4, labelsize=6.5)

    out_path = HERE / "wsb_box_evolution.png"
    fig.savefig(out_path)
    print(f"wrote {out_path.name}")

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
