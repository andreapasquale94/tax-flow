#!/usr/bin/env python3
"""
examples/two_body/plot.py

Render the three example JSONs (taylor / ads / loads) into a 3-panel
figure showing the IC box pushed forward over one orbit. Panels are
fully framed with a dotted grid; snapshots are colour-coded by the
*true anomaly* ν along the reference orbit (degrees, 0° = periapsis,
360° = one full revolution).

Output: two_body_box_evolution.png
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

HERE   = Path.cwd()
ECC    = 0.5
MEAN_N = 1.0  # mean motion: sqrt(GM / a^3) = 1 in canonical units

METHODS = ("taylor", "ads", "loads")
TITLES  = {"taylor": "Taylor", "ads": "ADS", "loads": "LOADS"}


# ---- Framed Nature-ish rcParams -------------------------------------------
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
    "axes.spines.top":      True,
    "axes.spines.right":    True,
    "grid.linestyle":       ":",
    "grid.linewidth":       0.45,
    "grid.color":           "#9a9a9a",
    "grid.alpha":           0.6,
    "figure.dpi":           120,
    "savefig.dpi":          400,
    "savefig.bbox":         "tight",
})


# ---- Kepler equation -------------------------------------------------------
def true_anomaly(t: float, ecc: float = ECC, n: float = MEAN_N) -> float:
    """True anomaly ν(t) for an elliptic orbit, ν ∈ [0, 2π)."""
    mean_anom = n * t
    # Solve Kepler's equation M = E - e sin E for the eccentric anomaly E.
    e_anom = mean_anom
    for _ in range(40):
        delta = (e_anom - ecc * math.sin(e_anom) - mean_anom) / (1.0 - ecc * math.cos(e_anom))
        e_anom -= delta
        if abs(delta) < 1e-13:
            break
    nu = 2.0 * math.atan2(
        math.sqrt(1.0 + ecc) * math.sin(0.5 * e_anom),
        math.sqrt(1.0 - ecc) * math.cos(0.5 * e_anom),
    )
    return nu % (2.0 * math.pi)


# ---- IO --------------------------------------------------------------------
def load(method: str) -> dict | None:
    p = HERE / f"{method}.json"
    return json.loads(p.read_text()) if p.exists() else None


def panel_xy_limits(*datasets: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    xs, ys = [], []
    for d in datasets:
        ref = d.get("reference_orbit", {})
        xs.extend(ref.get("x0", []))
        ys.extend(ref.get("x1", []))
        for poly in d.get("polygons", []):
            if "leaves" in poly:
                for lf in poly["leaves"]:
                    xs.extend(lf["x"]); ys.extend(lf["y"])
            else:
                xs.extend(poly["x"]); ys.extend(poly["y"])
    if not xs:
        return (-2.0, 2.0), (-2.0, 2.0)
    xa, xb = min(xs), max(xs)
    ya, yb = min(ys), max(ys)
    px = 0.05 * max(xb - xa, 1e-12)
    py = 0.05 * max(yb - ya, 1e-12)
    return (xa - px, xb + px), (ya - py, yb + py)


def draw_panel(ax: plt.Axes, data: dict, *,
               title: str, xlim, ylim, cmap, norm) -> None:
    polygons = data["polygons"]

    # ---- Reference orbit ----
    ref = data.get("reference_orbit")
    if ref is not None:
        ax.plot(ref["x0"], ref["x1"], color="#7c7c7c",
                lw=0.55, alpha=0.85, zorder=1)

    # ---- Polygon snapshots colour-coded by true anomaly ----
    for snap in polygons:
        nu    = true_anomaly(snap["t"])
        color = cmap(norm(nu))
        if "leaves" in snap:
            for lf in snap["leaves"]:
                ax.fill(lf["x"], lf["y"], color=color, alpha=0.72,
                        edgecolor="black", linewidth=0.25, zorder=2)
        else:
            ax.fill(snap["x"], snap["y"], color=color, alpha=0.72,
                    edgecolor="black", linewidth=0.25, zorder=2)

    # ---- Primary at origin ----
    ax.plot(0.0, 0.0, marker="o", color="black", markersize=3.2, zorder=10)

    # ---- Axes, title, grid ----
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
        print("No JSON outputs found. Run two_body_taylor / two_body_ads / two_body_loads first.")
        return

    # Colour by true anomaly ν ∈ [0°, 360°). Use a cyclic colour map so
    # 0° and 360° map to the same colour — true anomaly is periodic on
    # a closed orbit.
    cmap = plt.cm.twilight
    norm = Normalize(vmin=0.0, vmax=2.0 * math.pi)

    xlim, ylim = panel_xy_limits(*(d for _, d in loaded))

    n_panels = len(loaded)
    fig      = plt.figure(figsize=(2.7 * n_panels, 3.3), constrained_layout=True)
    gs       = fig.add_gridspec(2, n_panels, height_ratios=[1.0, 0.035])

    for col, (m, d) in enumerate(loaded):
        ax = fig.add_subplot(gs[0, col])
        draw_panel(ax, d, title=TITLES[m], xlim=xlim, ylim=ylim,
                   cmap=cmap, norm=norm)

    # ---- Shared horizontal colour bar (true anomaly, in degrees) ----
    cax  = fig.add_subplot(gs[1, :])
    sm   = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    deg_ticks = [0, 0.5 * math.pi, math.pi, 1.5 * math.pi, 2.0 * math.pi]
    cbar.set_ticks(deg_ticks)
    cbar.set_ticklabels(["0°", "90°", "180°", "270°", "360°"])
    cbar.set_label(r"true anomaly  $\nu$", labelpad=2.0)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(length=2.0, width=0.45, labelsize=6.5)

    out_path = HERE / "two_body_box_evolution.png"
    fig.savefig(out_path)
    print(f"wrote {out_path.name}")

    # ---- Terminal summary ----
    print()
    print(f"  {'method':<8}  {'elapsed':>9}   {'snaps':>5}   leaves (per snap)")
    print(f"  {'-'*8:<8}  {'-'*9:>9}   {'-'*5:>5}   {'-'*32}")
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
