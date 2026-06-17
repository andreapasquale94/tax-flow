#!/usr/bin/env python3
"""
examples/wsb/plot_search.py

Render the wsb_search.json output as a 4-panel figure:

  (a) Wide Earth-centred view of the best trajectory, with Sun-Earth
      L1/L2 marks and a Moon-orbit reference circle.
  (b) Earth-vicinity zoom of the best trajectory.
  (c) r(t) of the best trajectory on a log axis.
  (d) Score heatmap over the (r_a, omega) sweep grid; the best cell
      is marked with a white star.

Inputs:  wsb_search.json (written by ./wsb_search)
Output:  wsb_search.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

HERE = Path.cwd()


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


def main() -> None:
    p = HERE / "wsb_search.json"
    if not p.exists():
        print("wsb_search.json not found. Run wsb_search first.")
        return
    data = json.loads(p.read_text())

    cfg          = data["config"]
    best         = data["best"]
    traj         = data["trajectory"]
    sweep_coarse = data["sweep_coarse"]
    sweep_fine   = data.get("sweep_fine", [])
    sweep_ultra  = data.get("sweep_ultra", [])
    AU_km        = cfg["AU_km"]
    day_u        = cfg["time_unit_days"]

    x = np.asarray(traj["x_earth"]) * AU_km
    y = np.asarray(traj["y_earth"]) * AU_km
    t = np.asarray(traj["t"]) * day_u
    r = np.sqrt(x * x + y * y)

    L1_dx  = (cfg["L1_x"] - cfg["earth_x"]) * AU_km
    L2_dx  = (cfg["L2_x"] - cfg["earth_x"]) * AU_km
    moon_r = cfg["moon_orbit_AU"] * AU_km
    hill_r = cfg["earth_hill_radius"] * AU_km

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0.0, vmax=t.max())

    fig = plt.figure(figsize=(14.5, 9.0), constrained_layout=True)
    gs  = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.0])

    # ---- (a) Wide view ----
    ax = fig.add_subplot(gs[0, 0:2])
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], color=cmap(norm(t[i])),
                lw=0.55, zorder=2)
    ax.add_patch(Circle((0, 0), hill_r, fill=False, edgecolor="#1f77b4",
                        linewidth=0.6, linestyle="--", alpha=0.6, zorder=1))
    ax.add_patch(Circle((0, 0), moon_r, fill=False, edgecolor="#aeaeae",
                        linewidth=0.6, linestyle=":", alpha=0.7, zorder=1))
    ax.plot(0, 0, marker="o", color="#1f77b4", markersize=8,
            markeredgecolor="black", markeredgewidth=0.4, zorder=10)
    ax.plot(L1_dx, 0, marker="x", color="black", markersize=5,
            markeredgewidth=1.0, zorder=10)
    ax.plot(L2_dx, 0, marker="x", color="black", markersize=5,
            markeredgewidth=1.0, zorder=10)
    ax.text(L1_dx, 1.2e5, r"$L_1$", ha="center", va="bottom", fontsize=8.0)
    ax.text(L2_dx, 1.2e5, r"$L_2$", ha="center", va="bottom", fontsize=8.0)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Earth-centred $x$  (km)")
    ax.set_ylabel("Earth-centred $y$  (km)")
    ax.set_title("(a) Best trajectory, wide view", loc="left")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("days")

    # ---- (b) Earth-vicinity zoom ----
    ax = fig.add_subplot(gs[0, 2])
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], color=cmap(norm(t[i])),
                lw=0.55, zorder=2)
    ax.add_patch(Circle((0, 0), moon_r, fill=False, edgecolor="#aeaeae",
                        linewidth=0.7, linestyle=":", alpha=0.85, zorder=1))
    ax.plot(0, 0, marker="o", color="#1f77b4", markersize=8,
            markeredgecolor="black", markeredgewidth=0.4, zorder=10)
    zoom = 3.0 * moon_r
    ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("Earth-centred $x$  (km)")
    ax.set_ylabel("Earth-centred $y$  (km)")
    ax.set_title("(b) Earth-vicinity zoom (~3 Moon orbits across)", loc="left")
    ax.text(moon_r * 1.05, 0, "Moon orbit", ha="left", va="center",
            fontsize=6.5, color="#666666")

    # ---- (c) r(t) ----
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(t, r, color="#1f77b4", lw=0.8, label=r"$r(t)$")
    ax.axhline(hill_r, color="#7d7d7d", linestyle="--", lw=0.6, label="Earth Hill")
    ax.axhline(moon_r, color="#aeaeae", linestyle=":",  lw=0.6, label="Moon orbit")
    ax.axhline(best.get("r_apogee_km", 0.0), color="#d62728",
               linestyle="-.", lw=0.6, label="observed $r_a$")
    ax.axvline(best["t_arrival_days"], color="black", linestyle="-",
               lw=0.4, alpha=0.4)
    ax.set_yscale("log")
    ax.set_xlabel("days since launch")
    ax.set_ylabel("distance to Earth (km)")
    ax.set_title("(c) r(t)", loc="left")
    ax.legend(loc="lower right", fontsize=6.5)

    # ---- (d/e/f) Sweep heatmaps ----
    def _draw_heatmap(ax, sweep_data, *, vmax, title, mark_best=True):
        if not sweep_data:
            ax.set_visible(False)
            return None
        r_as_h = sorted(set(s["r_a_km"] for s in sweep_data))
        ws_h   = sorted(set(round(s["omega_deg"], 6) for s in sweep_data))
        Z = np.full((len(r_as_h), len(ws_h)), np.nan)
        for s in sweep_data:
            if s["reached_moon"] and s["prograde"]:
                Z[r_as_h.index(s["r_a_km"]),
                  ws_h.index(round(s["omega_deg"], 6))] = s["score"]
        im = ax.pcolormesh(ws_h, r_as_h, Z, cmap="cividis_r",
                           shading="nearest", vmin=0.0, vmax=vmax)
        if mark_best:
            ax.plot(best["omega_deg"], best["r_a_km"], marker="*",
                    color="white", markeredgecolor="black",
                    markeredgewidth=0.5, markersize=12, zorder=10)
        ax.set_xlabel(r"$\omega$  (deg)")
        ax.set_ylabel(r"$r_a$  (km)")
        ax.set_title(title, loc="left", fontsize=9.5)
        return im

    ax_d = fig.add_subplot(gs[1, 0:2])
    im_d = _draw_heatmap(ax_d, sweep_coarse, vmax=1.0,
                         title="(d) coarse sweep — |$v_r$|/|$v$|")
    if im_d is not None:
        cbar = fig.colorbar(im_d, ax=ax_d, fraction=0.04, pad=0.02)
        cbar.set_label("score")

    ax_e = fig.add_subplot(gs[1, 2])
    im_e = _draw_heatmap(ax_e, sweep_fine, vmax=0.30,
                         title="(e) fine sweep")
    if im_e is not None:
        cbar = fig.colorbar(im_e, ax=ax_e, fraction=0.04, pad=0.02)

    ax_f = fig.add_subplot(gs[1, 3])
    im_f = _draw_heatmap(ax_f, sweep_ultra, vmax=0.05,
                         title="(f) ultra sweep")
    if im_f is not None:
        cbar = fig.colorbar(im_f, ax=ax_f, fraction=0.04, pad=0.02)

    fp_deg = np.degrees(np.arctan2(best["vr_arrival_kms"],
                                   best["vt_arrival_kms"]))
    fig.suptitle(
        rf"WSB search — best $r_a$ = {best['r_a_km']*1e-6:.3f} Mkm,  "
        rf"$\omega$ = {best['omega_deg']:.1f}°,  "
        rf"$t_a$ = {best['t_arrival_days']:.1f} d,  "
        rf"flight-path angle = {fp_deg:.2f}°",
        y=1.02,
    )

    out_path = HERE / "wsb_search.png"
    fig.savefig(out_path)
    print(f"wrote {out_path.name}")


if __name__ == "__main__":
    main()
