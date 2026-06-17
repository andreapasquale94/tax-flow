#!/usr/bin/env python3
"""
examples/two_body/plot_validation.py

Render three figures from validation.json:

  1. two_body_validation_envelope.png — orbit overlay with MC samples
     and the single-Taylor envelope polygon at each snapshot.
  2. two_body_validation_error.png    — max + RMS position error vs
     truncation order P (one panel) and vs criterion tolerance (one
     panel), separately for ADS and LOADS. Taylor is shown as a
     baseline (depends on P only).
  3. two_body_validation_timing.png   — wall-clock vs P and vs tol,
     same layout as the error figure.
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
MEAN_N = 1.0


# ---- Matplotlib defaults (match plot.py) ------------------------------------
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


# ---- Kepler equation: t -> nu ----------------------------------------------
def true_anomaly(t: float, ecc: float = ECC, n: float = MEAN_N) -> float:
    M = n * t
    E = M
    for _ in range(40):
        d = (E - ecc * math.sin(E) - M) / (1.0 - ecc * math.cos(E))
        E -= d
        if abs(d) < 1e-13:
            break
    nu = 2.0 * math.atan2(
        math.sqrt(1.0 + ecc) * math.sin(0.5 * E),
        math.sqrt(1.0 - ecc) * math.cos(0.5 * E),
    )
    return nu % (2.0 * math.pi)


def load_validation() -> dict | None:
    p = HERE / "validation.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---- Figure 1: orbit + ADS envelope + MC scatter ---------------------------
def figure_envelope(data: dict) -> None:
    snapshots = data["snapshots"]
    ref       = data["reference_orbit"]
    cfg       = data["config"]
    env_P     = cfg.get("envelope_P", 6)
    env_tol   = cfg.get("envelope_tol", 1e-4)

    cmap = plt.cm.twilight
    norm = Normalize(vmin=0.0, vmax=2.0 * math.pi)

    fig, ax = plt.subplots(figsize=(5.2, 5.2), constrained_layout=True)

    # ---- Reference orbit ----
    ax.plot(ref["x0"], ref["x1"], color="#7c7c7c", lw=0.55, alpha=0.85, zorder=1)

    # ---- ADS leaf polygons + MC scatter ----
    for snap in snapshots:
        nu    = true_anomaly(snap["t"])
        color = cmap(norm(nu))
        for leaf in snap.get("ads_leaves", []):
            ax.fill(leaf["x"], leaf["y"], facecolor=color, alpha=0.30,
                    edgecolor=color, linewidth=0.7, zorder=2)
        ax.scatter(snap["mc_x"], snap["mc_y"], color=color, s=4.5,
                   edgecolor="black", linewidths=0.15, alpha=0.95, zorder=3)

    ax.plot(0.0, 0.0, marker="o", color="black", markersize=3.2, zorder=10)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect("equal", "box")
    ax.set_title(
        rf"MC samples + ADS envelope  ($P = {env_P}$, tol $= {env_tol:.0e}$)",
        loc="left",
    )

    # Shared colour bar.
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        orientation="vertical", fraction=0.045, pad=0.04)
    cbar.set_ticks([0, 0.5 * math.pi, math.pi, 1.5 * math.pi, 2.0 * math.pi])
    cbar.set_ticklabels(["0°", "90°", "180°", "270°", "360°"])
    cbar.set_label(r"true anomaly $\nu$", labelpad=2.0)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.tick_params(length=2.0, width=0.45, labelsize=6.5)

    fig.savefig(HERE / "two_body_validation_envelope.png")
    print("wrote two_body_validation_envelope.png")


# ---- Sweep slicing ---------------------------------------------------------
def cells_for(sweep: list[dict], method: str) -> list[dict]:
    return [c for c in sweep if c["method"] == method]


def cells_at_P(sweep: list[dict], method: str, P: int) -> list[dict]:
    return sorted([c for c in sweep if c["method"] == method and c["P"] == P],
                  key=lambda c: c["tol"] if c["tol"] is not None else float("inf"))


def cells_at_tol(sweep: list[dict], method: str, tol: float) -> list[dict]:
    return sorted([c for c in sweep
                   if c["method"] == method
                   and c["tol"] is not None
                   and abs(c["tol"] - tol) < 1e-15],
                  key=lambda c: c["P"])


def taylor_by_P(sweep: list[dict]) -> dict[int, dict]:
    return {c["P"]: c for c in sweep if c["method"] == "taylor"}


# ---- Figures 2 + 3 helpers --------------------------------------------------
METHOD_COLORS = {"ads": "#1f77b4", "loads": "#d62728"}
METHOD_LABEL  = {"ads": "ADS", "loads": "LOADS"}
TAYLOR_COLOR  = "#444444"


def _plot_vs_P(ax, sweep, *, tols, P_list, value_key, method,
               method_color, marker_cycle=("o", "s", "^", "D")):
    for i, tol in enumerate(tols):
        cells = cells_at_tol(sweep, method, tol)
        if not cells:
            continue
        Ps = [c["P"] for c in cells]
        ys = [c[value_key] for c in cells]
        ax.plot(Ps, ys, marker=marker_cycle[i % len(marker_cycle)],
                color=method_color, markersize=4, linewidth=1.1,
                markerfacecolor="white", markeredgewidth=0.9,
                label=f"tol = {tol:.0e}")
    # Taylor baseline (no tol; same value across tol-lines).
    taylors = taylor_by_P(sweep)
    Ps      = sorted(taylors.keys())
    ys      = [taylors[p][value_key] for p in Ps]
    ax.plot(Ps, ys, marker="x", linestyle="--", color=TAYLOR_COLOR,
            markersize=5, linewidth=1.0, label="Taylor")
    ax.set_xticks(P_list)
    ax.set_xlabel(r"truncation order $P$")


def _plot_vs_tol(ax, sweep, *, tols, P_list, value_key, method,
                 method_color, marker_cycle=("o", "s", "^", "D")):
    for i, P in enumerate(P_list):
        cells = cells_at_P(sweep, method, P)
        cells = [c for c in cells if c["tol"] is not None]
        if not cells:
            continue
        xs = [c["tol"] for c in cells]
        ys = [c[value_key] for c in cells]
        ax.plot(xs, ys, marker=marker_cycle[i % len(marker_cycle)],
                color=method_color, markersize=4, linewidth=1.1,
                markerfacecolor="white", markeredgewidth=0.9,
                label=f"P = {P}")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(tols)
    ax.set_xticklabels([f"{t:.0e}" for t in tols])
    ax.set_xlabel(r"criterion tolerance")


# ---- Figure 2: error vs (P, tol) -------------------------------------------
def figure_error(data: dict) -> None:
    sweep   = data["sweep"]
    tols    = data["config"]["tol_list"]
    P_list  = data["config"]["P_list"]

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.8), constrained_layout=True)

    for row, method in enumerate(("ads", "loads")):
        color = METHOD_COLORS[method]
        _plot_vs_P(axes[row][0], sweep,
                   tols=tols, P_list=P_list,
                   value_key="rms_pos_err",
                   method=method, method_color=color)
        _plot_vs_tol(axes[row][1], sweep,
                     tols=tols, P_list=P_list,
                     value_key="rms_pos_err",
                     method=method, method_color=color)
        axes[row][0].set_yscale("log")
        axes[row][1].set_yscale("log")
        axes[row][0].set_ylabel("RMS position error")
        axes[row][1].set_ylabel("RMS position error")
        axes[row][0].set_title(f"{METHOD_LABEL[method]} — vs P", loc="left")
        axes[row][1].set_title(f"{METHOD_LABEL[method]} — vs tol", loc="left")
        axes[row][0].legend(fontsize=6.5, ncol=2)
        axes[row][1].legend(fontsize=6.5, ncol=2)

    fig.savefig(HERE / "two_body_validation_error.png")
    print("wrote two_body_validation_error.png")


# ---- Figure 3: timing vs (P, tol) ------------------------------------------
def figure_timing(data: dict) -> None:
    sweep   = data["sweep"]
    tols    = data["config"]["tol_list"]
    P_list  = data["config"]["P_list"]

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.8), constrained_layout=True)

    for row, method in enumerate(("ads", "loads")):
        color = METHOD_COLORS[method]
        _plot_vs_P(axes[row][0], sweep,
                   tols=tols, P_list=P_list,
                   value_key="elapsed_ms",
                   method=method, method_color=color)
        _plot_vs_tol(axes[row][1], sweep,
                     tols=tols, P_list=P_list,
                     value_key="elapsed_ms",
                     method=method, method_color=color)
        axes[row][0].set_yscale("log")
        axes[row][1].set_yscale("log")
        axes[row][0].set_ylabel("elapsed (ms)")
        axes[row][1].set_ylabel("elapsed (ms)")
        axes[row][0].set_title(f"{METHOD_LABEL[method]} — vs P", loc="left")
        axes[row][1].set_title(f"{METHOD_LABEL[method]} — vs tol", loc="left")
        axes[row][0].legend(fontsize=6.5, ncol=2)
        axes[row][1].legend(fontsize=6.5, ncol=2)

    fig.savefig(HERE / "two_body_validation_timing.png")
    print("wrote two_body_validation_timing.png")


# ---- Main ------------------------------------------------------------------
def main() -> None:
    data = load_validation()
    if data is None:
        print("validation.json not found. Run two_body_validation first.")
        return

    figure_envelope(data)
    figure_error(data)
    figure_timing(data)


if __name__ == "__main__":
    main()
