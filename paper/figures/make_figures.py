#!/usr/bin/env python3
"""Render all figures for the paper from the experiment JSON in ../data.

Every figure uses only shades drawn from the Matplotlib *Blues* colormap.

Usage:
    python3 make_figures.py            # writes *.pdf into this directory
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.7,
    "mathtext.fontset": "cm",
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.8",
    "legend.fontsize": 7.5,
})

BL = plt.cm.Blues
C_DARK = BL(0.92)
C_MID = BL(0.66)
C_LIGHT = BL(0.45)
C_PALE = BL(0.28)
C_ORBIT = BL(0.20)
C_MC = BL(0.35)


def load(name):
    with open(DATA / name) as f:
        return json.load(f)


def parallelogram(c, J):
    c = np.asarray(c, float)
    J = np.asarray(J, float).reshape(2, 2)
    k = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]], float)
    return c[None, :] + k @ J.T


def depth_color(depth, dmax):
    return BL(0.30 + 0.62 * (depth / dmax if dmax > 0 else 0.0))


# ---------------------------------------------------------------------------
def fig_representations():
    d = load("representations.json")
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.2), layout="constrained")

    for a in ax:
        a.scatter(d["mc"]["x"], d["mc"]["y"], s=1.5, color=C_PALE, alpha=0.6,
                  rasterized=True, zorder=1)

    x0, x1, y0, y1 = d["box"]
    ax[0].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=C_LIGHT, lw=1.6,
               ls="--", label="interval / box", zorder=3)
    pg = parallelogram(d["zonotope"]["c"], d["zonotope"]["J"])
    ax[0].plot(pg[:, 0], pg[:, 1], color=C_MID, lw=1.7, label="zonotope (linear image)",
               zorder=4)
    ax[0].plot(d["poly_zonotope"]["x"], d["poly_zonotope"]["y"], color=C_DARK, lw=2.0,
               label="polynomial zonotope", zorder=5)
    ax[0].plot([], [], color=C_PALE, marker="o", ls="", ms=3, label="true set (MC)")
    ax[0].set_title(r"(a) box $\supseteq$ zonotope $\supseteq$ poly. zonotope")
    ax[0].legend(loc="upper left")

    for a, key, title in ((ax[1], "split2", "(b) 4 sub-domains"),
                          (ax[2], "split4", "(c) 16 sub-domains")):
        for fr in d[key]:
            pg = parallelogram(fr["c"], fr["J"])
            a.fill(pg[:, 0], pg[:, 1], facecolor=C_LIGHT, alpha=0.45, edgecolor=C_MID,
                   lw=0.7, zorder=3)
        a.plot(d["poly_zonotope"]["x"], d["poly_zonotope"]["y"], color=C_DARK, lw=1.0,
               ls=":", zorder=5)
        a.set_title(title)

    for a in ax:
        a.set_aspect("equal")
        a.set_xlabel("$x$")
        a.set_ylabel("$y$")
    fig.savefig(HERE / "fig_representations.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def _orbit(a, ref):
    a.plot(ref["x0"], ref["x1"], color=C_ORBIT, lw=0.8, zorder=1)


def fig_ads_partition():
    d = load("ads.json")
    snaps = d["snapshots"]
    dmax = max((lf["depth"] for s in snaps for lf in s["leaves"]), default=1)
    tf = float(d["params"]["t_final"])

    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.4), layout="constrained")
    _orbit(ax[0], d["reference_orbit"])
    ax[0].scatter([0], [0], marker="o", s=60, color=C_DARK, zorder=6)
    for lf in snaps[-1]["leaves"]:
        ax[0].fill(lf["x"], lf["y"], facecolor=depth_color(lf["depth"], dmax),
                   edgecolor=C_DARK, lw=0.3, alpha=0.9, zorder=3)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, dmax), cmap=BL)
    cb = fig.colorbar(sm, ax=ax[0], fraction=0.045, pad=0.02)
    cb.set_label("leaf depth")
    ax[0].set_title(r"(a) ADS partition at $t=T$")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("$x$")
    ax[0].set_ylabel("$y$")

    tau = [s["t"] / tf for s in snaps]
    nlv = [len(s["leaves"]) for s in snaps]
    ax[1].plot(tau, nlv, "-o", color=C_MID, ms=4)
    ax[1].set_title("(b) number of leaves")
    ax[1].set_xlabel(r"$t/T$")
    ax[1].set_ylabel("leaves")
    ax[1].grid(True, alpha=0.25)
    fig.savefig(HERE / "fig_ads_partition.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_oriented():
    z = load("zonotope.json")
    b = load("zonotope_box.json")
    p = z["params"]
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.5), layout="constrained")

    cy, cv = p["ic_center_yv"]
    hy, hv = p["bbox_hw_yv"]
    ax[0].plot([cy - hy, cy + hy, cy + hy, cy - hy, cy - hy],
               [cv - hv, cv - hv, cv + hv, cv + hv, cv - hv], color=C_LIGHT, lw=1.6,
               ls="--", label="bounding box")
    pg = parallelogram([cy, cv], p["ic_gen_yv"])
    ax[0].fill(pg[:, 0], pg[:, 1], facecolor=C_MID, alpha=0.35, edgecolor=C_DARK, lw=1.7,
               label="oriented zonotope")
    ax[0].set_title(r"(a) initial set, $(y, v_y)$")
    ax[0].set_xlabel("$y$")
    ax[0].set_ylabel("$v_y$")
    ax[0].set_aspect("equal")
    ax[0].legend(loc="upper left")

    tau = np.linspace(0, 1, len(p["zono_leaf_counts"]))
    ax[1].plot(tau, p["box_leaf_counts"], "-o", color=C_LIGHT, ms=4, label="box")
    ax[1].plot(tau, p["zono_leaf_counts"], "-o", color=C_DARK, ms=4, label="oriented")
    ax[1].set_title("(b) leaves vs time")
    ax[1].set_xlabel(r"$t/t_f$")
    ax[1].set_ylabel("leaves")
    ax[1].legend(loc="upper left")
    ax[1].grid(True, alpha=0.25)

    _orbit(ax[2], z["reference_orbit"])
    for lf in b["snapshots"][-1]["leaves"]:
        ax[2].fill(lf["x"], lf["y"], facecolor="none", edgecolor=C_LIGHT, lw=0.7,
                   zorder=2)
    for lf in z["snapshots"][-1]["leaves"]:
        ax[2].fill(lf["x"], lf["y"], facecolor=C_MID, alpha=0.30, edgecolor=C_DARK,
                   lw=0.4, zorder=3)
    nb = len(b["snapshots"][-1]["leaves"])
    nz = len(z["snapshots"][-1]["leaves"])
    ax[2].plot([], [], color=C_LIGHT, lw=1.2, label=f"box ({nb})")
    ax[2].fill([], [], color=C_MID, alpha=0.5, label=f"oriented ({nz})")
    ax[2].set_title(r"(c) leaf tiling at $t_f$")
    ax[2].set_xlabel("$x$")
    ax[2].set_ylabel("$y$")
    ax[2].set_aspect("equal")
    ax[2].legend(loc="upper left")
    fig.savefig(HERE / "fig_oriented.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_adaptive():
    d = load("zonotope_adaptive.json")
    p = d["params"]
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.5), layout="constrained")

    c = p["ic_center_yv"]

    def ell(cov):
        C = np.asarray(cov, float).reshape(2, 2)
        L = np.linalg.cholesky(C)
        th = np.linspace(0, 2 * np.pi, 200)
        return np.asarray(c)[None, :] + np.stack([np.cos(th), np.sin(th)], 1) @ L.T

    e = ell(p["cov"])
    ax[0].plot(e[:, 0], e[:, 1], color=C_DARK, lw=1.3, ls=":", label=r"$1\sigma$ ellipse")
    hy, hv = p["box_hw_yv"]
    ax[0].plot([c[0] - hy, c[0] + hy, c[0] + hy, c[0] - hy, c[0] - hy],
               [c[1] - hv, c[1] - hv, c[1] + hv, c[1] + hv, c[1] - hv], color=C_LIGHT,
               lw=1.5, ls="--", label="box")
    pc = parallelogram(c, p["chol_yv"])
    ax[0].plot(pc[:, 0], pc[:, 1], color=C_MID, lw=1.5, label="Cholesky")
    pa = parallelogram(c, p["aligned_yv"])
    ax[0].fill(pa[:, 0], pa[:, 1], facecolor=C_MID, alpha=0.30, edgecolor=C_DARK, lw=1.7,
               label="flow-aligned")
    ax[0].set_title(r"(a) one ellipsoid, three coverings")
    ax[0].set_xlabel("$y$")
    ax[0].set_ylabel("$v_y$")
    ax[0].set_aspect("equal")
    ax[0].legend(loc="upper left")

    tau = np.linspace(0, 1, len(p["box_leaf_counts"]))
    ax[1].semilogy(tau, p["chol_leaf_counts"], "-o", color=C_PALE, ms=3.5,
                   label="Cholesky")
    ax[1].semilogy(tau, p["box_leaf_counts"], "-o", color=C_MID, ms=3.5, label="box")
    ax[1].semilogy(tau, p["aligned_leaf_counts"], "-o", color=C_DARK, ms=3.5,
                   label="flow-aligned")
    ax[1].set_title("(b) leaves over a full period")
    ax[1].set_xlabel(r"$t/T$")
    ax[1].set_ylabel("leaves")
    ax[1].legend(loc="upper left")
    ax[1].grid(True, which="both", alpha=0.25)

    snaps = d["snapshots"]
    dmax = max((lf["depth"] for lf in snaps[-1]["leaves"]), default=1)
    _orbit(ax[2], d["reference_orbit"])
    for lf in snaps[-1]["leaves"]:
        ax[2].fill(lf["x"], lf["y"], facecolor=depth_color(lf["depth"], dmax),
                   edgecolor=C_DARK, lw=0.3, alpha=0.9)
    ax[2].set_title(r"(c) flow-aligned tiling at $t=T$")
    ax[2].set_xlabel("$x$")
    ax[2].set_ylabel("$y$")
    ax[2].set_aspect("equal")
    fig.savefig(HERE / "fig_adaptive.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def _zoom(ax, xs, ys, frac=0.1):
    px = frac * (max(xs) - min(xs) + 1e-9)
    py = frac * (max(ys) - min(ys) + 1e-9)
    ax.set_xlim(min(xs) - px, max(xs) + px)
    ax.set_ylim(min(ys) - py, max(ys) + py)


def fig_curved():
    d = load("poly_zonotope.json")
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.6), layout="constrained")

    ax[0].scatter(d["ic_mc"]["x"], d["ic_mc"]["y"], s=1.5, color=C_PALE, alpha=0.6,
                  rasterized=True, label="Monte Carlo")
    ax[0].plot(d["ic_linear"]["x"], d["ic_linear"]["y"], color=C_MID, lw=1.7, ls="--",
               label="linear (tangent)")
    ax[0].plot(d["ic_curved"]["x"], d["ic_curved"]["y"], color=C_DARK, lw=1.9,
               label="polynomial zonotope")
    ax[0].set_title(r"(a) initial set $(x, y)$")
    ax[0].set_xlabel("$x$")
    ax[0].set_ylabel("$y$")
    ax[0].set_aspect("equal")
    ax[0].legend(loc="best")

    for a, key, title in (
        (ax[1], "leaves_curved",
         f"(b) polynomial zonotope\n{d['n_curved']} leaves, "
         f"RMS {d['rms_curved']:.1e}"),
        (ax[2], "leaves_linear",
         f"(c) linear (tangent)\n{d['n_linear']} leaves, RMS {d['rms_linear']:.1e}")):
        _orbit(a, d["reference_orbit"])
        xs, ys = list(d["mc"]["x"]), list(d["mc"]["y"])
        for lf in d[key]:
            a.fill(lf["x"], lf["y"], facecolor=C_MID, alpha=0.28, edgecolor=C_DARK,
                   lw=0.5)
            xs += lf["x"]
            ys += lf["y"]
        a.scatter(d["mc"]["x"], d["mc"]["y"], s=1.5, color=C_PALE, alpha=0.6,
                  rasterized=True)
        _zoom(a, xs, ys)
        a.set_title(title)
        a.set_xlabel("$x$")
        a.set_ylabel("$y$")
        a.set_aspect("equal")
    fig.savefig(HERE / "fig_curved.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
def fig_mc():
    d = load("two_body_mc_ellipse.json")
    doms = d["domains"]
    fig, ax = plt.subplots(1, len(doms), figsize=(3.9 * len(doms), 3.7),
                           layout="constrained")
    mx, my = d["mc"]["x"], d["mc"]["y"]
    px = 0.08 * (max(mx) - min(mx))
    py = 0.08 * (max(my) - min(my))
    for a, dom in zip(ax, doms):
        _orbit(a, d["reference_orbit"])
        for lf in dom["leaves"]:
            a.fill(lf["x"], lf["y"], facecolor=C_MID, alpha=0.22, edgecolor=C_DARK,
                   lw=0.4)
        a.scatter(mx, my, s=1.3, color=C_PALE, alpha=0.6, rasterized=True)
        a.set_title(f"{dom['name']}\n{dom['n_leaves']} leaves, "
                    f"RMS {dom['rms']:.1e}")
        a.set_xlim(min(mx) - px, max(mx) + px)
        a.set_ylim(min(my) - py, max(my) + py)
        a.set_aspect("equal")
        a.set_xlabel("$x$")
        a.set_ylabel("$y$")
    fig.savefig(HERE / "fig_mc.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_representations()
    fig_ads_partition()
    fig_oriented()
    fig_adaptive()
    fig_curved()
    fig_mc()
    print("wrote:", ", ".join(sorted(p.name for p in HERE.glob("*.pdf"))))
