#!/usr/bin/env python3
"""Render the conjunction / collision-probability example figures.

Usage:
    python3 plot_conjunction.py [--data DIR] [--out DIR]

Reads the files written by the conjunction_pc example (run it first; it writes
to its working directory):

    conjunction.json            — scalars (TCA, miss, vrel, Pc's, B-plane cov)
    conjunction_bplane.csv      — B-plane scatter (truth + polynomial MC)
    conjunction_fidelity.csv    — |poly - SGP4| per order vs perturbation size

and produces:

    conjunction_bplane.png      — the encounter B-plane: uncertainty footprint,
                                  hard-body disk, polynomial-MC vs SGP4 cloud
    conjunction_fidelity.png    — map error vs k-sigma for orders 1..4 (why
                                  higher order matters)
"""

import argparse
import csv
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "font.size": 11,
        "axes.linewidth": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)


def load(data):
    meta = json.loads((data / "conjunction.json").read_text())
    rows = {"truth": [], "poly": []}
    with open(data / "conjunction_bplane.csv") as f:
        for r in csv.DictReader(f):
            rows[r["source"]].append((float(r["xi"]), float(r["zeta"])))
    fid = {"k": [], 1: [], 2: [], 3: [], 4: []}
    with open(data / "conjunction_fidelity.csv") as f:
        for r in csv.DictReader(f):
            fid["k"].append(float(r["k_sigma"]))
            for o in (1, 2, 3, 4):
                fid[o].append(float(r[f"order{o}"]))
    return meta, {k: np.array(v) for k, v in rows.items()}, fid


def ellipse(mean, cov, nsig, n=200):
    th = np.linspace(0, 2 * np.pi, n)
    L = np.linalg.cholesky(cov)
    circle = np.vstack([np.cos(th), np.sin(th)])
    pts = mean[:, None] + nsig * (L @ circle)
    return pts[0], pts[1]


def plot_bplane(meta, rows, out):
    cov = np.array(meta["cov2d"]).reshape(2, 2)
    miss = np.array(meta["miss2d"])
    R = meta["hard_body_km"]
    fig, ax = plt.subplots(figsize=(6.4, 6.0))

    tr, po = rows["truth"], rows["poly"]
    ax.scatter(tr[:, 0], tr[:, 1], s=3, c="0.55", alpha=0.35, label="SGP4 Monte Carlo")
    ax.scatter(po[:, 0], po[:, 1], s=3, c="tab:blue", alpha=0.25,
               label=f"order-{meta['order']} polynomial MC")

    # primary at origin; the cloud is centred on the miss vector.
    for ns, c in [(1, "tab:red"), (3, "tab:orange")]:
        ex, ey = ellipse(miss, cov, ns)
        ax.plot(ex, ey, c=c, lw=1.6, label=f"{ns}σ linear (Gaussian)")
    th = np.linspace(0, 2 * np.pi, 100)
    ax.plot(R * np.cos(th), R * np.sin(th), "k-", lw=2.0, label=f"hard body R={R*1000:.0f} m")
    ax.plot(0, 0, "k+", ms=10)
    ax.plot(miss[0], miss[1], "x", c="k", ms=9, label="nominal miss")

    ax.set_aspect("equal")
    ax.set_xlabel(r"B-plane $\xi$  [km]")
    ax.set_ylabel(r"B-plane $\zeta$  [km]")
    ax.set_title(
        f"Encounter B-plane  (miss {meta['miss_km']:.3f} km, "
        f"$v_{{rel}}$ {meta['vrel_kms']:.2f} km/s)\n"
        f"Pc: linear {meta['pc_linear']:.2e}   "
        f"order-{meta['order']} {meta['pc_poly_orderN']:.2e}   "
        f"truth {meta['pc_truth']:.2e}"
    )
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out / "conjunction_bplane.png", dpi=140)
    print("wrote", out / "conjunction_bplane.png")


def plot_fidelity(fid, out):
    k = np.array(fid["k"])
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    colors = {1: "tab:red", 2: "tab:orange", 3: "tab:green", 4: "tab:blue"}
    for o in (1, 2, 3, 4):
        e = np.array(fid[o])
        e = np.where(e <= 0, np.nan, e)
        ax.semilogy(k, e, "o-", c=colors[o], lw=1.6, ms=4, label=f"order {o}")
    ax.axhline(1e-3, color="0.6", ls="--", lw=1.0)
    ax.text(0.1, 1.2e-3, "1 m", color="0.4", fontsize=9)
    ax.set_xlabel(r"perturbation size  [$\sigma$]")
    ax.set_ylabel(r"$|\,$poly $-$ SGP4$\,|$  position error  [km]")
    ax.set_title("Map fidelity: the linear map breaks down,\nhigher orders track SGP4 far from the nominal")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "conjunction_fidelity.png", dpi=140)
    print("wrote", out / "conjunction_fidelity.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".", type=pathlib.Path)
    ap.add_argument("--out", default=".", type=pathlib.Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    meta, rows, fid = load(args.data)
    plot_bplane(meta, rows, args.out)
    plot_fidelity(fid, args.out)


if __name__ == "__main__":
    main()
