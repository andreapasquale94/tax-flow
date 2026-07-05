#!/usr/bin/env python3
"""Render the validated Taylor-model two-body tutorial figures.

Input:  taylor_model.json  (written by ./two_body_taylor_model)
Output: four PNGs (default into the current directory; pass an output dir):

  two_body_tm_flow.png       - snapshot polygons + rigorous hull rectangles
  two_body_tm_remainder.png  - remainder growth: single model vs ADS leaves
  two_body_tm_partition.png  - final IC-box partition colored by remainder
  two_body_tm_enclosure.png  - nested set representations on one leaf

Usage: python3 plot_two_body_taylor_model.py [taylor_model.json] [outdir]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

path = Path(sys.argv[1] if len(sys.argv) > 1 else "taylor_model.json")
outdir = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
outdir.mkdir(parents=True, exist_ok=True)
data = json.loads(path.read_text())

params = data["params"]
ref = data["reference_orbit"]
single = data["single"]
ads = data["ads"]
conv = data["conversions"]
T = params["t_final"]

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def zono_polygon(c, gens):
    """Vertices of a 2-D zonotope: sort generators by angle and walk."""
    g = np.array([gv for gv in gens if abs(gv[0]) + abs(gv[1]) > 0.0])
    if len(g) == 0:
        return np.array([c])
    g = np.where(g[:, 1:2] < 0, -g, g)  # flip into upper half-plane
    order = np.argsort(np.arctan2(g[:, 1], g[:, 0]))
    g = g[order]
    half = np.array(c) - g.sum(axis=0) / 1.0  # start at support of -y direction
    pts = [np.array(c) - g.sum(axis=0)]
    for gv in g:
        pts.append(pts[-1] + 2 * gv)
    for gv in g:
        pts.append(pts[-1] - 2 * gv)
    return np.array(pts)


# ---------------------------------------------------------------- fig 1: flow
fig, ax = plt.subplots(figsize=(7.2, 5.6))
ax.plot(ref["x"], ref["y"], color="0.75", lw=1.0, zorder=1, label="reference orbit")
ax.plot([0], [0], marker="+", color="0.4", ms=10, mew=1.4)

snaps = single["snapshots"]
cmap = plt.cm.viridis
for i, s in enumerate(snaps):
    col = cmap(i / max(1, len(snaps) - 1))
    ax.fill(s["poly"]["x"], s["poly"]["y"], color=col, alpha=0.55, lw=0.8, ec=col, zorder=3)

# rigorous hull rectangles on the recording grid (every 4th to avoid clutter)
for h in single["hulls"][::6]:
    ax.add_patch(
        Rectangle(
            (h["cx"] - h["hx"], h["cy"] - h["hy"]), 2 * h["hx"], 2 * h["hy"],
            fill=False, ec="crimson", lw=0.6, alpha=0.8, zorder=2,
        )
    )

reached = params["single_reached"]
ax.set_title(
    f"One validated Taylor model over the IC box (P={int(params['P'])})\n"
    f"polynomial images (filled), rigorous hulls (red) — verified to t = {reached:.2f} of {T:.2f}",
    fontsize=10,
)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(outdir / "two_body_tm_flow.png")
plt.close(fig)

# ----------------------------------------------------------- fig 2: remainder
fig, ax = plt.subplots(figsize=(7.2, 4.4))
t1 = np.array(single["t"])
rem = np.maximum.reduce([np.array(single[f"rem{i}"]) for i in range(4)])
mask = rem > 1e-12  # drop the exact-zero start (denormal floor)
ax.semilogy(t1[mask], rem[mask], color="crimson", lw=1.6, label="single Taylor model")

t2 = np.array(ads["snap_t"])
r2 = np.array(ads["snap_rem_max"])
m2 = r2 > 1e-12
ax.semilogy(t2[m2], r2[m2], "o-", color="tab:blue", lw=1.6, ms=4,
            label=f"ADS, worst leaf ({int(params['ads_leaves'])} leaves)")

ax.set_ylim(3e-9, 1.0)
if reached < T - 1e-9:
    ax.axvline(reached, color="crimson", ls=":", lw=1.2)
    ax.annotate(
        "verification fails —\nsingle model cannot continue",
        xy=(reached, rem[mask][-1]), xytext=(reached - 2.4, 3e-3),
        fontsize=8, color="crimson",
        arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8),
    )

for tmark, lab in [(0.0, "periapsis"), (np.pi, "apoapsis")]:
    ax.axvline(tmark, color="0.85", lw=0.8, zorder=0)
    ax.text(tmark, 5e-9, f" {lab}", fontsize=7, color="0.5", rotation=90, va="bottom")

ax.set_xlabel("t")
ax.set_ylabel("max remainder width (rigorous error bound)")
ax.set_title("The price of one global model: remainder growth, single vs ADS")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(outdir / "two_body_tm_remainder.png")
plt.close(fig)

# ----------------------------------------------------------- fig 3: partition
fig, ax = plt.subplots(figsize=(6.6, 5.2))
leaves = ads["leaves"]
rems = np.array([l["rem"] for l in leaves])
norm = plt.matplotlib.colors.LogNorm(vmin=rems.min(), vmax=rems.max())
cm = plt.cm.plasma
for l in leaves:
    col = cm(norm(l["rem"]))
    ax.add_patch(
        Rectangle(
            (l["y"][0], l["vy"][0]), l["y"][1] - l["y"][0], l["vy"][1] - l["vy"][0],
            facecolor=col, edgecolor="white", lw=1.2,
        )
    )
    cx = 0.5 * (l["y"][0] + l["y"][1])
    cy = 0.5 * (l["vy"][0] + l["vy"][1])
    ax.text(cx, cy, str(l["depth"]), ha="center", va="center", fontsize=8,
            color="white", weight="bold")

ic_c = params["ic_center"]
ic_h = params["ic_half_width"]
ax.set_xlim(ic_c[1] - 1.15 * ic_h[1], ic_c[1] + 1.15 * ic_h[1])
ax.set_ylim(ic_c[3] - 1.15 * ic_h[3], ic_c[3] + 1.15 * ic_h[3])
sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
fig.colorbar(sm, ax=ax, label="leaf remainder width at t = T")
ax.set_xlabel("$y_0$")
ax.set_ylabel("$v_{y,0}$")
ax.set_title(
    f"ADS partition of the IC box ({int(params['ads_leaves'])} leaves, numbers = depth)\n"
    "each leaf carries its own validated remainder"
)
fig.tight_layout()
fig.savefig(outdir / "two_body_tm_partition.png")
plt.close(fig)

# ----------------------------------------------------------- fig 4: enclosure
fig, ax = plt.subplots(figsize=(6.8, 5.4))

hull = conv["hull"]
ax.add_patch(
    Rectangle(
        (hull["cx"] - hull["hx"], hull["cy"] - hull["hy"]), 2 * hull["hx"], 2 * hull["hy"],
        fill=False, ec="crimson", lw=1.6, label="intervalHull (rigorous box)",
    )
)

pz = zono_polygon(conv["enclosure"]["c"], conv["enclosure"]["gens"])
ax.fill(pz[:, 0], pz[:, 1], color="tab:blue", alpha=0.18, lw=1.4, ec="tab:blue",
        label="zonotopeEnclosure (rigorous)")

pf = zono_polygon(conv["frame"]["c"], conv["frame"]["gens"])
ax.plot(np.append(pf[:, 0], pf[0, 0]), np.append(pf[:, 1], pf[0, 1]),
        color="tab:green", lw=1.2, ls="--", label="zonotopeFrame (linear part only)")

ax.plot(conv["poly"]["x"], conv["poly"]["y"], color="0.3", lw=1.0,
        label="polynomial boundary image")
ax.scatter(conv["mc_x"], conv["mc_y"], s=5, color="k", zorder=5,
           label="Monte-Carlo truth")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")
ax.set_title(
    "Domain conversions on one ADS leaf at t = T\n"
    "truth $\\subset$ enclosures; the frame ignores curvature and remainder"
)
ax.legend(loc="best", fontsize=8)
fig.tight_layout()
fig.savefig(outdir / "two_body_tm_enclosure.png")
plt.close(fig)

mc_total = int(params["mc_total"])
mc_loc = int(params["mc_located"])
mc_in = int(params["mc_contained"])
print(f"wrote 4 figures to {outdir}/")
print(f"MC containment: {mc_in}/{mc_loc} located (of {mc_total} sampled)")
