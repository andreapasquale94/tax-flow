# Nominal-transfer design (preparatory)

These three Python scripts produce the **nominal** Earth → near-Earth-asteroid
(NEA) transfer that the dispersion example (`../transfer_dispersion.cpp`)
propagates uncertainty boxes through. They are kept here so the trajectory can
be regenerated or retargeted; they are **not** part of the C++ build.

Pipeline (run in order, in this directory):

```bash
pip install numpy scipy matplotlib lamberthub
python3 porkchop.py        # → pork.npz, porkchop.png   (impulsive baseline)
python3 finite_burn.py     # → lt.npz                   (finite-burn fit, 3 thrust levels)
python3 design_plots.py    # → lt_inertial.png, lt_rotating.png
```

1. **`porkchop.py`** — heliocentric planar two-body porkchop (multi-revolution
   Lambert via [`lamberthub`](https://github.com/jorgepiloto/lamberthub)),
   Earth → NEA. Picks the minimum-Δv impulsive rendezvous (departure epoch,
   time of flight) and auto-tunes the NEA phase so a cheap window falls within
   a few years.
2. **`finite_burn.py`** — transcribes the impulsive optimum into a
   constant-direction **thrust–coast–thrust** finite-burn rendezvous for three
   thrust levels (0.10 / 0.20 / 0.30 N at 1000 kg), solving the burn structure
   `(tau1, phi1, tau2, phi2)` with `fsolve` over a flight-time search. The
   excess Δv over the impulsive optimum is the low-thrust penalty.
3. **`design_plots.py`** — renders the three trajectories in the inertial and
   Sun–Earth rotating frames.

The burn parameters printed by `finite_burn.py` are the `Preset` constants
hard-coded in [`../common.hpp`](../common.hpp).
