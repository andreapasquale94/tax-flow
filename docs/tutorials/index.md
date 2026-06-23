# Tutorials

Worked, end-to-end examples that combine the three layers of the library —
the `TaylorExpansion` core, the `tax::ode` integrator, and `tax::ads`
Automatic Domain Splitting — on real dynamical systems. Each tutorial walks
through the math, the code, and the figures it produces.

The source code lives under [`examples/`](https://github.com/andreapasquale94/tax/tree/main/examples)
and builds with:

```bash
cmake -S . -B build -DTAX_BUILD_EXAMPLES=ON
cmake --build build -j
```

Each example writes a JSON file into its working directory; the plotting
scripts in `examples/plot/` turn those into the figures shown here
(`pip install matplotlib numpy`).

<div class="grid cards" markdown>

-   **[Two-body problem](two_body.md)**

    ---

    Propagate a whole *box* of initial conditions around an eccentric
    Kepler orbit with a single polynomial — and watch Automatic Domain
    Splitting take over when one polynomial stops being enough.

-   **[Parallel ADS by refinement](two_body_refine.md)**

    ---

    Flip ADS around: propagate every box to the end *first*, then split the
    initial conditions wherever the answer changes. Fully parallel, and it
    converges onto a Monte-Carlo reference as the box count grows.

-   **[Three-body problem](three_body.md)**

    ---

    Ride the unstable manifold of the Earth–Moon \(L_1\) point in the
    CR3BP, where the exponential stretching of the flow makes domain
    splitting unavoidable — and predictable.

-   **[Low-thrust reachability](reachability.md)**

    ---

    Expand the *control* instead of the initial state: propagate a whole
    set of thrust choices at once and read off the reachable-set envelope
    over one orbit, for a 1000 kg spacecraft and a 24U CubeSat.

-   **[Missed-thrust dispersion](missed_thrust.md)**

    ---

    Monte-Carlo Markov-chain outages meet a DA polynomial surrogate for
    execution errors: build the 1/2/3σ dispersion set over one revolution
    under three thruster-reliability scenarios, at the cost of a single
    flow-map propagation per Markov sequence.

-   **[Missed-thrust on/off duty cycle](missed_thrust_onoff.md)**

    ---

    A bang-bang thruster that trips ON/OFF on a 4-day grid, modelled as a
    two-state Markov chain: the same DA execution-error surrogate builds the
    1/2/3σ dispersion set across three thruster-reliability scenarios.

-   **[Low-thrust transfer dispersion](transfer_dispersion.md)**

    ---

    Design an Earth → NEA low-thrust transfer from a porkchop plot, then
    propagate initial-navigation and thrust-execution uncertainty boxes through
    it: the set-valued delivery dispersion shows thrust error dominating
    initial knowledge by ~75×.

</div>
