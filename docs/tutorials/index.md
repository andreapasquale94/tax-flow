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

</div>
