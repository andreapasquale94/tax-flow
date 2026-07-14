# Refinement

`tax::ads::refine` grows the same `AdsTree` as classic
[`propagate`](api.md#propagate) but inverts the control flow: rather than
splitting a box the instant its flow map degrades mid-integration, it carries
**every box all the way to `t_final` first**, and only then decides whether the
box needed splitting at all.

This page is the reference for the algorithm and its tuning. For a narrative,
figure-driven walk-through see the tutorial
[Parallel ADS by Refinement](../tutorials/two_body_refine.md).

---

## The algorithm

For a box with identity state `init` and flow map `Φ = propagate(init)`:

1. **Pick directions.** Take the top-`split_dirs` axes by order-`P` coefficient
   mass (`detail::topKDims`). Call them `dims` (`k = |dims|`).
2. **Trial-split.** Form the `2^k` child sub-boxes; re-identify `init` onto each
   and propagate them to `t_final`, giving children `Φ_1 … Φ_{2^k}`.
3. **Assess.** Ask the [quality criterion](#quality-indices):
   `acceptable(Φ, {Φ_i}, dims, depth)`.
    - **accept** → finalize the box with `Φ`.
    - **reject** → keep the children (the trial propagations are *reused*, not
      recomputed) and recurse step 1 on each.

The recursion bottoms out when a box is accepted or reaches `maxDepth`.

Why compare against the children? A child re-expands its Taylor series on a
*smaller* domain, where the series converges better, so it is a strictly more
accurate model of the flow there. The parent-vs-children gap is therefore a
direct readout of the parent's **truncation error** — the quantity ADS exists
to control.

!!! note "Cost"
    Each box pays for its own propagation plus the two trial children needed to
    *prove* it is converged. Those trial propagations are fundamental to the
    method — they are exactly what an in-flight criterion avoids, which is why
    classic `propagate` is cheaper per leaf. Refinement trades that compute for
    a propagation pattern with no time-ordering constraints. See the
    [runtime comparison](../tutorials/two_body_refine.md#runtime-vs-classic-ads).

---

## Quality indices

Both indices share the split *direction* (top order-`P` coefficient mass) and
differ only in how they measure parent-vs-children disagreement.

### Coefficient match

`CoefficientMatchCriterion` — dimension-free. Re-identify the parent onto each
child's sub-box with the same affine substitution a split uses
(`ξ_d → ±0.5 + 0.5 ξ'_d`) and compare coefficient by coefficient:

$$
\delta \;=\; \max_i \;
  \frac{\big\| \Phi^{(i)}\!\restriction_{\text{child}} - \Phi^{(i)}_{\text{child}} \big\|_\infty}
       {\big\| \Phi^{(i)}_{\text{child}} \big\|_\infty},
\qquad \text{accept when } \delta \le \texttt{tol}.
$$

`tol` is a **relative coefficient error**. No geometry; works in any dimension;
the cheap default.

### Volume ratio

`VolumeRatioCriterion` — geometric, any dimension. Measure the `m`-volume of the
image of the box face (the integral of the Jacobian Gram determinant over the
`axes`) and compare the parent volume to the sum of the children's:

$$
V = \int_{[-1,1]^m}\!\!\sqrt{\det\!\big(J^\top J\big)}\,\mathrm d\xi
  \approx \frac{2^m}{n^m}\!\sum_{\text{grid}}\!\sqrt{\det\!\big(J^\top J\big)},
\qquad
\rho = \frac{V(\Phi)}{\sum_i V(\Phi_i)},
\qquad \text{accept when } |\rho - 1| \le \texttt{tol}.
$$

Because the children's domains exactly tile the parent's, an accurate parent has
`ρ ≈ 1`; stretching or folding past the radius of convergence drives `ρ` off 1
(and `|det|` does not cancel over a fold). `tol` is a **relative change in set
volume**. Set `axes` to the active box dimensions (those with nonzero
half-width) and `nQuad` to the grid density per axis. For two active axes `V` is
an image *area*.

The two indices ride the same accuracy-vs-leaf-count curve but, at a given
`tol`, stop at different resolutions because `tol` means different things to
each.

---

## Multi-way splitting

`split_dirs > 1` splits the top-`split_dirs` directions simultaneously into
`2^k` children per node (a cascade of `k` binary tree splits, assessed once).
The criterion `acceptable` takes the variable-length children span, so any child
count is handled uniformly; `split_dirs = 1` is bit-identical to plain binary
refinement.

In practice multi-way **does not reduce cost** for anisotropic problems: it
splits all chosen axes equally and cannot pour boxes into the single direction
that needs them, so it over-refines and loses accuracy per box (see the
[benchmark](../tutorials/two_body_refine.md#aggressive-multi-way-splitting)). It
is offered for genuinely isotropic nonlinearity. `split_dirs` is clamped to the
axes that still carry coefficient mass, so a box that is linear in some axis is
never split there.

---

## Parallelism and determinism

`refine` runs the recursion across `num_threads` workers pulling `WorkItem`s
from a shared queue. The expensive part — the trial propagations in `assess` —
runs **lock-free on copied-out inputs**; the mutex guards only the queue and the
tree mutation plus an in-flight counter. Termination is *queue empty AND nothing
in flight*; the first worker exception wins and is rethrown on the caller.

The accept/split decision for a box depends solely on that box and its trial
children — never on global ordering — so the resulting partition is **identical
for any `num_threads`**. Leaves are canonicalised by box centre, so `done()`
ordering and every coefficient match bit-for-bit between serial and parallel
runs. This is checked directly in `tests/ads/test_refine.cpp`.

---

## Choosing between methods

| Use… | returns | when |
|---|---|---|
| `propagate` (classic) | `AdsSolution<Stepper, M>` (tree + per-leaf ODE Solutions) | you want the cheapest partition; in-flight splitting is fine |
| `refine` + `CoefficientMatch` | bare `AdsTree` (flow maps only, no per-leaf Solutions) | you want a final-time accuracy guarantee, dimension-free, and full-recursion parallelism |
| `refine` + `VolumeRatio` | bare `AdsTree` (flow maps only, no per-leaf Solutions) | you specifically want to bound the growth of the reachable *set* (uncertainty / reachability) |

!!! note "Return-type difference"
    `tax::ads::propagate()` returns `AdsSolution<Stepper, M>`, which wraps the
    `AdsTree` together with per-leaf `Solution` objects (full step grids, events).
    `tax::ads::refine()` returns a bare `AdsTree` by design — the
    propagate-then-assess pattern does not keep intermediate ODE Solutions, so
    no `AdsSolution` wrapper is produced.

---

## Benchmark

[`benchmarks/bench_ads_refine.cpp`](https://github.com/andreapasquale94/tax/tree/main/benchmarks/bench_ads_refine.cpp)
(Google Benchmark) times classic ADS, binary refine, 4-way refine and the volume
ratio across 1–4 threads, plus classic ADS at orders `P = 2, 4, 6`:

```bash
cmake -S . -B build -DTAX_BUILD_BENCHMARK=ON && cmake --build build -j
./build/benchmarks/bench_ads_refine
```

The `leaves` counter reports partition size; divide time by leaves for a per-box
cost. Headline numbers and discussion are in the
[tutorial](../tutorials/two_body_refine.md#runtime-vs-classic-ads).
