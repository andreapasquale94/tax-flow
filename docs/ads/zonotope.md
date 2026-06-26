# Oriented domains: the Zonotope prototype

ADS subdivides an **axis-aligned box** of initial conditions. On problems whose
uncertainty is *correlated* or *rotated* — a long, thin set lying along a
diagonal — the axis-aligned box that bounds it is much larger than the set
itself, carries more truncation mass, and therefore splits into more leaves
than the geometry warrants.

`Zonotope<T, M>` is a prototype alternative domain that addresses this. It is a
**parallelotope** — the unconstrained member of the constrained-zonotope family
(Scott et al. 2016; Kochdumper & Althoff 2020) — i.e. an affine image of the
cube:

```
Box:       { center + halfWidth ⊙ ξ : ξ ∈ [-1, 1]^M }   (diagonal scaling)
Zonotope:  { center + G · ξ        : ξ ∈ [-1, 1]^M }     (full generator matrix G)
```

A `Box` is exactly the case `G = diag(halfWidth)` (`Zonotope::axisAligned` /
`Zonotope::fromBox`). Giving `G` a general orientation lets a single leaf wrap a
rotated set, so it splits along the *oriented* generator directions rather than
the coordinate axes.

## Usage

```cpp
#include <tax/ads.hpp>
#include <tax/ode.hpp>
using namespace tax::ode::methods;

// Oriented thin initial set: G = R(θ) · diag(longHalf, thinHalf)
tax::ads::Zonotope<double, 2> ic;
ic.center     = center_vec;
ic.generators = R * scale;   // any M×M matrix

auto tree = tax::ads::propagate</*P=*/6>(
    Verner89{}, tax::ads::TruncationCriterion{1e-4, 8},
    rhs, ic, ic_center, 0.0, 2 * M_PI, cfg);

for (int i : tree.done()) {
    const auto& leaf = tree.leaf(i);
    // leaf.box is the leaf's Zonotope; recover local factors with
    //   ξ = leaf.box.generators.partialPivLu().solve(x - leaf.box.center);
    // then evaluate leaf.payload(ξ).
}
```

## Why it needs almost no new machinery

The DA flow-map payload and every split/merge substitution
(`da_state.hpp`, `merge.hpp`) operate in the **normalised factor coordinates**
`ξ ∈ [-1, 1]^M`, independent of how those factors map to physical space. Only
four operations are domain-aware — `contains`, `denormalize`, `split`,
`splitOrdinate` — so the entire pipeline (`Leaf`, `AdsTree`, `AdsDriver`,
`merge`) was generalised over a `Domain` type that **defaults to `Box<T, M>`**;
existing code and tests are unaffected. A generator-bisection split reuses the
same per-axis binomial re-identification (`substituteAxis`) the box split uses.

## Result

On the mildly nonlinear oscillator `ẍ = -x - 0.1x³` with a 45°-rotated thin
initial set (`tests/ads/test_zonotope.cpp`), at the same tolerance:

| Domain | Leaves |
|--------|--------|
| Oriented `Zonotope` | **3** |
| Its axis-aligned bounding `Box` | 7 |

## Worked example: two-body

`examples/two_body/zonotope.cpp` runs a correlated (45°-rotated) position/
velocity uncertainty set through one Kepler arc, propagating both the oriented
`Zonotope` and the axis-aligned box that bounds it. The oriented set needs
fewer leaves at every snapshot (e.g. **48 vs 75** mid-arc, **116 vs 129** at the
final time):

```bash
cmake -S . -B build -DTAXFLOW_BUILD_EXAMPLES=ON && cmake --build build -j
./build/examples/two_body_zonotope          # writes zonotope.json, zonotope_box.json
python3 examples/plot/plot_two_body_zonotope.py   # -> two_body_zonotope.png
```

The advantage is **configuration-dependent**: over most of the orbit the rotated
factors align with the flow and split less, but the strong shear of the
periapsis *return* is best resolved by axis-aligned cuts — there the box would
in turn split less. The example integrates the favourable arc and prints both
leaf counts at every snapshot so the trade-off is visible, not hidden.

## Adaptive orientation: aligning the frame to the flow

A *fixed* orientation is arbitrary — the two-body example above wins over most
of the orbit but loses at the periapsis return. The fix is to choose the frame
from the dynamics. `include/tax/ads/reorient.hpp` provides the primitives:

| Function | Role |
|----------|------|
| `reorientState(x, R)` | compose the flow map with a linear factor change: `y(η) = x(R·η)` |
| `linearPart(x)` | `A = ∂x/∂ξ\|₀`, the local STM (D×M) |
| `flowAlignedRotation(A)` | `V` from `SVD(A)`, so `A·V` has orthogonal columns |
| `reorientZonotope(z, R)` | keep generators in step: `G → G·R` |

**The geometric catch.** A linear change of factor variables `ξ = R·η` maps the
cube `[-1,1]^M` to `R·[-1,1]^M`, which is a cube again *only* when `R` is a
signed permutation. So re-orientation is exact only when (a) the uncertainty is
an **ellipsoid** — the covering parallelotope is then free to orient — or (b)
you re-wrap a leaf with a controlled over-approximation. The prototype
demonstrates case (a).

**Recipe (ellipsoidal IC).** Cover the covariance `Σ = L Lᵀ` with the
parallelotope `L·cube ⊇` ellipsoid. Probe the flow once (one un-split
propagation) to read `Φ = ∂x/∂ξ`, then pick `V = flowAlignedRotation(Φ·L)` and
use generators `G = L·V`. The propagated set `Φ·L·V = U·Σ` then has orthogonal
generators — no thin diagonal sliver to over-split. `examples/two_body/
zonotope_adaptive.cpp` compares three coverings of the *same* ellipsoid over a
**full period** (the case the fixed frame lost):

| Covering | Leaves @ T |
|----------|-----------|
| Flow-aligned `L·V` | **50** |
| Axis-aligned box | 56 |
| Fixed Cholesky `L` | 210 |

The flow-aligned covering needs fewest leaves *at every snapshot*, fixing the
periapsis flip — even though it covers a **larger** initial area than the box.
The win is from orientation, not size: a badly oriented frame (the fixed
Cholesky one) is 4× worse. The same run also re-expresses the probe map with
`reorientState` and confirms the deformation is diagonalised (the STM
off-diagonal drops from ~1.2 to ~0).

```bash
./build/examples/two_body_zonotope_adaptive
python3 examples/plot/plot_two_body_zonotope_adaptive.py   # -> two_body_zonotope_adaptive.png
```

## Scope and limitations (prototype)

- **Parallelotope only.** `G` is square (`M` generators); there is no support yet
  for *redundant* generators or for the equality **constraints** `Aξ = b` that
  make a constrained zonotope able to represent arbitrary polytopes and to
  split-by-constraint (no re-expansion). Adding a constraint store to the leaf
  and honouring it in `contains` / `merge` is the natural next step.
- **`refine.hpp` is not generalised** — the classic `AdsDriver` / `propagate`
  path is. The refine driver still assumes `Box`.
- **Orientation is chosen once, up front** (a single probe-STM alignment).
  `reorientState` is the building block for *time-adaptive* re-orientation —
  re-wrapping a leaf mid-flight when the local frame drifts — but that is not
  wired into the driver yet, and the mid-flight case needs the
  over-approximation accounting noted above.
- **`merge` ordering** of a sibling pair uses `splitOrdinate` (centre projected
  onto the split generator), which is correct for a parallelotope but has not
  been stressed on near-degenerate generator matrices.
- **`contains`** solves `G⁻¹(pt − center)` per query (an `M×M` LU). Fine for the
  `M ≲` few dimensions ADS targets; a stored factorisation would help at scale.
