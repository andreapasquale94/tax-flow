# Automatic Domain Splitting

The ADS module propagates a whole **set of initial conditions** — an
axis-aligned box — through a flow, keeping the result accurate by *subdividing*
the box wherever a single Taylor polynomial stops being faithful. The output is
a **tree of leaves**, each owning a sub-box and the flow polynomial valid on it,
which together tile the propagated set.

It builds directly on the [ODE integrator](../ode/index.md): every box is
carried forward by a `tax::ode` stepper over a DA-valued state
(`tax::la::VecNT<D, TaylorExpansion>`), and splitting is wired in through the
ODE [event system](../ode/events.md).

```cpp
#include <tax/ads.hpp>
#include <tax/ode.hpp>
using namespace tax::ode::methods;

tax::ads::Box<double, 2> ic_box{ center, half_width };

auto sol = tax::ads::propagate</*P=*/6>(
    Verner89{},
    tax::ads::TruncationCriterion{ /*tol=*/1e-4, /*maxDepth=*/8 },
    rhs, ic_box, ic_center, /*t0=*/0.0, /*t1=*/2 * M_PI, cfg, /*n_threads=*/8 );

const auto& tree = sol.tree();
for ( int li : tree.done() )
{
    const auto& leaf = tree.leaf( li );   // leaf.box, leaf.depth, leaf.payload
}
```

---

## Two strategies

The module ships two complementary ways to grow the tree. They return the
*same* `AdsTree`, so everything downstream — point lookup, the
[merger](api.md#merging), the [CSV writers](api.md#io) — is shared.

| | `propagate` — classic / in-flight | `refine` — propagate-then-assess |
|---|---|---|
| When a box is split | the moment its flow map stops converging, *mid-integration* | after a *full* propagation to `t_final` |
| Child initial state | the parent's partial map at the split time | a fresh identity on the child sub-box |
| Quality probe | one flow map, inspected in flight (`SplitCriterion`) | parent **vs. its children** at `t_final` (`QualityCriterion`) |
| Cost per box | one integration to the split time | three (self + two trial children) |
| Parallelism | independent boxes | the **whole recursion** fans out |
| Reference | [Wittig 2015](https://doi.org/10.1007/s10569-015-9618-3), [LOADS 2024](https://doi.org/10.2514/1.G007271) | this module — see [Refinement](refine.md) |

`propagate` is the established, frugal choice. `refine` spends more arithmetic
to buy a propagation pattern with no time-ordering constraints — see the
[refinement page](refine.md) and the
[runtime comparison](../tutorials/two_body_refine.md#runtime-vs-classic-ads).

---

## Pages in this section

| Page | Topic |
|---|---|
| [API Reference](api.md) | Domain interface, `Box`, `Zonotope`, `PolynomialZonotope`, `reorient`, `AdsTree`, criteria, `propagate`, `refine`, `merge` |
| [Refinement](refine.md) | The propagate-then-assess driver, quality indices, multi-way splitting, parallelism |
| [Zonotope](zonotope.md) | Oriented parallelotope domain — fewer leaves on anisotropic problems |
| [PolynomialZonotope](polynomial_zonotope.md) | Curved-IC domain + the tiered `Domain`/`LocatableDomain` interface |

See also the worked tutorials:
[Two-Body Problem](../tutorials/two_body.md) (classic ADS / LOADS) and
[Parallel ADS by Refinement](../tutorials/two_body_refine.md).

---

## Domain interface

The module is generic over the *domain* type — the geometric primitive a leaf
owns. Two concept tiers gate which features a domain can access:

| Concept | Key operations | Models |
|---------|---------------|--------|
| `Domain` | `center(dim)`, `split(dim)`, `domain_traits` | `Box`, `Zonotope`, `PolynomialZonotope` |
| `LocatableDomain` | `Domain` + `splitOrdinate(dim)` | `Box`, `Zonotope` |

`propagate` and `AdsDriver` require only `Domain`. `merge` requires
`LocatableDomain` (exact sibling ordering). `PolynomialZonotope` models `Domain`
but not `LocatableDomain`; its `contains` is a conservative over-approximation.

| Primitive | Header | Notes |
|-----------|--------|-------|
| `Box<T, M>` | `tax/ads/domains/box.hpp` | axis-aligned hyperrectangle; the default |
| `Zonotope<T, M>` | `tax/ads/domains/zonotope.hpp` | parallelotope (full `M×M` generator matrix) |
| `PolynomialZonotope<T,N,M>` | `tax/ads/domains/polynomial_zonotope.hpp` | polynomial image of the cube (curved IC) |

`include/tax/ads/domains/reorient.hpp` provides `reorientState`, `linearPart`,
`flowAlignedRotation`, and `reorientZonotope` — building blocks for adaptive
frame alignment (see [Zonotope](zonotope.md)).

---

## Core types

| Type | Role |
|---|---|
| `Box<T, M>` | axis-aligned IC hyperrectangle: `center`, `halfWidth` |
| `Zonotope<T, M>` | parallelotope IC domain: `center`, `generators` (M×M) |
| `PolynomialZonotope<T,N,M>` | polynomial-image IC domain: `value[i](ξ)` |
| `Leaf<Payload, M, T, Domain>` | one arena record: a domain, a `Payload`, tree links and metadata |
| `AdsTree<Payload, M, T, Domain>` | leaf-only arena tree with a BFS work queue |
| `AdsSolution<Stepper, M>` | result of `propagate`: wraps `AdsTree` + per-leaf ODE `Solution`s |
| `SplitCriterion` | in-flight split test for `propagate` (`shouldSplit`, `splitDim`) |
| `QualityCriterion` | parent-vs-children test for `refine` (`acceptable`, `splitDim`) |

The `Payload` is the DA-valued flow map — an `Eigen::Matrix<TaylorExpansion<T,
P, M>, D, 1>`. `M` is the number of factor (expansion) variables; `D` is the
state dimension.

---

## Split criteria

| Criterion | Used by | Idea | Constructor |
|---|---|---|---|
| `TruncationCriterion` | `propagate` | order-`P` coefficient mass exceeds `tol` (Wittig) | `{ tol, maxDepth }` |
| `NliCriterion` | `propagate` | Jacobian nonlinearity index exceeds `tol` (LOADS) | `{ tol, maxDepth }` |
| `CoefficientMatchCriterion` | `refine` | parent restricted to a half ≠ the child map | `{ tol, maxDepth }` |
| `VolumeRatioCriterion` | `refine` | parent image volume ≠ Σ child volumes | `{ tol, maxDepth, axes, nQuad }` |

---

## Headers

```cpp
#include <tax/ads.hpp>      // umbrella — every public ADS name
#include <tax/ode.hpp>      // the steppers ADS drives
```

| Header | Contents |
|---|---|
| `tax/ads/domains/domain.hpp` | `Domain`, `LocatableDomain` concepts; `domain_traits` |
| `tax/ads/domains/box.hpp` | `Box<T, M>` |
| `tax/ads/domains/zonotope.hpp` | `Zonotope<T, M>` |
| `tax/ads/domains/polynomial_zonotope.hpp` | `PolynomialZonotope<T, N, M>` |
| `tax/ads/domains/reorient.hpp` | `reorientState`, `linearPart`, `flowAlignedRotation`, `reorientZonotope` |
| `tax/ads/leaf.hpp` | `Leaf<Payload, M, T, Domain>` |
| `tax/ads/tree.hpp` | `AdsTree<Payload, M, T, Domain>` |
| `tax/ads/da_state.hpp` | `create`, `split` (DA-state identity & re-identification) |
| `tax/ads/split_criteria.hpp` | `SplitCriterion`, `TruncationCriterion`, `NliCriterion` |
| `tax/ads/split_event.hpp` | `SplitRequest`, `SplitEvent` |
| `tax/ads/driver.hpp` | `AdsDriver<Stepper, Criterion>` |
| `tax/ads/propagate.hpp` | `propagate<P>(method, criterion, …)` → `AdsSolution` |
| `tax/ads/solution.hpp` | `AdsSolution<Stepper, M>` |
| `tax/ads/refine_criteria.hpp` | `QualityCriterion`, `CoefficientMatchCriterion`, `VolumeRatioCriterion` |
| `tax/ads/refine.hpp` | `AdsRefineDriver<Stepper, Quality>`, `refine<P>(method, quality, …)` |
| `tax/ads/merge.hpp` | `merge(tree, criterion)`, `MergeStats` |
