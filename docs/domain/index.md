# `tax::domain` — set-valued domains

The domain module describes **sets of initial conditions** as images of the
normalized factor cube `ξ ∈ [-1,1]^M`, and provides the query/enclosure layer
that turns propagated DA flow maps back into consumable sets. It is consumed
by [`tax::ads`](../ads/index.md) (every leaf owns a domain) and usable
standalone.

```cpp
#include <tax/domain.hpp>   // umbrella — every public domain name
```

---

## Primitives

| Primitive | Header | Notes |
|-----------|--------|-------|
| `Box<T, M>` | `tax/domain/box.hpp` | axis-aligned hyperrectangle; the default |
| `Zonotope<T, M>` | `tax/domain/zonotope.hpp` | parallelotope (full `M×M` generator matrix) — see [Zonotope](zonotope.md) |
| `PolynomialZonotope<T,N,M>` | `tax/domain/polynomial_zonotope.hpp` | polynomial image of the cube (curved sets) — see [PolynomialZonotope](polynomial_zonotope.md) |

All three are images of the cube: `Box` scales it by a diagonal, `Zonotope` by
a dense matrix, `PolynomialZonotope` by a polynomial map. Constructors:

```cpp
Box<double, 4>  b{ center, halfWidth };
auto z1 = Zonotope<double, 2>::axisAligned(center, halfWidth);
auto z2 = Zonotope<double, 2>::fromBox(b2);
auto z3 = Zonotope<double, 2>::oriented(center, R, halfWidth);  // R·diag(hw)
auto pz = PolynomialZonotope<double, 6, 2>::fromBox(b2);
```

## Concept tiers

```cpp
// tax/domain/domain.hpp
template <class D> concept Domain = /* center, split, denormalize, traits */;
template <class D> concept LocatableDomain =
    Domain<D> && /* localize, contains, splitOrdinate */;
```

| Concept | Adds | Models |
|---------|------|--------|
| `Domain` | `center(dim)`, `split(dim)`, `denormalize(ξ)` | `Box`, `Zonotope`, `PolynomialZonotope` |
| `LocatableDomain` | `localize(pt)` (exact inverse of `denormalize`), `contains(pt)`, `splitOrdinate(dim)` | `Box`, `Zonotope` |

`tax::ads::propagate` / `AdsDriver` need only `Domain`; point location
(`AdsTree::locate`/`locateFactors`, `AdsSolution::evaluate`) and `merge` need
`LocatableDomain`. A polynomial image has no closed-form inverse, so
`PolynomialZonotope` stays on the lower tier (its `contains` is a
conservative hull test, deliberately not part of either tier).

`localize` is rank-safe: `Zonotope::localize` solves a rank-revealing
least-squares, so generator matrices with deliberately zeroed (pinned) axes
recover the active factors and pin the rest to 0 — mirroring
`Box::localize` on zero-width axes. `contains` additionally checks the
reconstruction residual, so off-span points are rejected.

## DA-state bridge

```cpp
// tax/domain/create.hpp (+ the PZ overload in polynomial_zonotope.hpp)
auto x0da = tax::domain::create<P, M>(domain, x0);   // identity DA state on domain
```

`x0` supplies all `D` state components; its leading `M` components **must**
equal the domain's center (asserted in debug builds) — downstream point
location reads the center from the domain's geometry while the payload's
constant term comes from `x0`.

## Enclosures of a DA image

A DA state over the cube *is* a polynomial zonotope, so the image of a leaf
payload can be bounded without any new representation
(`tax/domain/enclosure.hpp`):

```cpp
auto hull = tax::domain::intervalHull(state);                    // Box<T, D>
auto hullXY = tax::domain::intervalHull(state, std::array{0, 1}); // Box<T, 2>

// Zonotope over-approximation via the even-exponent shift
// (Kochdumper & Althoff 2020): always at least as tight as the interval hull.
auto z = tax::domain::zonotopeEnclosure(state, std::array{0, 1}); // ImageZonotope<T, 2>
double rho = z.support(direction);       // support function
auto zh = z.intervalHull();              // Box<T, 2>

// Exact degree-1 frame (center = constant terms, G = linear coefficients):
// an enclosure only where the map is linear, a frame otherwise.
auto f = tax::domain::zonotopeFrame(state);           // Zonotope<T, M>
```

Domain-level hulls: `Zonotope::intervalHull()` (L1 row norms — the hull of
the parallelotope) and `PolynomialZonotope::intervalHull()` (coefficient
sums).

## Ellipsoid coverings

For a covariance ellipsoid `E = { c + L·u : ‖u‖₂ ≤ 1 }`
(`tax/domain/ellipsoid.hpp`):

```cpp
auto cover = tax::domain::ellipsoidCover(c, L);        // parallelotope ⊇ E
auto coverV = tax::domain::ellipsoidCover(c, L, V);    // any orthogonal V still covers
auto hull  = tax::domain::ellipsoidIntervalHull(c, L); // EXACT hull of E (L2 row norms)
```

Note the two hull notions: `Zonotope::intervalHull` uses **L1** row norms
(hull of the parallelotope), `ellipsoidIntervalHull` uses **L2** row norms
(hull of the ellipsoid the parallelotope covers). Handing the ellipsoid hull
to a box ADS run covers `E` but *not* `ellipsoidCover(c, L)`.

## Factor-frame reorientation

`tax/domain/reorient.hpp` provides `reorientState` (compose the flow map with
`ξ = R·η`), `linearPart` (the local STM), `flowAlignedRotation` (SVD frame)
and `reorientZonotope` (`G → G·R`) — the building blocks for flow-aligned
oriented ADS. See [Zonotope — adaptive orientation](zonotope.md#adaptive-orientation-aligning-the-frame-to-the-flow)
for the geometry (and its caveats: a generic rotation of the factor cube
changes the represented set unless the uncertainty is an ellipsoid).

## Headers

| Header | Contents |
|---|---|
| `tax/domain/domain.hpp` | `Domain`, `LocatableDomain` concepts; `domain_traits` |
| `tax/domain/box.hpp` | `Box<T, M>` |
| `tax/domain/zonotope.hpp` | `Zonotope<T, M>` |
| `tax/domain/polynomial_zonotope.hpp` | `PolynomialZonotope<T, N, M>` (+ its `create`) |
| `tax/domain/create.hpp` | `create` (domain → identity DA state) |
| `tax/domain/enclosure.hpp` | `intervalHull`, `zonotopeEnclosure`, `zonotopeFrame`, `ImageZonotope` |
| `tax/domain/ellipsoid.hpp` | `ellipsoidCover`, `ellipsoidIntervalHull` |
| `tax/domain/reorient.hpp` | `reorientState`, `linearPart`, `flowAlignedRotation`, `reorientZonotope` |
| `tax/domain/detail/substitute_axis.hpp` | the per-axis split/merge substitution |
