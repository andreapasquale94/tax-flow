# Polynomial Zonotope domain

`PolynomialZonotope<T, N, M, Storage>` extends the [Zonotope](zonotope.md)
idea to **curved** initial sets: the IC set is the image of the cube
`[-1, 1]^M` under a polynomial map

```
x(ξ) = [ value[0](ξ), …, value[M-1](ξ) ]
```

where each `value[i]` is a `TaylorExpansion<T, IsotropicScheme<N, M>, Storage>`
— a polynomial of degree `≤ N` in the `M` factor variables. `Box` and
`Zonotope` are the degree-1 special cases (diagonal and dense linear generators,
respectively).

---

## Tiered Domain interface

The module exposes two concept tiers (`include/tax/ads/domains/domain.hpp`):

| Concept | Required operations | Models |
|---------|---------------------|--------|
| `Domain` | `center(dim)`, `split(dim)`, `domain_traits<D>::scalar`, `domain_traits<D>::dim` | `Box`, `Zonotope`, `PolynomialZonotope` |
| `LocatableDomain` | `Domain` + `splitOrdinate(dim)` (exact affine inversion) | `Box`, `Zonotope` |

`PolynomialZonotope` models `Domain` but **not** `LocatableDomain`:

- **`propagate` / `AdsDriver`** require only `Domain` — they drive any of the
  three primitives without change.
- **`merge`** requires `LocatableDomain` (needs `splitOrdinate` for sibling
  ordering) and is disabled for `PolynomialZonotope` at compile time.
- **`contains`** on a `PolynomialZonotope` is a **conservative
  over-approximation** (a bounding-box test on each component's range). It may
  return `true` for a point slightly outside the curved set. For exact membership
  you need a `Box` or `Zonotope`.

---

## Usage

```cpp
#include <tax/ads.hpp>
#include <tax/ode.hpp>
using namespace tax::ode::methods;

// Build a degree-N poly zonotope from an axis-aligned box of ICs:
tax::ads::Box<double, 2> ic_box{ center_vec, half_width_vec };
auto ic = tax::ads::PolynomialZonotope<double, /*N=*/6, /*M=*/2>::fromBox(ic_box);
// (or construct ic.value directly from any polynomial expressions over ξ)

auto sol = tax::ads::propagate</*P=*/6>(
    Verner89{}, tax::ads::TruncationCriterion{1e-4, 8},
    rhs, ic, ic_center, 0.0, 2 * M_PI, cfg);

const auto& tree = sol.tree();
for (int i : tree.done()) {
    const auto& leaf = tree.leaf(i);
    // leaf.box is the leaf's PolynomialZonotope;
    // evaluate leaf.payload(ξ) for a factor vector ξ ∈ [-1, 1]^M.
}
```

---

## When to use

Use `PolynomialZonotope` when the true IC set is intrinsically curved (e.g. it
was itself the image of a prior polynomial map) and you want to propagate without
projecting back to a box first. The polynomial-image representation lets the
leaf's payload and IC set stay in the same coordinate frame.

If you only need a *rotated* linear IC set (an ellipsoid, a correlation
structure), prefer `Zonotope` — it is lighter, admits `merge`, and supports exact
point location.
