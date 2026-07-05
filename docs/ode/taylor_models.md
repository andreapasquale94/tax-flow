# Validated integration with Taylor models

*Requires a `tax` core that ships the `tax::model` module (tax PR
"feat(model): Taylor models with rigorous remainder bounds"). All Taylor-model
support in `tax-flow` is compiled only when `<tax/model.hpp>` is available —
against an older core the rest of the library builds unchanged.*

A [Taylor model](https://github.com/andreapasquale94/tax) $T = (P, I)$
encloses a function on a domain: $f(x) \in P(x - x_0) + I$. Where the classic
`TaylorExpansion` pipeline propagates *polynomial approximations* of the flow
map, the Taylor-model pipeline propagates *guaranteed enclosures*: every
accepted step carries a rigorous remainder interval that bounds all truncation
error, so the final state is a mathematically verified bound on the true flow
— for every initial condition in the box at once.

## ODE integration: `methods::Picard`

`TaylorModelStepper` (selected by the `methods::Picard` tag) plugs validated
Taylor-model integration into the ordinary `tax::ode` machinery — the
`Integrator` driver, adaptive step control, events, and `Solution` capture all
work unchanged.

```cpp
#include <tax/domain.hpp>
#include <tax/ode.hpp>

constexpr int P = 7, M = 2, D = 2;
using TM    = tax::model::TaylorModel<double, P, M>;
using State = Eigen::Matrix<TM, D, 1>;

tax::domain::Box<double, M> box{{1.0, 0.5}, {0.1, 0.1}};
Eigen::Vector2d x0{1.0, 0.5};
State s = tax::domain::createModel<P>(box, x0);   // identity models, remainder 0

auto rhs = [](const auto& x, const auto& t) {     // Taylor-model arithmetic!
    std::decay_t<decltype(x)> out{x.size()};
    out(0) = x(1);
    out(1) = -1.0 * x(0);
    return out;
};

auto sol = tax::ode::propagate(tax::ode::methods::Picard{}, rhs, s, 0.0, 1.0);
auto enc = sol.x.back()(0).eval({0.3, -0.5});     // rigorous Interval enclosure
```

Each step:

1. **Lift** the $M$-variate state into $M{+}1$ variables (the extra slot is
   the step time $\tau$ over $[t, t+h]$).
2. **Picard-iterate** $r \leftarrow x_0 + \int_t^\tau f(r)\,d\tau'$ using the
   `tax::model` antiderivation; $N{+}1$ passes fix the polynomial part.
3. **Verify** the enclosure: one more Picard application must map the
   candidate remainder set into itself (Schauder fixed point ⇒ the true flow
   is inside). Failed inclusions are retried with ε-inflation; persistent
   failure rejects the step and halves $h$ — the Makino/Berz scheme.
4. **Fix** $\tau = t + h$ (exact) and project back to the $M$ factor
   variables.

Step control keys on the τ-truncation of the flow polynomial (the masses of
the last two τ-orders, Jorba–Zou style) — the error stepping can influence.
The ξ-truncation of the DA representation is harvested into the remainder at
a rate independent of $h$; ADS *splitting*, not stepping, reduces it.
The RHS is invoked in Taylor-model arithmetic (a generic callable, as with the
`Taylor<N>` stepper); time arrives as an $(M{+}1)$-variate model, so
non-autonomous systems work naturally.

## Domain conversions

`<tax/domain/model.hpp>` bridges the set primitives and Taylor models — all
models live over the normalized cube $\xi \in [-1,1]^M$ (expansion point 0),
matching the rest of the pipeline:

| Function | Meaning |
|----------|---------|
| `createModel<P>(box \| zonotope \| polyZonotope, x0)` | identity Taylor-model state (zero remainder) — mirrors `create` |
| `intervalHull(state[, comps])` | rigorous axis-aligned `Box`: polynomial range bound + remainder per component |
| `zonotopeEnclosure(state, comps)` | `ImageZonotope`: even-exponent shift on the polynomial + one axis-aligned generator per remainder |
| `zonotopeFrame(state[, comps])` | exact degree-1 frame (not an enclosure) |
| `toPolynomialZonotope(state)` | polynomial parts as a `PolynomialZonotope`; **drops remainders** (debug-asserted zero) |

## ADS over Taylor models

`tax::ads::propagate<P>` accepts the `Picard` tag directly — the whole
Automatic Domain Splitting pipeline then runs on validated payloads:

```cpp
auto sol = tax::ads::propagate<P>(
    tax::ode::methods::Picard{},
    tax::ads::TruncationCriterion{1e-4, 8},
    rhs, ic_box, ic_center, 0.0, t1, cfg);

auto enc = sol.evaluate(ic_point);   // vector of rigorous Interval enclosures
```

- **Splits** substitute $\xi_{\text{dim}} \to \pm\tfrac12 + \tfrac12\xi'$ on
  the *polynomial* parts; the remainder, expansion point and domain carry over
  unchanged (the child's cube maps into a subset of the parent's, where the
  parent's bound already holds).
- **Criteria** (`TruncationCriterion`, `NliCriterion`) read the polynomial
  parts; `TruncationCriterion` sums the top **two** degrees (the antiderivation
  harvests the order-$N$ block into the remainder every step, so a propagated
  payload's truncation frontier sits at degree $N{-}1$). The remainder itself
  is validated integration error, not splittable structure — a custom
  criterion may key on `state(i).remainder()` directly.
- **`AdsSolution::evaluate`** (and `Partition::evaluate`) return vectors of
  `tax::model::Interval` enclosures instead of point values.

**Not supported for Taylor-model payloads** (refused at compile time, with the
rationale at the call site): `merge()` — the inverse substitution extrapolates
a child's polynomial onto the full parent domain, where the child's remainder
bound does not hold — and `refine()` (TE-specific).

## Rigor contract

Inherited from `tax::model`: all *interval* computations (range bounds,
remainder propagation, verification) are outward-rounded and therefore
guaranteed enclosures; floating-point round-off of the *polynomial
coefficients* themselves (~1 ulp per operation) is not swept into the
remainder.
