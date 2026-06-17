# ODE Examples

End-to-end examples that exercise the integrator surface. All snippets assume
`#include <tax/ode.hpp>` and `#include <Eigen/Core>`.

---

## Scalar ODE

```cpp
// dx/dt = x  →  x(t) = exp(t)
auto f = [](const auto& x, auto /*t*/) { return x; };

tax::ode::IntegratorConfig<double> cfg;
cfg.abstol = cfg.reltol = 1e-12;

auto integ = tax::ode::makeTaylorIntegrator<16, double, 1>(f, cfg);

Eigen::Matrix<double, 1, 1> x0{1.0};
auto sol = integ.integrate(x0, /*t0=*/0.0, /*tmax=*/1.0);

sol.x.back()(0);   // ≈ e = 2.7182818…
```

---

## Vector ODE — harmonic oscillator

```cpp
// dx/dt = v, dv/dt = -x
auto f = [](const auto& x, auto /*t*/) {
    using S = std::decay_t<decltype(x)>;
    S out;
    out(0) =  x(1);
    out(1) = -x(0);
    return out;
};

tax::ode::IntegratorConfig<double> cfg;
cfg.abstol = cfg.reltol = 1e-12;

auto integ = tax::ode::makeTaylorIntegrator<12, double, 2>(f, cfg);

Eigen::Matrix<double, 2, 1> x0{1.0, 0.0};
auto sol = integ.integrate(x0, 0.0, M_PI / 2);

sol.x.back();      // ≈ (0, -1) — quarter period
```

---

## Same problem with a Verner 9(8) stepper

The user code differs only in the factory call.

```cpp
auto integ = tax::ode::makeVerner89Integrator<double, 2>(f, cfg);
auto sol   = integ.integrate(x0, 0.0, M_PI / 2);
```

Replace `makeVerner89Integrator` with `makeVerner78Integrator`,
`makeFehlberg78Integrator`, `makeFeagin12Integrator`, or
`makeFeagin14Integrator` to swap RK pairs.

---

## Dynamic-dimension state

```cpp
// dx/dt = -x³ on a runtime-sized 1-vector
using State = Eigen::VectorXd;

const auto f = [](const auto& x, auto /*t*/) {
    using S = std::decay_t<decltype(x)>;
    S out{x.size()};
    out(0) = -x(0) * x(0) * x(0);
    return out;
};

auto integ = tax::ode::makeTaylorIntegrator<14>(f);
State x0(1);
x0(0) = 1.0;

auto sol = integ.integrate(x0, 0.0, 1.0);
```

`makeTaylorIntegrator<N>` (no `T`, `D`, `Dense` overrides) defaults to
`T = double`, `D = Eigen::Dynamic`, `Dense = false`.

---

## Dense output

Setting `Dense = true` opts into per-step continuous-extension storage and
unlocks `sol(t)` for any `t ∈ [t0, tmax]`.

```cpp
auto integ = tax::ode::makeTaylorIntegrator<16, double, 1, /*Dense=*/true>(f, cfg);
auto sol = integ.integrate(x0, 0.0, 1.0);

double xq = sol(0.37)(0);   // value at t = 0.37 via Horner on the per-step TE
```

For RK steppers, `sol(t)` falls back to cubic-Hermite — third-order
accurate, fine for plotting and event location but not the method's full
order.

---

## Events — terminate on a level set

```cpp
using State = Eigen::Matrix<double, 2, 1>;
using Stepper = tax::ode::TaylorStepper<16, State>;

const auto f = [](const auto& x, auto /*t*/) {
    using S = std::decay_t<decltype(x)>;
    S out;
    out(0) =  x(1);
    out(1) = -x(0);
    return out;
};

tax::ode::IntegratorConfig<double> cfg;
cfg.abstol = cfg.reltol = 1e-12;

std::vector<tax::ode::Event<Stepper>> events;
events.emplace_back(
    tax::ode::ZeroCrossing(
        [](const auto& x, const auto&) { return x(0); },
        tax::ode::Direction::Decreasing),
    tax::ode::Terminate());

auto integ = tax::ode::makeTaylorIntegrator<16, double, 2, /*Dense=*/false>(
    f, cfg, events);
State x0; x0 << 1.0, 0.0;
auto sol = integ.integrate(x0, 0.0, 5.0);

sol.t.back();   // ≈ π/2, the first downward crossing of x(0) = cos t
```

For a complete tour of triggers and actions, see [Events](events.md).

---

## Choosing a controller explicitly

The default controllers (`JorbaZou` on Taylor, `PI` on RK) are usually right.
To pin a different one, pass it as the `Controller` template parameter on the
factory:

```cpp
using tax::ode::controllers::H211b;

auto integ = tax::ode::makeVerner89Integrator<
    double, /*D=*/2, /*Dense=*/false, H211b<double>>(f, cfg);
```

---

## Where to look next

- [Methods & Benchmarks](methods.md) — when each method earns its keep.
- [Events](events.md) — full Trigger + Action surface, including the `Record`
  and `Custom` actions.
- [API Reference](api.md) — every public name, listed.
