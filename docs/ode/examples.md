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

Eigen::Matrix<double, 1, 1> x0{1.0};
auto sol = tax::ode::propagate(tax::ode::methods::Taylor<16>{}, f, x0, 0.0, 1.0, cfg);

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

Eigen::Matrix<double, 2, 1> x0{1.0, 0.0};
auto sol = tax::ode::propagate(tax::ode::methods::Taylor<12>{}, f, x0, 0.0, M_PI / 2, cfg);

sol.x.back();      // ≈ (0, -1) — quarter period
```

---

## Same problem with a Verner 9(8) stepper

The user code differs only in the method tag.

```cpp
auto sol = tax::ode::propagate(tax::ode::methods::Verner89{}, f, x0, 0.0, M_PI / 2, cfg);
```

Replace `Verner89{}` with `Verner78{}`, `Fehlberg78{}`, `Feagin12{}`, or
`Feagin14{}` to swap RK pairs.

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

State x0(1);
x0(0) = 1.0;
auto sol = tax::ode::propagate(tax::ode::methods::Taylor<14>{}, f, x0, 0.0, 1.0);
```

---

## Step recording

By default (`cfg.save_steps = true`) the solution stores every accepted step
boundary in `sol.t` / `sol.x`. Set `save_steps = false` to keep only the
initial and final states — useful when only the endpoint or event records matter
and memory is a concern.

```cpp
tax::ode::IntegratorConfig<double> cfg;
cfg.save_steps = false;   // keep only (t0, x0) and (tmax, x_final)

auto sol = tax::ode::propagate(tax::ode::methods::Taylor<16>{}, f, x0, 0.0, 1.0, cfg);

sol.x.back()(0);   // final state — same value regardless of save_steps
sol.t.size();      // 2 (initial + final)
```

Events are **always** recorded in `sol.events` regardless of `save_steps`.

To query the state at intermediate times, keep `save_steps = true` and
binary-search the stored grid:

```cpp
cfg.save_steps = true;
auto sol = tax::ode::propagate(tax::ode::methods::Verner89{}, f, x0, 0.0, 1.0, cfg);

// Find the stored step closest to t = 0.37.
auto it = std::lower_bound(sol.t.begin(), sol.t.end(), 0.37);
auto& x_nearest = sol.x[std::distance(sol.t.begin(), it)];
```

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

State x0; x0 << 1.0, 0.0;
auto sol = tax::ode::propagate(tax::ode::methods::Taylor<16>{}, f, x0, 0.0, 5.0, cfg, events);

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
using State = Eigen::Matrix<double, 2, 1>;

tax::ode::Verner89<State, H211b<double>> integ{ f, cfg };
auto sol = integ.integrate(x0, 0.0, M_PI / 2);
```

---

## Where to look next

- [Methods & Benchmarks](methods.md) — when each method earns its keep.
- [Events](events.md) — full Trigger + Action surface, including the `Record`
  and `Custom` actions.
- [API Reference](api.md) — every public name, listed.
