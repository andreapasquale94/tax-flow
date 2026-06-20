# ODE API Reference

All names live in `namespace tax::ode` unless noted. The whole surface is
brought in by the umbrella header:

```cpp
#include <tax/ode.hpp>
```

---

## Configuration

```cpp
template <class T = double>
struct IntegratorConfig {
    T    abstol               = T{1e-12};   // absolute local-error tolerance
    T    reltol               = T{1e-12};   // relative local-error tolerance
    T    initial_step         = T{0};       // 0 ⇒ stepper picks an initial guess
    T    min_step             = T{0};       // 0 ⇒ no lower bound (≈ ε·|tmax − t0|)
    T    max_step             = T{0};       // 0 ⇒ tmax − t0
    int  max_steps            = 100'000;
    int  max_rejects_per_step = 16;
    bool save_steps           = true;       // false ⇒ keep only initial + final state
};
```

When `save_steps == true` (the default) every accepted step boundary is appended
to `sol.t` / `sol.x`. When `false`, only the initial and final states are kept —
useful when only the endpoint or event records are needed and memory is a concern.
Events are **always** recorded in `sol.events` regardless of `save_steps`.

Validated at `Integrator` construction; invalid values throw
`std::invalid_argument`.

---

## Stepper concepts

```cpp
namespace tax::ode::concepts {

template <class S>
concept Stepper = requires(S s, typename S::Rhs f,
                           typename S::State x, typename S::T t,
                           typename S::T h, const typename S::Config& cfg) {
    typename S::State;
    typename S::T;
    typename S::Config;
    typename S::Rhs;
    typename S::StepData;

    { s.step(f, x, t, h, cfg) } -> std::same_as<StepResult<typename S::State, S>>;
};

template <class S>
concept AdaptiveStepper = Stepper<S>
    && requires { { S::is_adaptive } -> std::convertible_to<bool>; }
    && S::is_adaptive;

}  // namespace tax::ode::concepts
```

Steppers additionally expose `static constexpr bool has_step_expansion`:

- **`true`** (Taylor): `StepData` is `tax::la::VecNT<D, TE>` — the per-step time-Taylor
  expansion of the solution about the step start. The integrator evaluates it via
  `tax::la::eval(data, τ)` to obtain the state at any `t_old + τ` within the step.
- **`false`** (RK): `StepData` is empty. For event location the integrator performs
  a controller-free full-order re-step of size `τ` to query the state at `t_old + τ`.

---

## Step result

```cpp
template <class State, class Stepper>
struct StepResult {
    State                        x_new;
    typename Stepper::T          h_used;
    typename Stepper::StepData   data;     // per-step expansion (Taylor) or empty (RK)
    typename Stepper::T          h_next;
    typename Stepper::T          err_norm;
    bool                         accepted;
};
```

The adaptive fields (`h_next`, `err_norm`, `accepted`) are meaningful when
`AdaptiveStepper<Stepper>` holds; the `Integrator` drives the
rejection-and-retry loop from them.

---

## Step-size controllers

`namespace tax::ode::controllers`. Each `next_step` overload returns the
recommended step from the previous step and the latest error norm.

```cpp
template <class T = double>
struct I {
    T safety = 0.9, min_factor = 0.2, max_factor = 5.0;
    [[nodiscard]] T next_step(T h_used, T err_norm, T tol, int p_emb) const noexcept;
};

template <class T = double>
struct PI {
    T safety = 0.9, alpha = 0.7, beta = 0.4, min_factor = 0.2, max_factor = 5.0;
    [[nodiscard]] T next_step(T h_used, T err_norm, T tol, int p_emb) noexcept;
};

template <class T = double>
struct H211b {
    T safety = 0.9, b = 4.0, min_factor = 0.2, max_factor = 5.0;
    [[nodiscard]] T next_step(T h_used, T err_norm, T tol, int p_emb) noexcept;
};

// Taylor-specific (compile-error for non-Taylor steppers)
template <class T = double>
struct JorbaZou {
    T safety = 0.9, min_factor = 0.2, max_factor = 5.0;
    [[nodiscard]] T next_step(T h_used, T c_N_norm, T c_Nm1_norm,
                              T tol, int N_order) const noexcept;
};
```

PI and H211b own a mutable previous-error state; `step()` is therefore
non-`const` on every stepper for uniformity.

---

## Steppers

All steppers share the surface

```cpp
struct Stepper {
    using State, T, Config, Rhs, StepData;
    static constexpr bool is_adaptive;
    static constexpr bool has_step_expansion;  // true: Taylor, false: RK
    static constexpr int  order_v;             // method order p
    static constexpr int  order_emb_v;         // embedded estimator order (RK only)

    StepResult<State, Stepper>
        step(const Rhs& f, const State& x, T t, T h, const Config& cfg);
};
```

### Taylor method

```cpp
template <int N,
          class StateT,
          class Controller = controllers::JorbaZou<typename StateT::Scalar>>
struct TaylorStepper;
```

`N ≥ 2`. `has_step_expansion = true`. `StepData = tax::la::VecNT<D, TE<N>>` —
the per-component time-Taylor expansion of the solution about the step start,
evaluated via `tax::la::eval(data, τ)` to obtain the state at any `t_old + τ`.
This drives accurate event location at full $N$-th order.

### Runge–Kutta methods

All five embedded-pair RK methods are aliases of the single
`detail::EmbeddedRKStepper<Tab, StateT, Controller>` template over their
Butcher tableaus:

```cpp
template <class StateT, class Controller = controllers::PI<double>>
using Verner78Stepper   = /* EmbeddedRKStepper */;  // 8(7), 13 stages

template <class StateT, class Controller = controllers::PI<double>>
using Verner89Stepper   = /* EmbeddedRKStepper */;  // 9(8), 16 stages

template <class StateT, class Controller = controllers::PI<double>>
using Fehlberg78Stepper = /* EmbeddedRKStepper */;  // 7(8), 13 stages

template <class StateT, class Controller = controllers::PI<double>>
using Feagin12Stepper   = /* EmbeddedRKStepper */;  // 12(10), 25 stages

template <class StateT, class Controller = controllers::PI<double>>
using Feagin14Stepper   = /* EmbeddedRKStepper */;  // 14(12), 35 stages
```

`has_step_expansion = false`; `StepData` is empty. For event location the
integrator performs a controller-free full-order re-step of size `τ` to sample
the state, accurate to the full method order.
A zero embedded-error estimate (the "Fehlberg coincidence", Feagin's
stage-difference indicator at small h) is floored at machine eps · tol before
reaching the step controller.

---

## Solution

```cpp
template <class Stepper, class State>
class Solution {
public:
    using T = typename Stepper::T;
    std::vector<T>                       t;       // size = nsteps + 1 (save_steps=true)
                                                  //        or 2       (save_steps=false)
    std::vector<State>                   x;       // x[i] at t[i]
    std::vector<EventRecord<State, T>>   events;  // monotonic in EventRecord::t

    [[nodiscard]] std::size_t size() const noexcept;  // returns t.size()
};

template <class State, class T>
struct EventRecord {
    std::string label;           // "" if anonymous
    T           t;
    State       x;
};
```

`sol.t` and `sol.x` always contain at least the initial state. With
`cfg.save_steps = true` (the default) every accepted step boundary is appended;
with `false` only `(t0, x0)` and the final `(tmax, x_final)` are kept.
`sol.events` is always populated regardless of `save_steps`.

---

## Integrator

```cpp
template <concepts::Stepper Stepper, class F>
class Integrator {
public:
    using State     = typename Stepper::State;
    using T         = typename Stepper::T;
    using Config    = typename Stepper::Config;
    using Solution  = tax::ode::Solution<Stepper, State>;
    using EventList = std::vector<Event<Stepper>>;

    explicit Integrator(F f, Config cfg = {}, EventList events = {});

    [[nodiscard]] Solution integrate(const State& x0, const T& t0, const T& tmax) const;
};
```

`F` is template-deduced so a generic lambda (`[](const auto& x, const auto& t){…}`)
can be reused across the scalar-state RK steppers and the TE-state Taylor stepper.

---

## Per-method type aliases

| Alias                                              | Stepper                  | Default controller              |
| -------------------------------------------------- | ------------------------ | ------------------------------- |
| `Verner78<State, Ctrl=PI, F=Rhs>`                  | `Verner78Stepper`        | `controllers::PI<double>`       |
| `Verner89<State, Ctrl=PI, F=Rhs>`                  | `Verner89Stepper`        | `controllers::PI<double>`       |
| `Fehlberg78<State, Ctrl=PI, F=Rhs>`                | `Fehlberg78Stepper`      | `controllers::PI<double>`       |
| `Feagin12<State, Ctrl=PI, F=Rhs>`                  | `Feagin12Stepper`        | `controllers::PI<double>`       |
| `Feagin14<State, Ctrl=PI, F=Rhs>`                  | `Feagin14Stepper`        | `controllers::PI<double>`       |
| `Taylor<N, State, Ctrl=JorbaZou, F=Rhs>`           | `TaylorStepper<N,…>`     | `controllers::JorbaZou<double>` |

`F` defaults to `Stepper::Rhs` (a `std::function<State(const State&, double)>`).
Spell `F` explicitly to avoid the vtable indirection on benchmark hot loops.

!!! note "`Taylor<…>` requires explicit `F`"
    `TaylorStepper::step()` invokes the RHS with TE-valued state internally, so the
    `std::function<State(const State&, double)>` default does not compile. Spell
    `decltype(f)` as the 4th template parameter:

    ```cpp
    auto f = [](const auto& x, double t) { /*…*/ return x; };
    tax::ode::Taylor<25, Eigen::Matrix<double, 6, 1>,
                     tax::ode::controllers::JorbaZou<double>,
                     decltype(f)> integ{ f, cfg };
    ```

### Examples

```cpp
// Adaptive Verner 8(7) on a 6-state double system:
tax::ode::Verner78< Eigen::Matrix<double, 6, 1> > integ{ f, cfg };

// FixedStep grid (uses cfg.initial_step uniformly):
tax::ode::Verner78< Eigen::Matrix<double, 6, 1>,
                    tax::ode::controllers::FixedStep<double> > integ{ f, cfg };

// DA-vector state (vector of TE in IC deviations):
tax::ode::Verner78< Eigen::Matrix<tax::TEn<2, 4>, 6, 1> > integ_da{ f, cfg };
```

---

## Events

The full event surface — `Direction`, `ControlFlow`, `TriggerContext`,
`StepperCtx`, `Event<Stepper>`, the `EveryStep` / `ZeroCrossing` triggers, and
the `Continue` / `Terminate` / `Record` / `Custom` actions — is covered on its
own page: [Events](events.md).
