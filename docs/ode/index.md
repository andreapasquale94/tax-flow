# ODE Integrator

The ODE module provides **adaptive initial-value problem solvers** for
arbitrary first-order systems

$$
\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, t), \qquad \mathbf{x}(t_0) = \mathbf{x}_0
$$

over `Eigen::Matrix<T, D, 1>` state, with `D` static or `Eigen::Dynamic`.

A single `Integrator<Stepper>` class drives every method; the **Stepper** is
selected at compile time as a template policy and bundles its step-size
controller as a member.

```cpp
#include <tax/ode.hpp>
#include <Eigen/Core>
using namespace tax::ode::methods;

// dx/dt = x   →   x(t) = exp(t)
auto f = [](const auto& x, auto /*t*/) { return x; };

tax::ode::IntegratorConfig<double> cfg;
cfg.abstol = cfg.reltol = 1e-12;

Eigen::Matrix<double, 1, 1> x0{1.0};
auto sol = tax::ode::propagate(Taylor<16>{}, f, x0, /*t0=*/0.0, /*tmax=*/1.0, cfg);

sol.x.back()(0);                       // ≈ e = 2.7182818…
```

Replace `Taylor<16>{}` with `Verner89{}` or any other method tag to swap
methods. The user-side code is otherwise identical.

---

## Pages in this section

| Page | Topic |
|---|---|
| [Mathematical Foundations](math.md) | Taylor method, Runge–Kutta pairs, step-size controllers, root finding |
| [API Reference](api.md) | Configuration, stepper concepts, integrator, factory functions, solution types |
| [Methods & Benchmarks](methods.md) | When to use each method; CR3BP benchmark distilled |
| [Events](events.md) | Typed `Event<State,T>` objects, built-in events, `StepEvaluator`, user extension |
| [Examples](examples.md) | End-to-end runnable examples — scalar/vector, RK/Taylor, step recording, events |

---

## Available steppers

| Stepper | Order | Embedded order | Stages | Event location | Default controller |
|---|:-:|:-:|:-:|---|---|
| `TaylorStepper<N>`      | $N$ | $N-1$ | $N$ RHS evals | intrinsic per-step expansion (full $N$-th order) | `JorbaZou` |
| `DormandPrince45Stepper`| 5  | 4  | 7  | controller-free full-order re-step | `PI` |
| `Verner67Stepper`       | 7  | 6  | 10 | controller-free full-order re-step | `PI` |
| `Verner78Stepper`       | 8  | 7  | 13 | controller-free full-order re-step | `PI` |
| `Verner89Stepper`       | 9  | 8  | 16 | controller-free full-order re-step | `PI` |
| `Fehlberg78Stepper`     | 7  | 8  | 13 | controller-free full-order re-step | `PI` |
| `Feagin12Stepper`       | 12 | 10 | 25 | controller-free full-order re-step | `PI` |
| `Feagin14Stepper`       | 14 | 12 | 35 | controller-free full-order re-step | `PI` |

All RK steppers can also be instantiated with the I, PI, or H211b controller
via the `Controller` template parameter — see
[API Reference](api.md#step-size-controllers).

---

## Headers

```cpp
#include <tax/ode.hpp>           // umbrella — every public ODE name
```

| Header | Contents |
|---|---|
| `tax/ode/config.hpp`            | `IntegratorConfig<T>` — tolerances, step bounds, iteration limits |
| `tax/ode/concepts.hpp`          | `concepts::Stepper`, `concepts::AdaptiveStepper` |
| `tax/ode/step_result.hpp`       | `StepResult<State, Stepper>` |
| `tax/ode/solution.hpp`          | `Solution<Stepper, State>` |
| `tax/ode/controllers.hpp`       | `I`, `PI`, `H211b`, `JorbaZou` |
| `tax/ode/steppers/taylor.hpp`   | `TaylorStepper<N, State, Controller>` |
| `tax/ode/steppers/dormand_prince45.hpp` | `DormandPrince45Stepper<State, Controller>` |
| `tax/ode/steppers/verner67.hpp` | `Verner67Stepper<State, Controller>` |
| `tax/ode/steppers/verner78.hpp` | `Verner78Stepper<State, Controller>` |
| `tax/ode/steppers/verner89.hpp` | `Verner89Stepper<State, Controller>` |
| `tax/ode/steppers/fehlberg78.hpp` | `Fehlberg78Stepper<State, Controller>` |
| `tax/ode/steppers/feagin12.hpp` | `Feagin12Stepper<State, Controller>` |
| `tax/ode/steppers/feagin14.hpp` | `Feagin14Stepper<State, Controller>` |
| `tax/ode/event.hpp`             | `Event<State,T>`, `BaseEvent`, `Recorder`, `Direction` |
| `tax/ode/step_evaluator.hpp`    | `StepEvaluator<State,T>` — `eval`, `findRoot` |
| `tax/ode/events/step_event.hpp` | `StepEvent<State,T>` |
| `tax/ode/events/root_finding_event.hpp` | `RootFindingEvent<State,T,G>` |
| `tax/ode/events/grid_event.hpp` | `GridEvent<State,T>` |
| `tax/ode/integrator.hpp`        | `Integrator<Stepper,F>`, `addEvent`, factories |
