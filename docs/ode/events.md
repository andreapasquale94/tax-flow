# Events

Events in `tax::ode` are expressed as a **Trigger + Action** pair. The trigger
decides *when* an event fires (returning the local time $\tau \in [0, h]$
within the most recent step) and the action decides *what to do* at that time.
The factoring keeps `record an apoapsis` and `terminate at the moon's surface`
on the same machinery without re-architecting.

```cpp
using tax::ode::Direction;
using tax::ode::Event;
using tax::ode::ZeroCrossing;
using tax::ode::Terminate;
using namespace tax::ode::methods;

using State = Eigen::Matrix<double, 2, 1>;
using Stepper = tax::ode::TaylorStepper<16, State>;
std::vector<Event<Stepper>> events;

events.emplace_back(
    ZeroCrossing([](const auto& x, const auto& /*t*/) { return x(0); },
                 Direction::Decreasing),
    Terminate());

tax::ode::IntegratorConfig<double> cfg;
auto sol = tax::ode::propagate(Taylor<16>{}, f, x0, 0.0, 5.0, cfg, events);
// sol.t.back() = the time at which x(0) crossed 0 downward.
```

---

## Triggers

A trigger has signature

```cpp
std::optional<T>(const StepperCtx<…>&)
```

returning the $\tau \in [0, h_{\text{used}}]$ at which the event fires, or
`std::nullopt` to indicate "did not fire on this step".

### `EveryStep()`

Fires at every accepted step boundary with `τ = h_used`. The seam through
which custom per-step monitors and the (deferred) ADS driver plug in.

```cpp
events.emplace_back(EveryStep(), Custom(my_per_step_callback));
```

### `ZeroCrossing(g, dir)`

Locates a sign change of the scalar user function

```cpp
T g(const State& x, T t);
```

inside the most recent step. `dir` is one of:

| `Direction` | Fires when |
|---|---|
| `Increasing` | $g$ goes from negative to positive |
| `Decreasing` | $g$ goes from positive to negative |
| `Any`        | either of the above |

**Root finding strategy** (see [Mathematical Foundations](math.md#event-location)):

After confirming a sign change between step boundaries, `ZeroCrossing` runs
Brent's method on scalar samples drawn from a `flow(τ)` callable provided by
the integrator. `flow(τ)` returns the state at `t_old + τ` via:

- **Taylor** (`has_step_expansion = true`): evaluates the intrinsic per-step
  time-Taylor expansion at full $N$-th order — `tax::la::eval(data, τ)`.
- **RK** (`has_step_expansion = false`): performs a controller-free full-order
  re-step of size `τ` — accurate to the method order, not a low-order
  approximation.

**Multi-root caveat.** Both paths are *single-root, bracketed*: they find the
earliest zero between step boundaries when the boundary signs disagree, and
silently miss any inner zeros if `g` oscillates within a single accepted step.
Set `Config::max_step` consistently with the expected frequency of `g` to
avoid this.

---

## Actions

An action runs after a trigger fires with the located `τ` and receives an
`EventStorage` to optionally push records into the `Solution`. Signature:

```cpp
ControlFlow(const StepperCtx<…>&, T tau_fired, EventStorage<State, T>&)
```

| Factory | Behavior |
|---|---|
| `Continue()`       | No-op; integration proceeds. |
| `Terminate()`      | Integration loop exits cleanly after the current step. |
| `Record(label)`    | Push `EventRecord{label, t, x}` into the solution's `events` vector, then continue. |
| `Custom(fn)`       | Invoke a user lambda `fn(ctx, τ, storage) -> ControlFlow`. The storage exposes `push(EventRecord)` for writes. |

The `Record` action evaluates `ctx.flow(τ)` to obtain `x` at the located τ,
using the same integrator-provided `flow` callable as `ZeroCrossing` — accurate
to the method order for both Taylor and RK steppers.

---

## Control flow & ordering

If multiple events fire in the same step, the integrator orders them by
`τ_fired` ascending so the recorded time stream is monotonic. If *any* action
returns `ControlFlow::Terminate`, the integration loop exits after that step.

Captures by reference inside trigger or action lambdas must outlive the call
to `Integrator::integrate` — `Event<Stepper>` type-erases the callables, and a
dangling reference at trigger-evaluation time is undefined behaviour.

---

## Worked examples

### Record both apoapsis and periapsis of a harmonic oscillator

```cpp
const auto f = [](const auto& x, const auto&) {
    using S = std::decay_t<decltype(x)>;
    S out;
    out(0) =  x(1);
    out(1) = -x(0);
    return out;
};

using Stepper = tax::ode::TaylorStepper<16, Eigen::Matrix<double, 2, 1>>;
std::vector<tax::ode::Event<Stepper>> events;

// v = x(1) goes through 0 from above (apoapsis), and from below (periapsis).
events.emplace_back(
    tax::ode::ZeroCrossing(
        [](const auto& x, const auto&) { return x(1); },
        tax::ode::Direction::Decreasing),
    tax::ode::Record("apoapsis"));

events.emplace_back(
    tax::ode::ZeroCrossing(
        [](const auto& x, const auto&) { return x(1); },
        tax::ode::Direction::Increasing),
    tax::ode::Record("periapsis"));

Eigen::Matrix<double, 2, 1> x0; x0 << 1.0, 0.0;
auto sol = tax::ode::propagate(tax::ode::methods::Taylor<16>{}, f, x0,
                               0.0, 4.0 * M_PI, cfg, events);

for (const auto& e : sol.events) {
    std::cout << e.label << " at t = " << e.t << "\n";
}
```

### Same on a Verner 8(7) integrator (Brent root finder, otherwise identical)

```cpp
// Events must be typed for the matching stepper.
using State78 = Eigen::Matrix<double, 2, 1>;
using Stepper78 = tax::ode::Verner78Stepper<State78>;
std::vector<tax::ode::Event<Stepper78>> events78;
events78.emplace_back(
    tax::ode::ZeroCrossing([](const auto& x, const auto&) { return x(1); },
                           tax::ode::Direction::Decreasing),
    tax::ode::Record("apoapsis"));
events78.emplace_back(
    tax::ode::ZeroCrossing([](const auto& x, const auto&) { return x(1); },
                           tax::ode::Direction::Increasing),
    tax::ode::Record("periapsis"));

auto sol = tax::ode::propagate(tax::ode::methods::Verner78{}, f, x0,
                               0.0, 4.0 * M_PI, cfg, events78);
```

The user-facing event machinery is identical — `ZeroCrossing` routes through
Brent on `flow(τ)` for all steppers, Taylor and RK alike.
