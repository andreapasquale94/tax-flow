# Events

Events in `tax::ode` are **first-class typed objects**. Each event subclasses
`tax::ode::Event<State,T>`, implements `name()` and `onStep()`, and is
registered on an `Integrator` (or passed to `propagate`). The integrator calls
every event after each accepted step, in registration order.

```cpp
#include <tax/ode.hpp>
using namespace tax::ode::methods;

using State = Eigen::Matrix<double, 2, 1>;

auto rhs = [](const auto& x, auto /*t*/) {
    using S = std::decay_t<decltype(x)>;
    S out;
    out(0) =  x(1);
    out(1) = -x(0);
    return out;
};

tax::ode::IntegratorConfig<double> cfg;
cfg.abstol = cfg.reltol = 1e-12;

// Register a terminal zero-crossing event via the factory.
tax::ode::Integrator<tax::ode::TaylorStepper<16, State>, decltype(rhs)>
    integ{ rhs, cfg };

integ.addRootFindingEvent(
    [](const auto& x, auto /*t*/) { return x(0); },
    tax::ode::Direction::Decreasing, "zero_crossing", /*terminal=*/true);

State x0; x0 << 1.0, 0.0;
auto sol = integ.integrate(x0, 0.0, 5.0);
// sol.t.back() ≈ π/2 — the first downward crossing of x(0) = cos(t)
```

Alternatively, build a `shared_ptr<Event<State,T>>` and pass it to `propagate`
(see the [worked examples](#worked-examples) below for the full pattern).

---

## The `Event<State,T>` interface

```cpp
namespace tax::ode {

template <class State, class T>
class Event {
 public:
    enum class Action { Continue, Terminate };

    virtual ~Event() = default;

    // Label used for every EventRecord this event writes.
    [[nodiscard]] virtual std::string name() const = 0;

    // Pre-step hook: return the next absolute time the integrator must
    // land on exactly. nullopt = no constraint. Default: no constraint.
    [[nodiscard]] virtual std::optional<T> nextStop(T t) const { return std::nullopt; }

    // Post-step hook: called once per accepted step.
    // Read the step via eval_; push records via rec.
    virtual Action onStep(Recorder<State,T>& rec) = 0;

    // Deep copy — required by the parallel ADS driver.
    [[nodiscard]] virtual std::shared_ptr<Event> clone() const = 0;

    void setEvaluator(const std::shared_ptr<const StepEvaluator<State,T>>& e);

 protected:
    std::shared_ptr<const StepEvaluator<State,T>> eval_;
};

} // namespace tax::ode
```

Key points:

- `name()` is the string label stored in every `EventRecord` the event emits.
- `onStep(rec)` runs once per **accepted step**. Return `Action::Continue` to
  let integration proceed; return `Action::Terminate` to stop the loop after
  this step.
- `nextStop(t)` is a **pre-step** hook. If it returns a value, the integrator
  caps the next step so it lands exactly on that time. `GridEvent` uses this to
  force the integrator onto a schedule of times.
- `eval_` is the protected handle to the `StepEvaluator` — the extension seam
  that gives events access to the in-step trajectory. It is bound automatically
  when the event is registered via `addEvent`.
- `clone()` produces a deep copy. This is required by the parallel ADS driver,
  which copies the event list per leaf. Use `BaseEvent` (below) to get it
  for free.

---

## `Recorder<State,T>`

The write handle passed to `onStep`. All event output flows through it:

```cpp
template <class State, class T>
class Recorder {
 public:
    // Append an EventRecord to sol.events.
    void record(std::string name, T t, State x);

    // Number of records accumulated so far.
    [[nodiscard]] std::size_t count() const noexcept;
};
```

`record(name, t, x)` pushes an `EventRecord<State,T>{name, t, x}` into the
`Solution::events` vector. You can call it multiple times in one `onStep` (e.g.
`GridEvent` may fire multiple times if the step covers more than one grid point).

---

## `StepEvaluator<State,T>` — the extension seam

Events read the accepted step through the `eval_` member:

```cpp
template <class State, class T>
class StepEvaluator {
 public:
    [[nodiscard]] virtual const State& xOld() const noexcept = 0;
    [[nodiscard]] virtual const State& xNew() const noexcept = 0;
    [[nodiscard]] virtual T            tOld() const noexcept = 0;
    [[nodiscard]] virtual T            hUsed() const noexcept = 0;

    // State at t_old + τ (τ ∈ [0, h_used]) at the method's full order.
    [[nodiscard]] virtual State eval(T tau) const = 0;

    // First root of g(x(τ), t_old+τ) in (0, h_used] matching dir, or nullopt.
    template <class G>
    [[nodiscard]] std::optional<T> findRoot(G&& g, Direction dir) const;
};
```

- `xOld()` / `xNew()` — start and end of the accepted step.
- `tOld()` / `hUsed()` — start time and step length.
- `eval(τ)` — state at `t_old + τ` at full method order:
  - **Taylor** steppers evaluate the intrinsic per-step polynomial via
    `tax::la::eval(data, τ)`.
  - **RK** steppers perform a controller-free full-order re-step of size `τ`.
- `findRoot(g, dir)` — Brent's method on `φ(τ) = g(eval(τ), tOld()+τ)` in
  `(0, h_used]`, returning the first root matching `dir`, or `nullopt`. See
  [Mathematical Foundations](math.md#event-location) for the algorithm.

`eval_` is set by the integrator before any `onStep` call; it is always valid
inside `onStep` and `nextStop`.

---

## Built-in events

### `StepEvent` — record every accepted step

Fires at every accepted step boundary and records the step-end state:

```cpp
// Via Integrator factory:
integ.addStepEvent("step");

// Or directly:
auto ev = std::make_shared<tax::ode::StepEvent<State,T>>("step");
integ.addEvent(ev);
```

Records `{name, t_old + h_used, x_new}` on every accepted step. The raw step
grid also appears in `sol.t` / `sol.x` when `cfg.save_steps = true`; `StepEvent`
provides the same data as a named event stream.

---

### `RootFindingEvent` — locate a zero crossing

Finds the first root of a scalar function `g(x, t)` inside the step via
`findRoot`, records the located state, and optionally terminates:

```cpp
// Via Integrator factory (preferred):
integ.addRootFindingEvent(
    [](const auto& x, auto t) { return x(0); },  // g
    tax::ode::Direction::Decreasing,
    "zero_crossing",
    /*terminal=*/false);                          // continue after recording

// Or directly (template argument G must match):
auto ev = std::make_shared<
    tax::ode::RootFindingEvent<State, T, decltype(g)>>(g, dir, "label", terminal);
integ.addEvent(ev);
```

`Direction` is one of:

| `Direction` | Fires when |
|---|---|
| `Increasing` | `g` goes from negative to positive |
| `Decreasing` | `g` goes from positive to negative |
| `Any`        | either of the above |

When a root is found the event calls `rec.record(name, t_old+τ, eval(τ))`.
If `terminal = true` it then returns `Action::Terminate`; otherwise
`Action::Continue`.

**Multi-root caveat.** `findRoot` is a single-root Brent search on the
bracketed interval. It finds the first zero when endpoint signs disagree and
silently misses inner zeros when `g` oscillates within one accepted step. Set
`cfg.max_step` to be small relative to the expected frequency of `g`.

---

### `GridEvent` — stop exactly on a list of times

Forces the integrator to land exactly on each time in a sorted list, then
records the state at each one:

```cpp
// Via Integrator factory:
integ.addGridEvent({0.25, 0.5, 0.75, 1.0}, "grid");

// Or directly:
auto ev = std::make_shared<tax::ode::GridEvent<State,T>>(
    std::vector<T>{0.25, 0.5, 0.75, 1.0}, "grid");
integ.addEvent(ev);
```

`nextStop(t)` returns the next grid time after `t`, clamping the step.
`onStep` records the state once the integrator lands on each grid point.
`GridEvent` never returns `Action::Terminate`.

---

## Registering events

### On `Integrator` directly

```cpp
// Low-level: add any Event<State,T> shared_ptr.
integ.addEvent(std::make_shared<MyEvent>(...));

// Factories (type-safe convenience):
integ.addStepEvent("label");
integ.addRootFindingEvent(g, dir, "label", terminal);
integ.addGridEvent(times, "label");
```

### Via `propagate`

```cpp
std::vector<std::shared_ptr<tax::ode::Event<State,T>>> events;
events.push_back(std::make_shared<tax::ode::StepEvent<State,T>>("step"));
events.push_back(std::make_shared<
    tax::ode::RootFindingEvent<State,T,decltype(g)>>(g, dir, "crossing"));

auto sol = tax::ode::propagate(Verner89{}, rhs, x0, t0, t1, cfg, events);
```

---

## Control flow and ordering

Events are called in **registration order** at every accepted step. The first
event that returns `Action::Terminate` stops the rest; integration exits after
that step.

**Termination truncation rule:** when a terminating event also called
`rec.record(...)` in the same `onStep`, the solution truncates at the recorded
point (the last record pushed by that event). If no record was pushed, the
solution truncates at the step boundary `(t_old + h_used, x_new)`.

---

## User-defined events

Subclass `tax::ode::BaseEvent<Derived, State, T>` (the CRTP helper that
provides `clone()` via the copy constructor) and implement `name()` and
`onStep()`. Example — a simple stop-after-time event:

```cpp
template <class State, class T>
class StopAfter final
    : public tax::ode::BaseEvent<StopAfter<State,T>, State, T> {
 public:
    using Action = typename tax::ode::Event<State,T>::Action;

    explicit StopAfter(T t_stop) : t_stop_(t_stop) {}

    std::string name() const override { return "stop_after"; }

    Action onStep(tax::ode::Recorder<State,T>& /*rec*/) override {
        const T t_new = this->eval_->tOld() + this->eval_->hUsed();
        return t_new > t_stop_ ? Action::Terminate : Action::Continue;
    }

 private:
    T t_stop_;
};

// Usage:
auto ev = std::make_shared<StopAfter<State,double>>(3.14);
integ.addEvent(ev);
```

`this->eval_` is always valid inside `onStep` — the integrator sets it before
calling `onStep`. The CRTP base provides `clone()` via `Derived`'s copy
constructor, which satisfies the parallel ADS driver's per-leaf deep-copy
requirement.

---

## Worked examples

### Record both apoapsis and periapsis of a harmonic oscillator

```cpp
const auto f = [](const auto& x, auto /*t*/) {
    using S = std::decay_t<decltype(x)>;
    S out;
    out(0) =  x(1);
    out(1) = -x(0);
    return out;
};

using State = Eigen::Matrix<double, 2, 1>;

tax::ode::IntegratorConfig<double> cfg;
cfg.abstol = cfg.reltol = 1e-12;

// Build integrator and register two RootFindingEvents.
tax::ode::Integrator<tax::ode::TaylorStepper<16, State>, decltype(f)> integ{ f, cfg };

// v = x(1) goes through 0 from above (apoapsis), from below (periapsis).
integ.addRootFindingEvent(
    [](const auto& x, auto /*t*/) { return x(1); },
    tax::ode::Direction::Decreasing, "apoapsis");

integ.addRootFindingEvent(
    [](const auto& x, auto /*t*/) { return x(1); },
    tax::ode::Direction::Increasing, "periapsis");

State x0; x0 << 1.0, 0.0;
auto sol = integ.integrate(x0, 0.0, 4.0 * M_PI);

for (const auto& e : sol.events)
    std::cout << e.label << " at t = " << e.t << "\n";
```

### Same via `propagate` with a pre-built event list

```cpp
using State = Eigen::Matrix<double, 2, 1>;

// Store lambdas so decltype works for the template argument.
auto g_apo  = [](const auto& x, auto /*t*/) { return x(1); };
auto g_peri = [](const auto& x, auto /*t*/) { return x(1); };

std::vector<std::shared_ptr<tax::ode::Event<State,double>>> events;
events.push_back(std::make_shared<
    tax::ode::RootFindingEvent<State, double, decltype(g_apo)>>(
        g_apo, tax::ode::Direction::Decreasing, "apoapsis"));
events.push_back(std::make_shared<
    tax::ode::RootFindingEvent<State, double, decltype(g_peri)>>(
        g_peri, tax::ode::Direction::Increasing, "periapsis"));

State x0; x0 << 1.0, 0.0;
auto sol = tax::ode::propagate(tax::ode::methods::Taylor<16>{}, f, x0,
                               0.0, 4.0 * M_PI, cfg, std::move(events));

for (const auto& e : sol.events)
    std::cout << e.label << " at t = " << e.t << "\n";
```

### Terminate on a level set

```cpp
using State = Eigen::Matrix<double, 2, 1>;

// f as defined in the first example above.
tax::ode::IntegratorConfig<double> cfg;
cfg.abstol = cfg.reltol = 1e-12;

tax::ode::Integrator<tax::ode::TaylorStepper<16, State>, decltype(f)> integ{ f, cfg };
integ.addRootFindingEvent(
    [](const auto& x, auto /*t*/) { return x(0); },
    tax::ode::Direction::Decreasing, "zero_crossing", /*terminal=*/true);

State x0; x0 << 1.0, 0.0;
auto sol = integ.integrate(x0, 0.0, 5.0);
// sol.t.back() ≈ π/2 — integration stopped at the first downward crossing.
```

---

## ADS and `SplitEvent`

The ADS module wires splitting into the `tax::ode` event system via
`tax::ads::SplitEvent` — a `BaseEvent` that checks the split criterion at
each step boundary and returns `Action::Terminate` when a split is due, writing
a `SplitRequest` for the driver to consume. Users do not need to construct
`SplitEvent` directly; it is created and registered by `AdsDriver` internally.
