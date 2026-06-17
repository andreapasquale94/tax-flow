# tax-flow

**Adaptive ODE integration and Automatic Domain Splitting built on
[`tax`](https://github.com/andreapasquale94/tax).**

`tax-flow` is the higher-level numerics layer on top of the `tax`
truncated-Taylor core. It provides two header-only modules:

- **ODE Integrator (`tax::ode`)** — adaptive Runge–Kutta and Taylor-method
  integration with an event system (triggers/actions), dense output, and
  step-size controllers.
- **Automatic Domain Splitting (`tax::ads`)** — Wittig 2015 ADS and the LOADS
  variant (Losacco/Fossà/Armellin 2024), composed on top of the `tax::ode`
  event infrastructure.

The include paths (`<tax/ode.hpp>`, `<tax/ads.hpp>`) and namespaces
(`tax::ode`, `tax::ads`) are unchanged from when these modules lived in `tax`.

## Where to start

- **[Tutorials](tutorials/index.md)** — worked two-body and three-body problems,
  plus parallel ADS by refinement.
- **[ODE Integrator](ode/index.md)** — methods, controllers, events, dense output.
- **[Automatic Domain Splitting](ads/index.md)** — boxes, trees, criteria, merge,
  refinement.

## Building

`tax-flow` depends on `tax`. It is found via `find_package(tax)`, falling back to
a sibling source checkout (`../tax`) when `tax` is not installed:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

See the [README](https://github.com/andreapasquale94/tax-flow#readme) for the
full build matrix and CMake options.
