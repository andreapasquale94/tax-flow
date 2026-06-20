# tax-flow

[![Tests](https://github.com/andreapasquale94/tax-flow/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/andreapasquale94/tax-flow/actions/workflows/tests.yml)
[![Docs](https://github.com/andreapasquale94/tax-flow/actions/workflows/docs.yml/badge.svg?branch=main)](https://andreapasquale94.github.io/tax-flow/)

**Adaptive ODE integration and Automatic Domain Splitting built on
[`tax`](https://github.com/andreapasquale94/tax).**

`tax-flow` is the higher-level numerics layer that sits on top of the `tax`
truncated-Taylor core. It provides:

- **`tax::ode`** — adaptive Runge–Kutta and Taylor-method ODE integration with an
  event system (triggers/actions), dense output, and step-size controllers.
- **`tax::ads`** — Automatic Domain Splitting (Wittig 2015) and the LOADS variant
  (Losacco/Fossà/Armellin 2024), composed on top of the `tax::ode` event
  infrastructure.

The header include paths (`<tax/ode.hpp>`, `<tax/ads.hpp>`) and namespaces
(`tax::ode`, `tax::ads`) are unchanged from when these modules lived in `tax`.

## Requirements

- C++23 compiler
- CMake ≥ 3.28
- [`tax`](https://github.com/andreapasquale94/tax) (the core library)
- Eigen3 and Threads (pulled in transitively through `tax`)

## Building

`tax-flow` finds `tax` via `find_package(tax)`. If `tax` is not installed, it
falls back to a sibling source checkout (default `../tax`), so cloning both
repositories next to each other builds with no install step:

```
Codes/
├── tax/
└── tax-flow/
```

```bash
# Configure (uses ../tax if tax is not installed; override with -DTAX_SOURCE_DIR=...)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j

# Test
ctest --test-dir build --output-on-failure
```

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `TAXFLOW_BUILD_UNITTESTS` | `ON`  | Build the Google Test unit-test suite |
| `TAXFLOW_BUILD_EXAMPLES`  | `OFF` | Build example programs under `examples/` |
| `TAX_SOURCE_DIR`          | `../tax` | Path to the `tax` source tree (used only when `tax` is not installed) |

## Using the library

```cpp
#include <tax/ode.hpp>
using namespace tax::ode::methods;

auto rhs = [](const auto& x, double) { /* return state-shaped value */ };
tax::la::VecNT<2, double> x0{1.0, 0.0};

auto sol = tax::ode::propagate(Verner89{}, rhs, x0, 0.0, 2 * M_PI);
```

```cpp
#include <tax/ads.hpp>
#include <tax/ode.hpp>
using namespace tax::ode::methods;

tax::ads::Box<double, 2> ic_box{center_vec, half_width_vec};
auto tree = tax::ads::propagate</*P=*/6>(
    Verner89{}, tax::ads::TruncationCriterion{1e-4, 8},
    rhs, ic_box, ic_center, 0.0, 2 * M_PI, cfg);
```

## License

BSD 3-Clause (same as `tax`).
