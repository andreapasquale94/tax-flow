# CLAUDE.md — AI Assistant Guide for `tax-flow`

## Project Overview

**tax-flow** is the higher-level numerics layer built on top of
[`tax`](https://github.com/andreapasquale94/tax), the header-only C++23
Truncated Algebraic eXpansions (TAX) core. It ships two header-only modules:

- **`tax::ode`** — adaptive ODE integration (Runge–Kutta + Taylor methods) with
  an event system, dense output, and step-size controllers.
- **`tax::ads`** — Automatic Domain Splitting (Wittig 2015) and the LOADS
  variant (Losacco/Fossà/Armellin 2024), composed on the `tax::ode` events.

Both keep the original `<tax/ode.hpp>` / `<tax/ads.hpp>` include paths and
`tax::ode` / `tax::ads` namespaces — they were split out of the `tax` repository
unchanged. `ads` depends on `ode`; both depend on the `tax` core.

- **Version:** 0.1.0
- **License:** BSD 3-Clause
- **C++ Standard:** C++23 (required)
- **Build system:** CMake (≥ 3.28)

---

## Relationship to `tax`

`tax-flow` consumes `tax` as an external dependency:

- The build calls `find_package(tax)`. If `tax` is not installed, it falls back
  to a **sibling source checkout** (`TAX_SOURCE_DIR`, default `../tax`) via
  `add_subdirectory`, so cloning both repos next to each other builds with no
  install step.
- `tax-flow` links the `tax::tax` target, which carries Eigen3 and Threads as
  transitive interface dependencies.
- The CMake target is `tax-flow`, exported/aliased as `tax::flow`.

Core types (`TaylorExpansion`, dense/sparse storage, named expansions,
`tax::la` Eigen helpers) live in `tax` — do not reimplement them here. If a
change needs the core (e.g. a new kernel or operator), make it in `tax` first.

---

## Repository Structure

```
tax-flow/
├── include/tax/
│   ├── ode.hpp               # Facade: ODE integration module (tax::ode)
│   ├── ads.hpp               # Facade: Automatic Domain Splitting (tax::ads)
│   ├── ode/                  # Adaptive ODE integration (namespace tax::ode)
│   │   ├── propagate.hpp     #   propagate<Dense>(method, rhs, x0, t0, t1, cfg, events)
│   │   ├── integrator.hpp    #   Integrator<Stepper, F, Dense> driver
│   │   ├── config.hpp        #   IntegratorConfig<T>
│   │   ├── controllers.hpp   #   I, PI, H211b, JorbaZou, FixedStep
│   │   ├── steppers/         #   taylor.hpp + five per-method RK headers
│   │   ├── detail/           #   embedded_rk_stepper + Butcher tableaus, root-finders
│   │   ├── solution.hpp      #   Solution<Stepper, State, Dense> (dense output)
│   │   ├── event.hpp / triggers.hpp / actions.hpp
│   │   ├── vector_ops.hpp    #   VectorOps<S> trait (scalar / TE / Eigen states)
│   │   ├── named.hpp         #   VectorOps for tax::named expansions as ODE state
│   │   └── io.hpp            #   linspace, writeCsv (opt-in)
│   └── ads/                  # Automatic Domain Splitting (namespace tax::ads)
│       ├── box.hpp, leaf.hpp, tree.hpp
│       ├── criteria.hpp      #   SplitCriterion, TruncationCriterion, NliCriterion
│       ├── nonlinearity_index.hpp, split_event.hpp, da_state.hpp
│       ├── driver.hpp, propagate.hpp, merge.hpp
│       ├── refine.hpp, refine_criteria.hpp
│       └── io.hpp            #   CSV writers (opt-in)
├── tests/                    # Google Test suite
│   ├── ode/                  #   steppers/, integrator/, events/, problems/ (CR3BP, Kepler)
│   ├── ads/                  #   box, tree, criteria, driver, merge, parallel, refine
│   └── testUtils.hpp         #   shared helpers/macros
├── benchmarks/               # Google Benchmark suite (bench_ode_cr3bp, bench_ads_refine)
├── examples/                 # two_body/, three_body/, wsb/ — Taylor, ADS, LOADS
│   ├── common/output.hpp     #   shared I/O scaffolding (JSON schema, banners)
│   └── plot/                 #   matplotlib scripts rendering the JSON outputs
├── docs/                     # ode/, ads/, benchmarks/
├── cmake/                    # CMake package config template
├── .clang-format             # Code style (shared with tax)
├── CMakeLists.txt
└── README.md
```

---

## Building

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `TAXFLOW_BUILD_UNITTESTS` | `ON`  | Build Google Test unit-test suite |
| `TAXFLOW_BUILD_BENCHMARK` | `OFF` | Build Google Benchmark suite |
| `TAXFLOW_BUILD_EXAMPLES`  | `OFF` | Build example programs under `examples/` |
| `TAX_SOURCE_DIR`          | `../tax` | `tax` source tree (used only when `tax` is not installed) |

### Dependencies

- **Required:** `tax` (find_package or sibling source), Eigen3, Threads (both via `tax`)
- **Test framework:** Google Test v1.17 — fetched automatically if not found
- **Benchmark framework:** Google Benchmark v1.9 — fetched when `TAXFLOW_BUILD_BENCHMARK=ON`

---

## ODE Integration (`tax::ode`)

```cpp
#include <tax/ode.hpp>
using namespace tax::ode::methods;

auto rhs = [](const auto& x, double) { /* return state-shaped value */ };
tax::la::VecNT<2, double> x0{1.0, 0.0};

auto sol   = tax::ode::propagate(Verner89{}, rhs, x0, 0.0, 2 * M_PI);
auto sol_d = tax::ode::propagate</*Dense=*/true>(Taylor<16>{}, rhs, x0, 0.0, 2 * M_PI);
auto x_at  = sol_d(0.42);   // dense output
```

Methods: `methods::Taylor<N>`, `Verner78`, `Verner89`, `Fehlberg78`,
`Feagin12`, `Feagin14`. All embedded RK methods are aliases of
`detail::EmbeddedRKStepper<Tab, State, Controller>` over their Butcher tableaus
(`detail/*_tableaus.hpp`).

---

## Automatic Domain Splitting (`tax::ads`)

```cpp
#include <tax/ads.hpp>
#include <tax/ode.hpp>
using namespace tax::ode::methods;

tax::ads::Box<double, 2> ic_box{center_vec, half_width_vec};
auto tree = tax::ads::propagate</*P=*/6>(
    Verner89{}, tax::ads::TruncationCriterion{/*tol=*/1e-4, /*maxDepth=*/8},
    rhs, ic_box, ic_center, 0.0, 2 * M_PI, cfg);

for (int i : tree.done()) { const auto& l = tree.leaf(i); /* l.payload */ }
auto stats = tax::ads::merge(tree, tax::ads::TruncationCriterion{1e-4});
```

Architecture: leaf-only arena tree (`AdsTree`); ADS interops with `tax::ode`
events via `(SplitTrigger, SplitAction)`; splits happen at accepted-step
boundaries only; the parallel `AdsDriver` runs `num_threads` jthread workers.

---

## Code Conventions

Same as `tax`: PascalCase types, camelCase functions/methods, snake_case locals,
`lowercase` namespaces (`tax::ode`, `tax::ads`, `tax::ode::detail`,
`tax::ads::detail`). `.clang-format` (Google style, 4-space indent, 100-col) is
shared with `tax`:

```bash
clang-format -i $(git ls-files 'include/**/*.hpp')
```

Concepts over SFINAE (`tax::ode::concepts::Stepper`, `tax::ads::SplitCriterion`);
`[[nodiscard]]` on accessors/results; `std::vector` allowed in these modules
(unlike the allocation-free `tax` dense core).

---

## Testing

Tests are registered via `tax_add_test(name SOURCES path.cpp)` in
`tests/CMakeLists.txt` (ADS) and `tests/ode/CMakeLists.txt` (ODE). Each links
the `tax-flow` interface target and gtest_main.

```bash
ctest --test-dir build --output-on-failure
./build/tests/test_ode_integrator_basic   # single executable
```

### Before Submitting a PR

1. All ctest targets pass locally (against the sibling `tax` or an installed one)
2. Code is formatted with `clang-format`
3. ODE/ADS changes have tests in `tests/ode/` or `tests/ads/`
4. Core-type changes belong in `tax`, not here
