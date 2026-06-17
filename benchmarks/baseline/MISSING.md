# Baseline Benchmark Notes

> **NOTE (Stage 1):** The automated perf-gate workflow (`.github/workflows/bench.yml` and
> `.github/scripts/perf_diff.py`) is deferred — the benchmark infrastructure on the
> `stage1-cpp-base` branch is not wired up yet. Performance comparisons against this baseline
> are run manually for now (see `main-5965527-univariate.txt` below). The perf-gate will be
> reinstated once the Stage 1 benchmark suite is complete.

Captured from `main` at SHA `5965527` on 2026-05-19.

## What was captured

- `main-5965527-univariate.txt` — `bench_univariate` (includes DACE comparison, TAX_USE_DACE=ON)

## What is missing

At the time of capture the `benchmarks/` directory only contained `univariate.cpp`.
The `bench_multivariate`, `bench_vs_dace`, and `bench_dynamic_vs_static` executables
mentioned in the plan do not have corresponding source files in this version of `main`.
No fallback was needed — DACE built and ran successfully.
The univariate benchmark already exercises DACE comparisons via `TAX_BENCH_HAVE_DACE=1`.
