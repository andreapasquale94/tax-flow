# Methods & Benchmarks

A practical guide to choosing among the six shipped steppers and the four
controllers.

---

## At a glance

| Method | Strength | Pay attention to | Default controller |
|---|---|---|---|
| **Verner 8(7)**   | Best speed/accuracy balance in the RK family for smooth problems at $10^{-12}$ | Cubic-Hermite dense output | `PI` |
| **Verner 9(8)**   | Precision champion at moderate cost | 16 stages — more RHS evals per step | `PI` |
| **Fehlberg 7(8)** | Classical baseline, well understood | The "Fehlberg coincidence" can zero the embedded estimate on certain steps | `PI` |
| **Feagin 12(10)** | Very high order for smooth astrodynamics | Sparse error indicator — only stages 2 and $n-1$ contribute | `PI` |
| **Feagin 14(12)** | Highest order available | Even more sparse error contribution; expensive | `PI` |
| **Taylor (`N`)**  | Exact polynomial-Newton event location, exact dense output | Per-step cost grows roughly $\mathcal{O}(N^2)$; 30–100× slower than RK on smooth astro problems | `JorbaZou` |

---

## Picking a controller

| Controller | When |
|---|---|
| **PI** *(default for RK)* | Smooth problems. Best on most non-stiff astrodynamics. |
| **I** | Robust baseline, fewer assumptions; more rejections. Pair with Fehlberg78 for a classical RKF reference. |
| **H211b** | Bumpy / piecewise-smooth right-hand sides. Smoothed step sequence. **Do not** use with Taylor — it damps the natural high-order predictor. |
| **JorbaZou** *(Taylor only)* | The right choice for `TaylorStepper`. Reads the last two polynomial coefficient norms; compile-error on non-Taylor steppers. |

You can change a stepper's controller by instantiating it with the second
template parameter:

```cpp
using StateT  = Eigen::Matrix<double, 2, 1>;
using MyV89   = tax::ode::Verner89Stepper<StateT, tax::ode::controllers::H211b<double>>;

auto integ = tax::ode::Integrator<MyV89, decltype(f)>{ f, cfg };
```

Or spell the controller as the second type parameter of the alias:

```cpp
tax::ode::Verner89< Eigen::Matrix<double, 2, 1>,
                    tax::ode::controllers::H211b<double> > integ{ f, cfg };
```

---

## CR3BP benchmark

The planar circular restricted three-body problem (Earth–Moon $\mu = 0.01215$)
is a stable proxy for smooth astrodynamics propagation. The benchmark suite
(`benchmarks/bench_ode_cr3bp`) propagates a transit trajectory through L1,
past the Moon, and out via L2 over $t \in [0, 7]$ with
`abstol = reltol = 1e-12`. Reference: Fehlberg 7(8) + I at $10^{-14}$.

### RK methods with default PI controller

| Combination | Time | Endpoint error | Speedup vs Fehlberg baseline |
|---|---:|---:|---:|
| Fehlberg 7(8) + I *(reference, 1e-12)* | **48.3 µs** | 141 pm | 1.00× |
| Verner 8(7) + PI   | 59.7 µs | 20.6 pm | 0.81× |
| Verner 9(8) + PI   | 82.3 µs | 4.6 pm  | 0.59× |
| Feagin 12(10) + PI | 174 µs  | 14.4 pm | 0.28× |
| Feagin 14(12) + PI | 250 µs  | 69.2 pm | 0.19× |

### Same methods with H211b smoothing

| Combination | Time | Endpoint error |
|---|---:|---:|
| Verner 9(8) + H211b   | 213 µs | 28.5 pm |
| Feagin 14(12) + H211b | 690 µs | 1.26 nm |

H211b's smoothing increases the step count on smooth problems; PI is the
better default. H211b only earns its keep on non-smooth dynamics.

### Taylor method, JorbaZou order sweep

| Order $N$ | Time | Endpoint error |
|---:|---:|---:|
| 8  | 3.23 ms  | 753 fm  |
| 10 | 2.77 ms  | 11.7 pm |
| 12 | 3.10 ms  | 8.5 pm  |
| 16 | 4.71 ms  | 2.2 pm  |
| 20 | 6.65 ms  | 26.1 pm |
| 24 | 8.81 ms  | 18.4 pm |
| 30 | 14.03 ms | 23.4 pm |

Cost grows roughly linearly with $N$ (~0.45 ms / order beyond $N=12$).
Endpoint error becomes dominated by the abstol-tied step-size controller
rather than truncation around $N=16$. The Taylor method is 30–100× slower
than RK on this problem at this tolerance — its advantage is qualitative:
exact polynomial-Newton event location and exact $N$-th-order dense
output. None of those features show up in raw wall-time.

### Taylor + alternate controllers

| Combination | Time | Endpoint error |
|---|---:|---:|
| Taylor N=12 + PI    | 3.69 ms  | 4.5 pm  |
| Taylor N=24 + H211b | 48.21 ms | 11.6 pm |

PI on Taylor at low order is competitive with JorbaZou. **H211b on Taylor is
dramatically slower** — the smoothing kills the high-order step predictor
that JorbaZou exploits.

---

## Takeaways

- **Default choice for smooth problems**: Verner 8(7) + PI.
- **Precision-priority**: Verner 9(8) + PI.
- **Need polynomial event location or exact dense output**: TaylorStepper, even
  at the wall-time cost. Use it when the *qualitative* properties of the
  polynomial form matter (continuous functionals, multi-root detection in
  future versions, ADS integration in Stage 2b).
- **Step-size controller defaults are sane**: don't reach for H211b unless you
  measured a benefit on your actual problem.

The full benchmark report — environment, build flags, raw numbers, and known
caveats — is preserved in the repository at
[`docs/benchmarks/2026-05-21-stage2a-cr3bp.md`](https://github.com/andreapasquale94/tax/blob/main/docs/benchmarks/2026-05-21-stage2a-cr3bp.md).

---

## Propagating an expansion in the initial conditions (DA-vector state)

The five RK steppers accept any state for which `tax::ode::VectorOps<State>`
is specialized. The library provides built-in specializations for
floating-point scalars, `tax::TaylorExpansionT<T,N,M>`, and
`Eigen::Matrix<T,D,1>` of either.

To propagate a polynomial flow map about an initial-condition box, use a
vector of multivariate Taylor polynomials in the IC deviations:

```cpp
using DA    = tax::TEn<2, 4>;            // order 2, 4 IC variables
using State = Eigen::Matrix<DA, 4, 1>;

State x0;
for (int i = 0; i < 4; ++i)
    x0(i) = DA(centre(i)) + DA(halfWidth(i)) * DA::variable(0.0, i);

tax::ode::Verner78<State> integ{ f, cfg };
auto sol = integ.integrate(x0, t0, tmax);

// sol.x.back()(i)[0]    = component i at tmax, constant DA term
// sol.x.back()(i)[1+j]  = ∂(component i)/∂(δ_j) at tmax, scaled by halfWidth(j)
```

Step-size control still operates in `double`: `VectorOps<State>::norm`
returns the sup over all coefficients of the polynomial state, and the
adaptive controller compares that against `cfg.abstol + cfg.reltol *
state_norm`.

The Taylor stepper currently requires a real-scalar state — propagating a
DA-vector state through `Taylor<…>` would require a separate DA-Taylor
integrator (planned, not in this stage).

### Supported state types out of the box

| State type | `VectorOps<S>` specialization | Use case |
|---|---|---|
| `double` (or any floating-point) | `VectorOps<T>` | scalar ODE |
| `Eigen::Matrix<double, D, 1>` | `VectorOps<V>` recursing on inner `double` | classical N-state vector ODE |
| `tax::TEn<P, M>` (a single TE) | `VectorOps<TaylorExpansionT<…>>` | rare; advanced usage |
| `Eigen::Matrix<tax::TEn<P, M>, D, 1>` | `VectorOps<V>` recursing on inner TE | polynomial flow map / sensitivity propagation |

To support a custom state type, specialize `tax::ode::VectorOps<MyState>` in
the `tax::ode` namespace, providing three static functions: `norm(x) →
double`, `axpy(y, double a, x)`, `scale_assign(y, double a, x)`.
