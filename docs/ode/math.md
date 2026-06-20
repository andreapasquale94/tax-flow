# Mathematical Foundations — ODE Integrator

This page covers the mathematics behind the methods, controllers, and
event-location strategies shipped in `tax::ode`.

---

## The initial-value problem

Given a vector field $\mathbf{f} : \mathbb{R}^D \times \mathbb{R} \to \mathbb{R}^D$
and an initial condition $(\mathbf{x}_0, t_0)$, the integrator advances the
state by a sequence of accepted steps

$$
(t_n, \mathbf{x}_n) \;\to\; (t_n + h_n, \mathbf{x}_{n+1}), \qquad n = 0, 1, \ldots
$$

with step $h_n$ chosen so the embedded **local error estimate** satisfies the
user-supplied tolerance. The same `Integrator` core runs every shipped method;
the step-taking policy and the error estimator are the differentiator.

---

## Taylor method

### Time-Taylor expansion of the solution

The Taylor stepper of order $N$ constructs the time-Taylor polynomial of the
solution about the current step start $t_n$:

$$
\mathbf{x}(t_n + \tau) \;=\; \sum_{k=0}^{N} \mathbf{x}^{[k]} \, \tau^k
$$

with **normalised Taylor coefficients**

$$
\mathbf{x}^{[k]} \;=\; \frac{1}{k!} \, \frac{d^k \mathbf{x}}{dt^k}\bigg|_{t_n}.
$$

Typical orders are $N = 16$–$25$; the high order allows much larger steps
than classical methods of order 4–8.

### Picard iteration via automatic differentiation

Taylor coefficients are computed by propagating differential algebra through
the RHS $\mathbf{f}$. Starting from $\mathbf{x}^{[0]} = \mathbf{x}_n$, each
new coefficient is

$$
\mathbf{x}^{[k+1]} \;=\; \frac{1}{k+1} \, \mathbf{f}^{[k]}
$$

where $\mathbf{f}^{[k]}$ is the $k$-th Taylor coefficient of
$\mathbf{f}(\mathbf{x}(t), t)$. Because time is represented as a univariate
TE $t = t_n + \tau$, evaluating $\mathbf{f}$ on TE-valued state
automatically yields the entire series
$\mathbf{f}^{[0]}, \mathbf{f}^{[1]}, \ldots, \mathbf{f}^{[N-1]}$ via the
chain rule (the core machinery of [tax's recurrence relations](https://andreapasquale94.github.io/tax/internals/recurrences/)).

After $N$ evaluations of the RHS, every coefficient up to $\mathbf{x}^{[N]}$
is exact (modulo truncation).

### Per-step expansion as `StepEvaluator::eval(τ)`

The Taylor polynomial computed at each step *is* the continuous representation
of the solution within the step. For any $\tau \in [0, h_n]$,

$$
\mathbf{x}(t_n + \tau) \;=\; \sum_{k=0}^{N} \mathbf{x}^{[k]} \, \tau^k
$$

evaluated with Horner via `tax::la::eval(data, τ)`. The integrator exposes this
via `StepEvaluator::eval(τ)`, which events use for root-finding and
mid-step sampling, so event location operates at the full $N$-th-order
accuracy of the method — no separate dense-output data structure is needed.

---

## Runge–Kutta steppers

Each shipped Runge–Kutta pair advances the state by

$$
\mathbf{x}_{n+1} \;=\; \mathbf{x}_n + h \sum_{i=1}^{s} b_i \, \mathbf{k}_i,
\qquad
\mathbf{k}_i \;=\; \mathbf{f}\!\left(\mathbf{x}_n + h \sum_{j=1}^{i-1} a_{ij}\,\mathbf{k}_j,\; t_n + c_i h\right)
$$

with stage count $s$ and Butcher tableau $(c, A, b)$ compiled into
`tax/ode/detail/*_tableaus.hpp`. An **embedded estimator** of one order lower
shares the same stages:

$$
\tilde{\mathbf{x}}_{n+1} \;=\; \mathbf{x}_n + h \sum_{i=1}^{s} \tilde{b}_i \, \mathbf{k}_i,
\qquad
\hat{\mathbf{e}}_n \;=\; \mathbf{x}_{n+1} - \tilde{\mathbf{x}}_{n+1}.
$$

The shipped pairs:

| Method | Order $p$ | Embedded $p_{\text{emb}}$ | Stages $s$ | Source |
|---|:-:|:-:|:-:|---|
| Verner 8(7)   | 8  | 7  | 13 | Verner's "efficient" 8(7), 1978 |
| Verner 9(8)   | 9  | 8  | 16 | Verner's "efficient" 9(8) |
| Fehlberg 7(8) | 7  | 8  | 13 | Fehlberg 1968, classical |
| Feagin 12(10) | 12 | 10 | 25 | Feagin 2007 |
| Feagin 14(12) | 14 | 12 | 35 | Feagin 2010 |

### Event location for RK steppers

RK steppers have `has_step_expansion = false` and carry no continuous extension
in `StepData`. For event location the integrator performs a controller-free
full-order re-step of size $\tau$ — identical to a normal step with $h = \tau$,
but skipping the error estimate and step-size controller. This gives
method-order accurate samples at any query time, which Brent's method uses to
bracket and refine the zero crossing.

---

## Step-size control

### Tolerance norm

The error estimate $\hat{\mathbf{e}}_n$ is reduced to a scalar with the
component-wise **mixed tolerance**

$$
\text{tol}_i \;=\; \text{abstol} + \text{reltol} \cdot \max\bigl(|x_{n,i}|, |x_{n+1,i}|\bigr),
\qquad
\|\hat{\mathbf{e}}\|_{\text{tol}} \;=\; \sqrt{\frac{1}{D} \sum_{i=1}^{D} \left(\frac{\hat{e}_i}{\text{tol}_i}\right)^{\!2}}.
$$

A step is **accepted** when $\|\hat{\mathbf{e}}\|_{\text{tol}} \le 1$. On
rejection the controller proposes a smaller $h$, and the step is retried.

### I (integral) controller

The classical baseline. Stateless; rebuilds the next step purely from the
current error ratio:

$$
h_{n+1} \;=\; h_n \cdot \min\!\left(\rho_{\max},\;
\max\!\left(\rho_{\min},\; \sigma \, \bigl(\text{tol}/\|\hat{\mathbf{e}}\|\bigr)^{1/(p_{\text{emb}}+1)} \right)\right)
$$

with default safety $\sigma = 0.9$, $\rho_{\min} = 0.2$, $\rho_{\max} = 5$.

### PI (Gustafsson) controller — default for RK

Adds a proportional term over the previous error:

$$
h_{n+1} \;=\; h_n \cdot \sigma \,
\Bigl(\text{tol}/\|\hat{\mathbf{e}}_n\|\Bigr)^{\!\beta/(p_{\text{emb}}+1)}
\Bigl(\|\hat{\mathbf{e}}_{n-1}\|/\|\hat{\mathbf{e}}_n\|\Bigr)^{\!\alpha/(p_{\text{emb}}+1)}
$$

with $\alpha = 0.7$, $\beta = 0.4$ by default. On smooth problems this
gives smoother step-size sequences and fewer rejections than I.

### H211b (Söderlind digital filter)

A second-order digital filter on the error history:

$$
h_{n+1} \;=\; h_n \cdot \sigma \,
\Bigl(\text{tol}/\|\hat{\mathbf{e}}_n\|\Bigr)^{\!1/(b(p_{\text{emb}}+1))}
\Bigl(\text{tol}/\|\hat{\mathbf{e}}_{n-1}\|\Bigr)^{\!1/(b(p_{\text{emb}}+1))}
\Bigl(h_n / h_{n-1}\Bigr)^{\!-1/b}
$$

with $b = 4$. Tends to smooth bumpy step sequences on piecewise-smooth or
nearly-stiff problems. **Not** recommended for the Taylor method (it damps
the natural high-order step predictor).

### JorbaZou — default for Taylor

A Taylor-specific predictor (Jorba & Zou 2005) that reads the magnitudes of
the last two polynomial coefficients to estimate the radius of convergence:

$$
h \;=\; \min\!\left(
  \left(\frac{\varepsilon}{\|\mathbf{x}^{[N-1]}\|_\infty}\right)^{\!1/(N-1)},\;
  \left(\frac{\varepsilon}{\|\mathbf{x}^{[N]}\|_\infty}\right)^{\!1/N}
\right)
$$

with the component minimum across the state taken for vector systems. Small
top coefficients ⇒ the series converges over a wide interval ⇒ a large step is
safe. Compile-time guard: instantiating `JorbaZou` with any RK stepper is a
compile error — the signature mismatches by design.

---

## Event location

Events are first-class typed objects (see [Events](events.md)) tested at every
accepted step. For sign-change detection (`RootFindingEvent`), the integrator
must localise a root of a user function $g(\mathbf{x}, t)$ inside
$\tau \in [0, h_n]$.

The integrator exposes `StepEvaluator::eval(τ)`, which returns the state at
$t_n + \tau$ accurate to the method order:

- **Taylor** (`has_step_expansion = true`): evaluates the intrinsic per-step
  expansion via `tax::la::eval(data, τ)` at full $N$-th order.
- **RK** (`has_step_expansion = false`): performs a controller-free full-order
  re-step of size $\tau$.

### Brent's method on `eval(τ)`

`RootFindingEvent` (via `StepEvaluator::findRoot`) checks the endpoint signs
$g(\mathbf{x}_n, t_n)$ and $g(\mathbf{x}_{n+1}, t_{n+1})$. When a sign change
is detected, the scalar function

$$
\varphi(\tau) \;=\; g\!\bigl(\mathtt{eval}(\tau),\; t_n + \tau\bigr)
$$

is handed to the `detail::brent_root` helper, which runs the Dekker–Brent
algorithm (inverse quadratic interpolation with bisection fallback) on the
bracketed interval $[0, h_n]$. The result is derivative-free and
superlinearly convergent.

`brent_root` returns `std::optional<T>`; `std::nullopt` means the safeguard
refused to converge and the event silently skips that step.

---

## Truncation, rounding, and accuracy bounds

The user-tunable knobs that determine final accuracy are:

| Knob | Effect |
|---|---|
| `abstol`, `reltol` | Per-step local error tolerance |
| `min_step`, `max_step`, `max_steps` | Hard limits on $h$ and the step count |
| Stepper order | Local truncation $\mathcal{O}(h^{p+1})$; endpoint error $\sim \mathcal{O}(h^p)$ over a fixed interval |
| Method choice | At fixed tolerance, high-order methods (Verner 9, Feagin 14) take fewer, larger steps but more work per step |

For benchmark numbers on a representative astrodynamics problem, see
[Methods & Benchmarks](methods.md).
