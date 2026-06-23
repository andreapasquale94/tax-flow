---
title: "Low-Thrust Reachability via Differential Algebra and Automatic Domain Splitting"
author: "tax-flow project"
date: "June 2026"
geometry: "left=2.8cm, right=2.8cm, top=3.2cm, bottom=3.2cm"
fontsize: 11pt
linestretch: 1.25
mainfont: "TeX Gyre Pagella"
sansfont: "TeX Gyre Heros"
monofont: "DejaVu Sans Mono"
colorlinks: true
linkcolor: "NavyBlue"
citecolor: "NavyBlue"
urlcolor: "NavyBlue"
header-includes:
  - \usepackage{amsmath,amssymb,mathtools}
  - \usepackage{booktabs}
  - \usepackage{caption}
  - \usepackage{subcaption}
  - \usepackage[dvipsnames]{xcolor}
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhf{}
  - \fancyhead[L]{\small\textcolor{gray}{Low-Thrust Reachability}}
  - \fancyhead[R]{\small\textcolor{gray}{\thepage}}
  - \renewcommand{\headrulewidth}{0.4pt}
  - \setlength{\headheight}{14pt}
  - \captionsetup{font=small, labelfont=bf, format=hang}
---

\newcommand{\bm}[1]{\boldsymbol{#1}}
\newcommand{\R}{\mathbb{R}}

# Introduction

A fundamental question in spacecraft trajectory analysis is: *starting from a
fixed initial state, what region of space can a vehicle reach by the end of a
manoeuvring arc?* Answering it by brute-force sampling — thousands of separate
numerical integrations, one per control choice — is straightforward but
expensive. This document describes a different approach: the *control parameters
themselves* are treated as free variables, and the trajectory is expanded as a
polynomial function of those parameters. A single integration then captures the
response of the trajectory to the entire control space simultaneously.

The enabling technology is **Differential Algebra (DA)**, which replaces
ordinary floating-point arithmetic with arithmetic on truncated Taylor
polynomials. Propagating a DA-valued state through an ordinary differential
equation automatically yields a high-order polynomial map from any chosen free
parameter to the final state — the **flow map**. When the dynamics are
nonlinear enough that a single polynomial is no longer an accurate
representation, **Automatic Domain Splitting (ADS)** partitions the parameter
space into sub-domains, each with its own accurate polynomial. Together, these
two tools turn a reachability problem into a one-shot algebraic computation.

The setting here is a spacecraft on a heliocentric orbit applying continuous
low thrust, with the thrust magnitude and direction as the free (control)
parameters. The output is a family of closed curves — the *reachable boundary*
at successive times — that reveals how the spacecraft's accessible region grows
over one orbital revolution, and how it scales with control authority.

---

# Differential Algebra and the Flow Map

## Taylor Polynomials as Computational Objects

The idea of propagating polynomials instead of numbers is old in celestial
mechanics (Jet Transport, Taylor arithmetic), but DA makes it systematic. Every
real number is replaced by a truncated power series in $M$ free variables
$\xi = (\xi_1, \ldots, \xi_M)$:

$$
f(\xi) = \sum_{|\alpha| \le P} c_\alpha\, \xi^\alpha,
\qquad \alpha \in \mathbb{N}^M,
\quad |\alpha| = \alpha_1 + \cdots + \alpha_M,
$$

where $P$ is the *truncation order* and $\xi^\alpha = \xi_1^{\alpha_1} \cdots
\xi_M^{\alpha_M}$. The number of coefficients per variable is
$\binom{P+M}{M}$; for $P = 6$, $M = 2$ this is 28 — very compact. All
elementary operations (addition, multiplication, $\sin$, $\exp$, $1/x$, \ldots)
have well-defined DA counterparts that propagate the coefficient arrays exactly
up to order $P$, discarding any term of degree $> P$.

## The Flow Map

Consider an autonomous ordinary differential equation

$$
\dot{\mathbf{s}} = f(\mathbf{s}), \qquad \mathbf{s}(t_0) = \mathbf{s}_0(\xi),
$$

where the initial condition depends polynomially on the free variables $\xi$.
The simplest choice is a *linear* seed:

$$
s_{0,i}(\xi) = \bar{s}_{0,i} + h_i\, \xi_i,
$$

with $\bar{\mathbf{s}}_0$ a reference initial state and $h_i$ a half-width
along the $i$-th variable. Because every arithmetic operation on the right-hand
side $f$ is performed in DA algebra, the integrator automatically accumulates
all partial derivatives of the solution with respect to $\xi$. At any time $t$
the solution is the **flow polynomial**:

$$
\bm{\Phi}_t(\xi) = \sum_{|\alpha| \le P} \mathbf{c}_\alpha(t)\, \xi^\alpha,
$$

a polynomial map that sends any point $\xi \in [-1,1]^M$ (i.e., any point in
the initial-condition box) to its propagated state. The higher-order
coefficients $\mathbf{c}_\alpha$ encode the sensitivity of the final state to
the initial parameters, up to order $P$.

## Truncation and Convergence

The DA framework computes the *exact* Taylor coefficients up to order $P$ (no
finite-difference approximation is involved). The truncation error is the
neglected remainder of the series — terms of degree $> P$. Whether this error
is negligible depends on both the order and the size of $\xi$. Roughly, the
error is controlled by the smallest coefficient at order $P$: if the series is
converging, the order-$P$ block is much smaller than the lower-order terms. When
it is not converging — either because the box is too large, or the flow is too
nonlinear — the order-$P$ coefficients stay large, and the polynomial is no
longer trustworthy outside a small neighbourhood of the expansion centre.

---

# Automatic Domain Splitting

## Why a Single Polynomial Fails

Figure 1 illustrates the failure mode with a simple example: a small box of
initial conditions propagated around an eccentric Kepler orbit. Early in the
propagation the polynomial tracks the true boundary well. By the time the
spacecraft has returned to periapsis (the most nonlinear part of the orbit),
the polynomial extrapolates wildly outside the true set — the characteristic
symptom of a truncated series that has lost convergence.

\begin{figure}[h]
\centering
\includegraphics[width=0.82\textwidth]{img/two_body_flow.png}
\caption{A single order-6 flow polynomial tracking an eccentric Kepler orbit.
Each curve is the image of the initial-condition box at a snapshot time (blue
early, yellow at one full period). The yellow tail near the bottom is the
polynomial diverging — it is no longer an accurate representation of the true
set.}
\end{figure}

## The Splitting Criterion

ADS (Wittig et al., 2015) monitors the Taylor polynomial *during* propagation
and splits the current domain in two whenever the series stops converging. The
diagnostic is the **truncation mass**: the $L^1$ norm of the highest-degree
block of coefficients,

$$
\varepsilon_P = \sum_{|\alpha| = P} |c_\alpha|.
$$

When $\varepsilon_P$ exceeds a user-set tolerance $\tau$, the polynomial is no
longer faithful and the domain is bisected. The **split direction** is the
variable $j$ contributing most to the tail:

$$
j^* = \arg\max_j \sum_{|\alpha| = P} |c_\alpha|\, \alpha_j.
$$

Splitting along $j^*$ halves the interval $[-1, 1]$ for that variable. Each
child sub-domain is re-expanded on its own box — a fresh identity polynomial
centered on the child — and integration continues independently. The result is
a **binary tree of leaves**: each leaf owns a compact sub-domain and a flow
polynomial that is accurate on it. The leaves together tile the entire original
domain.

## LOADS: A Jacobian-Based Alternative

The truncation criterion needs a moderately high polynomial order ($P \ge 6$) to
have a meaningful tail to measure. An alternative, the **LOADS** criterion
(Losacco, Fossà & Armellin, 2024), estimates nonlinearity from the *Jacobian*
of the map. It forms the ratio of the nonlinear mass of $\partial\bm{\Phi}/\partial\xi$
to its linear part:

$$
\eta = \frac{\|\partial\bm{\Phi}/\partial\xi - J_1\bm{\Phi}\|}{\|J_1\bm{\Phi}\|},
$$

where $J_1\bm{\Phi}$ retains only the degree-1 part of the Jacobian. When
$\eta$ exceeds the tolerance, the Jacobian is telling us that the map is
bending significantly across the domain, regardless of whether the polynomial
tail is large. The LOADS index requires only a low-order polynomial (order 2 is
sufficient) and tends to split *earlier* — it reacts to the onset of
nonlinearity rather than to its full development.

## ADS in Practice

Figure 2 shows ADS applied to the same Kepler problem. Where the single
polynomial shoots off the frame, the ADS partition — here 15 leaves at the end
of one revolution — stays glued to the true boundary. Each coloured strip is a
leaf; their collective image faithfully represents the propagated set at all
snapshot times.

\begin{figure}[h]
\centering
\includegraphics[width=0.85\textwidth]{img/two_body_ads.png}
\caption{Single polynomial (dashed red) versus ADS (coloured strips) at two
snapshot times around an eccentric Kepler orbit. Left: $t = 5.50$, six leaves.
Right: $t = 2\pi$ (one full period), fifteen leaves. The ADS partition remains
accurate where the single polynomial diverges.}
\end{figure}

---

# Problem Formulation: Low-Thrust Reachability

## Dynamics

The spacecraft moves in the **planar heliocentric two-body problem** in
canonical units: gravitational parameter $\mu = 1$, reference radius
$r_0 = 1$ AU, circular speed $v_0 = 1$, and one orbital period
$T = 2\pi$. The equations of motion with continuous thrust are

$$
\ddot{x} = -\frac{x}{r^3} + m\,d_x,
\qquad
\ddot{y} = -\frac{y}{r^3} + m\,d_y,
\qquad
r = \sqrt{x^2 + y^2},
$$

where $m \ge 0$ is the thrust acceleration magnitude and $(d_x, d_y)$ is the
thrust direction.

The thrust direction is parameterised by an angle $\theta$ measured *from the
instantaneous velocity* ($\theta = 0$ is prograde):

$$
\begin{pmatrix} d_x \\ d_y \end{pmatrix}
= R(\theta)\,\hat{\mathbf{v}},
\qquad
\hat{\mathbf{v}} = \frac{(v_x, v_y)}{\|\mathbf{v}\|},
\qquad
R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}.
$$

The pair $(m, \theta)$ fully specifies the thrust law, and together they
define the **control**.

## The Augmented State and Zero Dynamics

The key move is to treat the two control parameters as extra *state
components* with identically zero time derivatives — so-called zero dynamics:

$$
\mathbf{s} =
\begin{pmatrix} m \\ \theta \\ x \\ y \\ v_x \\ v_y \end{pmatrix},
\qquad
\dot{m} = 0, \quad \dot{\theta} = 0.
$$

The full system is then six-dimensional ($D = 6$), but only two of those
components ($m$ and $\theta$) are the expansion variables ($M = 2$). The
right-hand side is a single smooth vector field, valid simultaneously for
real arithmetic (when integrating a specific trajectory) and for DA arithmetic
(when building the polynomial map over a range of controls). The integrator
sees a perfectly ordinary ODE; only the *type* of the state components changes
depending on the use.

## Control Box and Initial Condition

The initial state corresponds to a **circular heliocentric orbit** at $r_0 = 1$
AU: $(x_0, y_0) = (1, 0)$, $(v_{x,0}, v_{y,0}) = (0, 1)$. The thrust
magnitude and direction are free within the **control box**:

$$
m \in [0,\, a_{\max}], \qquad \theta \in [0,\, 2\pi).
$$

In normalized DA variables $\xi_m \in [-1,1]$ and $\xi_\theta \in [-1,1]$,
this box has centre and half-width:

$$
\bar{m} = \tfrac{a_{\max}}{2}, \quad h_m = \tfrac{a_{\max}}{2};
\qquad
\bar{\theta} = \pi, \quad h_\theta = \pi.
$$

The six-dimensional initial state seeded into the DA integrator is therefore

$$
\mathbf{s}_0(\xi) =
\begin{pmatrix}
\tfrac{a_{\max}}{2} + \tfrac{a_{\max}}{2}\,\xi_m \\
\pi + \pi\,\xi_\theta \\
1 \\ 0 \\ 0 \\ 1
\end{pmatrix}.
$$

The four physical state components ($x, y, v_x, v_y$) are fixed constants: the
only variability comes from the two control axes.

## Control Authority

The maximum acceleration $a_{\max}$ is determined by the spacecraft's thruster
and mass. With thrust force $F$ (in Newtons) and spacecraft mass $M_{\rm sc}$
(in kilograms), the physical acceleration $F/M_{\rm sc}$ is normalized by the
solar gravitational acceleration at 1 AU:

$$
a_{\max} = \frac{F / M_{\rm sc}}{\mu_\odot / \mathrm{AU}^2},
\qquad \frac{\mu_\odot}{\mathrm{AU}^2} = 5.9301 \times 10^{-3}\ \mathrm{m/s^2}.
$$

Two representative spacecraft are considered:

| Configuration | Mass | Thrust | $a_{\max}$ (canonical) |
|:---|---:|---:|---:|
| Spacecraft | 1000 kg | 100 mN | $\approx 0.0169$ |
| CubeSat | 40 kg | 2 mN | $\approx 0.0084$ |

The spacecraft's control authority is approximately twice that of the CubeSat.

## Reachable Set and Its Envelope

At any time $t$, the **reachable set** is the image of the control box under
the flow:

$$
\mathcal{R}(t) = \left\{ \bm{\Phi}_t(\xi_m, \xi_\theta) : \xi_m \in [-1,1],\; \xi_\theta \in [-1,1] \right\}\Big|_{(x,y)\text{ components}}.
$$

Its boundary — the **envelope** — is the image of the box's *outer edge*
where $m = a_{\max}$ (i.e., $\xi_m = +1$), swept over all directions
$\theta \in [0, 2\pi)$. The inner edge $m = 0$ collapses to the single
ballistic point (no thrust applied), which is interior to the set: the
spacecraft at $m = 0$ follows the reference circular orbit regardless of
$\theta$. Therefore the reachable boundary is

$$
\partial\mathcal{R}(t) = \left\{ \bm{\Phi}_t(+1,\, \xi_\theta) \,\Big|\, (x,y) : \xi_\theta \in [-1,1] \right\}.
$$

In practice, this curve is sampled by evaluating each ADS leaf polynomial at
$(+1, \xi_\theta)$ for a dense grid of $\xi_\theta$ values, selecting whichever
leaf contains each point, to preserve accuracy across the tree partition.

## Propagation Setup

Sweeping $\theta$ from $0$ to $2\pi$ makes $\cos\theta$ and $\sin\theta$
strongly nonlinear in $\xi_\theta$; a single polynomial cannot represent this
faithfully, and ADS must subdivide the $\theta$ axis. The propagation
parameters are:

| Parameter | Value |
|:---|:---|
| Polynomial order $P$ | 6 |
| Expansion variables $M$ | 2 (the control pair $(m, \theta)$) |
| State dimension $D$ | 6 |
| ADS tolerance $\tau$ | $10^{-5}$ |
| Maximum tree depth | 10 |
| Numerical integrator | Verner 8(9) embedded Runge–Kutta |
| Integration tolerance | $10^{-12}$ (absolute and relative) |

A separate ADS propagation is performed for each of 36 snapshot times, spaced
10 days apart over one orbital period (one period $= 365.25$ days in physical
units). This produces 36 trees, each with its own leaf count reflecting the
nonlinearity at that propagation time.

---

# Results and Findings

## Reachable-Set Growth

Figure 3 shows the reachable-set boundary at each snapshot time, for both
spacecraft configurations, overlaid on the same heliocentric coordinate system.
The Sun sits at the origin; the reference circular orbit is the grey circle of
radius 1.

\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{img/reachability_compare.png}
\caption{Low-thrust reachability envelopes over one orbital revolution.
Left: 1000 kg spacecraft with 100 mN thruster ($a_{\max} \approx 0.0169$).
Right: 40 kg CubeSat with 2 mN thruster ($a_{\max} \approx 0.0084$).
Each closed curve is the reachable boundary at one 10-day snapshot (blue
early in the orbit, red after one full revolution). The dot at $(0,0)$ is
the Sun.}
\end{figure}

Each closed curve is the image of the max-thrust boundary
$\{(m, \theta) : m = a_{\max},\, \theta \in [0, 2\pi)\}$ under the flow at
that snapshot time. The curves are coloured by $t/T$ (blue at $t \approx 0$,
red at $t = T = 2\pi$).

## Qualitative Behaviour

At early times, the reachable set is a tiny loop hugging the reference orbit:
with only a few days of continuous thrust, the accumulated $\Delta v$ is small
and the set has not diverged far from the ballistic trajectory. As time
progresses, the envelope grows into a broad crescent that wraps around the Sun.
The growth is not uniform: the leading and trailing edges (prograde and
retrograde thrust) accumulate faster, since an along-track burn changes the
orbital energy directly and therefore moves the spacecraft to a higher or lower
orbit. Transverse thrust directions generate more eccentric, inclined
trajectories that close the crescent laterally.

By the end of one revolution, the reachable set covers a substantial annular
arc around the reference orbit. The shape is asymmetric between the inner and
outer boundaries: thrust applied inward (reducing the orbit) is more effective
at changing position because periapsis is reached sooner; outward thrust raises
apoapsis and the spacecraft spends more time in the outer region.

## Scaling with Control Authority

The two panels in Figure 3 share exactly the same dynamics, reference orbit,
and plotting axes — only $a_{\max}$ differs. The result is striking in its
simplicity: **the reachable set scales approximately linearly with
$a_{\max}$**. The spacecraft (left) reaches roughly twice as far as the CubeSat
(right) in every direction, consistent with the factor-of-two ratio in their
control authorities.

This linear scaling is expected in the *small-thrust limit*: for small
$a_{\max}$ the effect of thrust on the trajectory is a first-order perturbation,
and the boundary of the reachable set moves proportionally to the magnitude of
the perturbation. The ADS polynomials make this explicit: the leading
(degree-1) Taylor coefficient of the position with respect to $m$ gives the
instantaneous drift rate, and the boundary curves are, to leading order, the
image of the segment $\{m = a_{\max},\, \theta \in [0, 2\pi)\}$ under this
linear map.

## Role of ADS

A single polynomial in $(\xi_m, \xi_\theta)$ cannot represent the reachable
set accurately because $\cos\theta$ and $\sin\theta$ are periodic over the
full $[0, 2\pi]$ sweep — a degree-6 polynomial simply cannot fit a full period
of a trigonometric function faithfully. ADS resolves this by splitting the
$\theta$ axis repeatedly until each leaf subtends a small enough arc that its
polynomial approximation is valid. The number of leaves grows with the
propagation time (longer arcs of thrusting accumulate more nonlinearity) and
is bounded by the depth limit of 10, corresponding to at most
$2^{10} = 1024$ leaves.

The tree-of-leaves structure means the envelope extraction is not just an
evaluation at a single polynomial: it is a piecewise evaluation, one per
relevant leaf, stitched together along the $\theta$ axis. Each stitch is
invisible to the user — the accuracy guarantee is uniform across the full
control circle, provided the ADS tolerance is tight enough.

## Efficiency

The entire computation for one snapshot time — building the ADS tree and
extracting the envelope — takes a fraction of a second on a modern workstation.
Thirty-six snapshots therefore complete in well under a minute, far faster than
any Monte Carlo alternative. This efficiency comes from the polynomial
structure: evaluating a degree-6 polynomial in two variables at a point costs
28 floating-point operations per state component, regardless of the integration
history. The integration work is done once, upfront, per leaf; evaluation is
essentially free thereafter.

---

# Summary

The method developed here addresses low-thrust reachability by combining three
ideas:

1. **Zero dynamics**: the two control parameters (magnitude $m$ and direction
   $\theta$) are appended to the physical state with trivial time derivatives.
   This converts a parametric family of ODE problems into a single ODE over a
   six-dimensional state.

2. **Taylor maps**: the augmented initial condition is seeded as a degree-6
   polynomial in the two control variables. Propagating it through the
   numerical integrator yields a polynomial flow map from control space to
   final state, capturing the full sensitivity to $m$ and $\theta$ in 28
   coefficients per state component.

3. **Automatic Domain Splitting**: wherever the polynomial map loses accuracy
   (diagnosed by the truncation mass criterion), the control box is bisected
   and each half is re-integrated independently. The result is a binary tree
   of accurate, piecewise polynomial maps that together cover the full control
   rectangle.

The reachable-set boundary is then read off directly as the image of the
max-thrust circle $m = a_{\max}$, $\theta \in [0, 2\pi)$ through the piecewise
map. No sampling, no root-finding, and no separate integrations per control
point are required.

The two spacecraft examples confirm that the reachable set grows as a crescent
around the reference orbit, that the boundary scales approximately linearly with
control authority $a_{\max}$, and that the method is computationally
inexpensive even for 36 time snapshots. These properties make it suitable for
preliminary mission analysis, contingency planning, and any setting where a
rapid global picture of reachable space is more useful than a small number of
optimised trajectories.

---

# References

Wittig, A., Di Lizia, P., Armellin, R., Makino, K., Bernelli-Zazzera, F., &
Berz, M. (2015). *Propagation of large uncertainty sets in orbital mechanics by
automatic domain splitting*. Celestial Mechanics and Dynamical Astronomy,
122(3), 239–261. <https://doi.org/10.1007/s10569-015-9618-3>

Losacco, M., Fossà, A., & Armellin, R. (2024). *LOADS: Low-order automatic
domain splitting for nonlinear uncertainty propagation*. Journal of Guidance,
Control, and Dynamics, 47(1), 107–122. <https://doi.org/10.2514/1.G007271>
