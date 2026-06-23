---
title: "Missed-Thrust Dispersion for a Bang-Bang Electric Thruster"
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
  - \usepackage[dvipsnames]{xcolor}
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhf{}
  - \fancyhead[L]{\small\textcolor{gray}{Missed-Thrust Dispersion}}
  - \fancyhead[R]{\small\textcolor{gray}{\thepage}}
  - \renewcommand{\headrulewidth}{0.4pt}
  - \setlength{\headheight}{14pt}
  - \captionsetup{font=small, labelfont=bf, format=hang}
---

\newcommand{\bm}[1]{\boldsymbol{#1}}

# Introduction

Electric propulsion systems for deep-space and heliocentric missions deliver
their impulse through long, near-continuous thrust arcs rather than short
impulsive burns. A fundamental reliability question is: *if the thruster
switches off unexpectedly — a missed-thrust event — where might the spacecraft
end up, and how does the answer depend on how often and how long such outages
occur?*

This document addresses that question for a **bang-bang thruster**: one that is
either fully ON or fully OFF, with no intermediate throttle setting. The
thruster's ON/OFF state is modelled as a two-state Markov chain that can switch
at every 4-day arc boundary, imposing a minimum dwell of 4 days. Three
thruster-reliability scenarios are compared, from a highly reliable unit
(outages rare and brief) to an unreliable one (nearly half the arcs are off on
average). In addition to the random switching pattern, two sources of
*execution error* are present: a small, fixed bias in thrust magnitude and a
small, fixed pointing offset, both active whenever the thruster is ON.

The computational method decouples the two uncertainty types. The execution
errors are handled analytically — they seed a **Taylor polynomial surrogate**
that captures the sensitivity of the spacecraft position to those biases without
any extra numerical integration. The random outage schedules are then handled by
a Monte Carlo over Markov-chain sequences, but each evaluation costs only a
cheap polynomial evaluation rather than a full orbital integration. The result
is a sequence of probability density histograms over the orbital plane, from
which **highest-density confidence bands** are extracted at 1σ, 2σ, and 3σ
levels.

---

# Mathematical Background

## Taylor Polynomial Surrogates

The core technique is **Differential Algebra (DA)**: replacing real numbers in
the equations of motion with truncated Taylor polynomials in a set of free
parameters. Here the free parameters are the two execution errors
$(\delta_m, \delta_\theta)$. The equations of motion are formally identical to
those for real numbers, but every arithmetic operation — including
trigonometric functions, square roots, and divisions — is performed on
polynomial objects rather than scalars. The result, after integrating one arc,
is a polynomial map

$$
\bm{\Phi}_{\Delta}(\delta_m, \delta_\theta) = \sum_{|\alpha| \le P}
\mathbf{c}_\alpha\, \delta_m^{\alpha_1} \delta_\theta^{\alpha_2}
$$

that sends any pair of execution errors to the state at the end of that arc.
The coefficients $\mathbf{c}_\alpha$ are computed *exactly* (not by finite
differences) as a by-product of the integration. At truncation order $P = 6$
and two variables, each state component is described by
$\binom{6+2}{2} = 28$ coefficients.

Because $\delta_m$ and $\delta_\theta$ are small — a few percent and a few
degrees respectively — the degree-6 polynomial is highly accurate over the
entire error box, and the truncation error is negligible.

## Arc-by-Arc Composition

The revolution is divided into $N$ arcs. On each arc the thruster state
$f_k \in \{0, 1\}$ is fixed, so the right-hand side of the ODE is smooth within
each arc: there is no switching during an integration step. The key operation
is to **carry the Taylor polynomial across arc boundaries** without explicit
composition.

After integrating arc $k$, the state is a polynomial in the execution errors.
This polynomial is then used directly as the initial condition for arc $k+1$,
with only the constant part updated by the arc's dynamics. The integrator
processes the Taylor-valued state exactly as it would a real-valued one — the
polynomial algebra is transparent to the time-stepping scheme. After $N$ arcs
the state polynomial is the composition of all $N$ per-arc maps:

$$
\bm{\Phi}_{N\Delta} = \bm{\Phi}_{\Delta}^{(N)} \circ \cdots \circ
\bm{\Phi}_{\Delta}^{(2)} \circ \bm{\Phi}_{\Delta}^{(1)},
$$

and this composition is performed automatically, arc by arc, at the cost of
$N$ integrations of a Taylor-valued ODE — one per arc, each of duration
$\Delta = 4\ \text{days}$.

## Monte Carlo over Outage Schedules

The execution errors are captured by the polynomial surrogate. The Markov
outage schedules are handled differently: each schedule (a sequence of 91
ON/OFF decisions) is drawn independently from the chain, and for that schedule
the entire arc-by-arc propagation is run once. At the end of each arc, the
polynomial is evaluated at $N_{\rm draw} = 16$ randomly sampled execution-error
pairs $(\delta_m, \delta_\theta)$, adding 16 points to the histogram at that
snapshot time. With 8000 schedules the histogram accumulates
$8000 \times 16 = 128\,000$ points per snapshot, enough for well-resolved
confidence contours.

This two-level approach — polynomial over execution errors, Monte Carlo over
schedules — is efficient because the polynomial evaluation is orders of
magnitude cheaper than an integration: 28 floating-point operations per state
component versus thousands of right-hand-side calls.

## Highest-Density Confidence Bands

From each snapshot histogram the confidence bands are extracted as
**highest-density regions (HDR)**: the smallest connected area of the plane
enclosing a given probability mass. A 2-D Gaussian-equivalent labelling is
used so the band labels carry familiar meaning:

| Band | 2-D HDR coverage |
|:---|---:|
| $1\sigma$ | 39.35% |
| $2\sigma$ | 86.47% |
| $3\sigma$ | 98.89% |

The threshold density for each band is found by a sorted cumulative-sum walk on
the (lightly Gaussian-smoothed) histogram: sort all pixels by descending
density, accumulate mass until the target fraction is reached, and use that
pixel's density as the contour level. The result is a contour that need not be
elliptical — it traces the actual shape of the probability cloud.

---

# Problem Formulation

## Orbit and Nominal Thrust Plan

The spacecraft moves in the **planar heliocentric two-body problem** in
canonical units: gravitational parameter $\mu = 1$, reference radius
$r_0 = 1\ \text{AU}$, circular speed $v_0 = 1$, and one orbital period
$T = 2\pi$ (approximately 1 year). The initial state is the circular orbit:

$$
(x_0, y_0, \dot{x}_0, \dot{y}_0) = (1,\, 0,\, 0,\, 1).
$$

The **nominal thrust plan** is continuous full thrust at constant magnitude
$m_{\rm nom}$ in the prograde direction (thrust angle $\theta_{\rm nom} = 0$).
For the reference spacecraft (1000 kg, 100 mN thruster):

$$
m_{\rm nom} = \frac{F}{M_{\rm sc}\,(\mu_\odot/\text{AU}^2)}
= \frac{0.1\ \text{N}}{1000\ \text{kg} \times 5.9301\times10^{-3}\ \text{m/s}^2}
\approx 0.0169.
$$

## The 4-Day Decision Grid

The revolution is divided into **91 arcs of 4 days each**. The thruster may
only change state at an arc boundary, so it holds ON or OFF for *at least* 4
days — the minimum dwell constraint. Each arc $k$ carries a binary thrust
fraction

$$
f_k \in \{0,\, 1\}, \qquad k = 1, \ldots, 91.
$$

When $f_k = 0$ the arc is a pure ballistic coast; when $f_k = 1$ the thruster
fires at full nominal thrust, modulated by the execution errors.

## Uncertainty Model

Two independent uncertainty sources act simultaneously:

**Execution errors** (per-trajectory, fixed for the whole revolution):

$$
\delta_m \sim \mathcal{U}[-2\%,\,+2\%], \qquad
\delta_\theta \sim \mathcal{U}[-5°,\,+5°].
$$

These represent the thruster's calibration uncertainty: a constant
fractional error in delivered thrust magnitude and a constant bias in the
pointing direction. They are fixed for each trajectory — the same
miscalibration acts on every ON arc of that particular trajectory.

**Outage sequence** (per-schedule, random from arc to arc):

$$
f_k \in \{0, 1\},
$$

drawn from the Markov chain described below. This represents random
thruster trips and recoveries.

## Equations of Motion with Zero Dynamics

The execution errors are appended to the physical state as two extra components
with identically zero time derivatives — the same zero-dynamics trick used in
the reachability analysis:

$$
\mathbf{s} = \bigl(\delta_m,\; \delta_\theta,\; x,\; y,\; v_x,\; v_y\bigr),
\qquad \dot\delta_m = 0, \quad \dot\delta_\theta = 0.
$$

The full equations of motion on arc $k$ are

$$
\dot{x} = v_x, \qquad \dot{y} = v_y,
$$

$$
\dot{v}_x = -\frac{x}{r^3} + f_k\, m_{\rm nom}(1+\delta_m)\, d_x, \qquad
\dot{v}_y = -\frac{y}{r^3} + f_k\, m_{\rm nom}(1+\delta_m)\, d_y,
$$

where $r = (x^2+y^2)^{1/2}$ and the thrust direction is

$$
\begin{pmatrix} d_x \\ d_y \end{pmatrix}
= R(\theta_{\rm nom}+\delta_\theta)\,\hat{\mathbf{v}},
\qquad
\hat{\mathbf{v}} = \frac{(v_x,\, v_y)}{\|(v_x,\, v_y)\|},
\qquad
R(\alpha) = \begin{pmatrix}\cos\alpha & -\sin\alpha\\ \sin\alpha & \cos\alpha\end{pmatrix}.
$$

On an OFF arc ($f_k = 0$) the thrust term vanishes and the motion is pure
Keplerian. The execution errors still appear in the state but have no effect
until the thruster is next ON.

## Two-State ON/OFF Markov Chain

The sequence of $f_k$ values follows a **sticky two-state Markov chain** on
$\{\text{OFF},\, \text{ON}\}$, transitioning once per 4-day arc boundary. The
transition matrix is

$$
P =
\begin{pmatrix}
1 - p_{\rm recover} & p_{\rm recover} \\[4pt]
p_{\rm fail} & 1 - p_{\rm fail}
\end{pmatrix},
$$

where $p_{\rm fail} = P(\text{ON} \to \text{OFF})$ is the probability of an
unexpected trip-off per arc, and $p_{\rm recover} = P(\text{OFF} \to \text{ON})$
is the probability of recovery per arc. The chain starts ON ($f_0 = 1$) at
$t = 0$.

The stationary ON-duty fraction of this chain is

$$
\pi_{\rm ON} = \frac{p_{\rm recover}}{p_{\rm fail} + p_{\rm recover}}.
$$

Three scenarios bracket the range of thruster reliability:

| Scenario | $p_{\rm fail}$ | $p_{\rm recover}$ | Stationary ON duty |
|:---|:---:|:---:|:---:|
| Reliable | 0.02 | 0.60 | $\approx 97\%$ |
| Intermittent | 0.08 | 0.30 | $\approx 79\%$ |
| Unreliable | 0.20 | 0.15 | $\approx 43\%$ |

In the reliable case, outages are rare ($p_{\rm fail} = 0.02$, or about one
trip every 50 arcs) and short-lived ($p_{\rm recover} = 0.60$, so on average
the thruster is off for $1/0.60 \approx 1.7$ arcs before recovering). In the
unreliable case, outages occur every 5 arcs on average and last for
$1/0.15 \approx 6.7$ arcs — the thruster is off nearly as often as it is on.

## Propagation Setup

| Parameter | Value |
|:---|:---|
| Polynomial order $P$ | 6 |
| Expansion variables $M$ | 2 ($\delta_m$, $\delta_\theta$) |
| State dimension $D$ | 6 |
| Error box half-widths | $2\%$ (magnitude), $5°$ (pointing) |
| Arc duration | 4 days |
| Number of arcs per revolution | 91 |
| Numerical integrator | Verner 8(9) embedded Runge–Kutta |
| Integration tolerance | $10^{-12}$ (absolute and relative) |
| Monte Carlo schedules | 8000 |
| Draws per schedule per snapshot | 16 |
| Histogram grid | $130 \times 130$ bins |

---

# Results and Findings

## Dispersion Growth over One Revolution

Figure 1 shows the $3\sigma$ confidence boundary at each 4-day snapshot,
overlaid on the orbital plane for all three reliability scenarios. The grey
circle is the ballistic orbit (no thrust ever applied); the green dashed arc
is the nominal trajectory (thruster always ON, no execution errors). Each
coloured ellipse is the $3\sigma$ boundary at one snapshot, coloured by $t/T$
from blue (early) to red (end of revolution). The Sun is at the origin.

\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{img/missed_thrust_onoff.png}
\caption{Three-sigma dispersion boundaries at each 4-day snapshot over one
heliocentric revolution. Left: reliable thruster ($p_{\rm fail}=0.02$,
$p_{\rm recover}=0.60$). Centre: intermittent ($0.08$, $0.30$). Right:
unreliable ($0.20$, $0.15$). Ellipses are coloured by $t/T$; the grey circle
is the ballistic orbit and the green dashed arc is the nominal (always-on)
trajectory.}
\end{figure}

The dispersion ellipses form a string of beads that trail the spacecraft around
the orbit. Several features stand out:

- In the **reliable** case the beads are small and tightly strung: with only
  $\approx 3\%$ of arcs OFF, the spacecraft almost always thrusts and the
  $3\sigma$ boundary remains a compact loop close to the nominal arc.

- In the **intermittent** case the beads widen visibly after the first quarter
  of the orbit. By the final epoch the $3\sigma$ ellipse is roughly three times
  larger than in the reliable case.

- In the **unreliable** case the beads expand rapidly and, crucially, the
  *mean position* drifts away from the nominal arc toward the ballistic circle.
  With $\approx 57\%$ of arcs OFF on average, the spacecraft delivers only
  about half the intended impulse, so its orbit is intermediate between the
  nominal and ballistic references.

## Final-Epoch Nested Confidence Bands

Figure 2 zooms to the end of the revolution ($t = T$), showing the $1\sigma$,
$2\sigma$, and $3\sigma$ confidence regions together with the nominal endpoint
(green star), the ballistic endpoint (grey circle), and the cloud mean
(marked $\times$).

\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{img/missed_thrust_onoff_zoom.png}
\caption{Nested $1\sigma / 2\sigma / 3\sigma$ confidence regions at the final
epoch ($t = T$) for the three reliability scenarios. Green star: nominal
endpoint. Grey circle: ballistic endpoint. Cross: sample mean. Note that
the plotting scale differs between panels — the unreliable case requires a
much wider field of view.}
\end{figure}

Several points deserve attention:

**Shape.** The confidence bands are not elliptical. Their elongation is aligned
roughly along the orbit arc, reflecting the dominant uncertainty: the spacecraft
can be ahead of or behind its expected position depending on *when* missed-thrust
events occurred. An early outage (coasting for the first few weeks) puts the
spacecraft much further around the orbit than a late outage of the same
duration.

**Mean displacement.** In the reliable case the cloud mean (×) sits very close
to the nominal endpoint — the mean number of missed arcs is small. As
reliability drops, the mean migrates steadily toward the ballistic endpoint. In
the unreliable scenario the mean lies roughly halfway between the nominal and
ballistic endpoints, consistent with a $\approx 43\%$ ON duty fraction
delivering about half the total impulse.

**Scale difference.** The three panels are plotted at very different scales: the
reliable $3\sigma$ band spans $\approx 0.10\ \text{AU}$, the intermittent one
$\approx 0.25\ \text{AU}$, and the unreliable one $\approx 0.70\ \text{AU}$.
Thruster reliability has a super-linear effect on position uncertainty at the
end of the orbit.

**Tail toward ballistic.** All three distributions show a heavier tail on the
side toward the ballistic endpoint (many arcs OFF) than on the side toward the
nominal endpoint (no arcs OFF). The all-ON outcome is unique; the many-OFF
outcomes span a continuum of intermediate trajectories, pushing probability mass
into the tail.

## Duty Cycle and Dispersion Radius versus Time

Figure 3 shows, for each scenario, two time series over the revolution.

\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{img/missed_thrust_onoff_time.png}
\caption{Top row: fraction of Monte Carlo trajectories with the thruster ON
(green) versus OFF (red) at each 4-day arc. Bottom row: radial dispersion
radius $|\mathbf{r} - \bar{\mathbf{r}}|$ at $1\sigma$, $2\sigma$, $3\sigma$
as a function of $t/T$. Columns correspond to the three reliability scenarios.}
\end{figure}

**Top row — duty cycle evolution.** Each stacked area chart shows what fraction
of the 8000 simulated trajectories have the thruster ON (green) or OFF (red) at
each arc. In the reliable case the ON fraction is nearly constant at 97% from
the second arc onward: the chain relaxes to its stationary distribution almost
immediately because recovery is fast ($p_{\rm recover} = 0.60$). In the
unreliable case the transition is visible: the chain starts fully ON but
within the first 10–15 arcs (40–60 days) the OFF fraction rises to settle
around 57%, the stationary level. This transient reflects the chain's mixing
time, $\tau_{\rm mix} \approx 1/(p_{\rm fail}+p_{\rm recover}) = 1/0.35 \approx
2.9\ \text{arcs}$.

**Bottom row — radial dispersion.** The 1/2/3$\sigma$ envelopes of the
radial distance from the ensemble mean grow monotonically with time. This
growth is approximately linear because missed-thrust events create velocity
offsets that translate into position offsets growing linearly in time (within
each arc). Over the full revolution the $3\sigma$ dispersion radius reaches:

| Scenario | $3\sigma$ radial dispersion at $t = T$ |
|:---|:---:|
| Reliable | $\approx 0.12\ \text{AU}$ |
| Intermittent | $\approx 0.24\ \text{AU}$ |
| Unreliable | $\approx 0.36\ \text{AU}$ |

The ratio is roughly proportional to the mean number of missed arcs, confirming
the physical picture: each missed arc contributes an independent velocity
impulse offset that accumulates without cancellation over the revolution.

---

# Summary

The on/off missed-thrust dispersion problem combines two uncertainty sources with
very different mathematical structures:

1. **Execution errors** $(\delta_m, \delta_\theta)$ — small, fixed biases that
   act continuously on every ON arc. These are handled by a **Taylor polynomial
   surrogate**: seeding the state as a degree-6 polynomial in the execution
   errors and propagating it arc by arc, composing the per-arc flow maps
   automatically. The polynomial remains accurate across the full error box
   because the errors are small.

2. **Outage schedules** — binary, random, sequence-valued. These are handled
   by a **Monte Carlo** over Markov-chain draws. For each schedule the
   arc-by-arc propagation runs once; the polynomial surrogate is then evaluated
   cheaply at 16 sample points per arc to fill the histogram without additional
   integrations.

Three reliability scenarios bracket the design space. The results show:

- **Dispersion is dominated by the outage Markov chain**, not the execution
  errors. The $3\sigma$ band grows roughly in proportion to the mean number of
  missed arcs; a spacecraft with 57% ON duty has three times the final-epoch
  uncertainty of one with 97% ON duty.

- **The mean position migrates** from the nominal endpoint toward the ballistic
  one as reliability drops. For the unreliable thruster the mean shift over one
  revolution exceeds 0.5 AU — a first-order navigation error, not a
  statistical tail.

- **The distribution is asymmetric**: heavier tails toward the ballistic
  endpoint (many outages) than toward the nominal one (no outages), regardless
  of scenario. Gaussian confidence ellipses would therefore underestimate the
  risk in the ballistic direction.

- **The DA polynomial surrogate is efficient**: 8000 Monte Carlo schedule
  evaluations complete in well under an hour, while achieving the accuracy of
  a direct ODE integration for the execution-error sensitivity.

These properties make the method well-suited for early mission design, where
thruster reliability parameters are uncertain and a rapid picture of the
position uncertainty at end-of-manoeuvre is more informative than a single
worst-case trajectory.

---

# References

Wittig, A., Di Lizia, P., Armellin, R., Makino, K., Bernelli-Zazzera, F., &
Berz, M. (2015). *Propagation of large uncertainty sets in orbital mechanics by
automatic domain splitting*. Celestial Mechanics and Dynamical Astronomy,
122(3), 239–261. <https://doi.org/10.1007/s10569-015-9618-3>

Losacco, M., Fossà, A., & Armellin, R. (2024). *LOADS: Low-order automatic
domain splitting for nonlinear uncertainty propagation*. Journal of Guidance,
Control, and Dynamics, 47(1), 107–122. <https://doi.org/10.2514/1.G007271>
