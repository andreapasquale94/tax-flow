# Library status review — 2026-07-05

*Scope: full `tax-flow` at `4a6fb4e` (`feat(domain)!: promote IC domains to a
first-class tax::domain module`) — the `tax::ode` module, the `tax::ads`
module, the new `tax::domain` module, tests, and examples. Follow-up to
`docs/reviews/ads-domain-interface/README.md` (2026-07-02): §1 audits what
happened to that review's findings; §2 and §3 are new findings.*

**Build status.** Builds clean against a sibling `tax` checkout (CMake 3.28,
GCC 13, Eigen 3.4). **All 45 ctest targets pass.** Examples build with
`-Wall -Wextra` with only three `-Wcomment` warnings (multi-line comments in
`missed_thrust*`/`reachability` file headers). All seven Butcher tableaus
were re-verified numerically (row sums, Σb = 1, quadrature order
conditions): coefficients are correct; one metadata defect (O1 below).

---

## 1. Status of the 2026-07-02 review findings

| Item | Status at `4a6fb4e` |
|------|---------------------|
| F1 `locate` on PZ trees | **Fixed** — `contains`/`localize`/`splitOrdinate` are on `LocatableDomain` (`domain/domain.hpp:49-58`); `AdsTree::locate/locateFactors` are constrained (`ads/tree.hpp:218-235`); PZ refuses at compile time. |
| F2 parallel SIGBUS on large DA states | **Open** — workaround comment still gates the PZ example to `num_threads=1` (`examples/two_body/polynomial_zonotope.cpp:68`) and lives in a test comment (`tests/domain/test_polynomial_zonotope.cpp:167`); queue plumbing still keeps full `State` objects by value on worker stacks (`work_pool.hpp:83`, `driver.hpp` `Item`/`LeafVerdict`). Not reproducible at small sizes (P=6, M=2, 4 threads) — consistent with the stack-size hypothesis, which needs large states. Fix or file it; a soundness caveat living only in comments will get lost. |
| F3 `create` center convention | **Fixed (debug only)** — assert in all three overloads (`domain/create.hpp:59/89`, `polynomial_zonotope.hpp:154`); contract documented. The check compiles out under `NDEBUG`, and the documented build is Release — the silent-corruption path the guard was added for is still silent in production builds. Consider an always-on throw or a `TAX_DOMAIN_CHECK` opt-in. |
| F4 `Domain` concept under-stated | **Mostly fixed** — `denormalize` is in `Domain`, exact-inverse trio in `LocatableDomain`. Residuals: `create(domain, x0)` is still an ADL requirement not expressed in any concept, and `AdsTree` default-constructs/moves leaves so `Domain` implicitly needs `std::default_initializable`/`movable` — neither is required, so a minimal conforming third-party domain passes the gate and explodes inside `AdsTree`/`AdsDriver` internals. |
| F5 degenerate `Zonotope` generators | **Fixed** — `localize` uses `completeOrthogonalDecomposition` (rank-revealing) and `contains` checks the reconstruction residual (`domain/zonotope.hpp:90-113`). |
| F6 API shape | **Mostly fixed** — `AdsTree<Payload, Domain>` / `Leaf::domain` landed. Residual: `Box::contains` is exact-boundary while `Zonotope::contains` takes `tol = 1e-12` (`domain/box.hpp:36` vs `zonotope.hpp:101`); `refine` remains Box-only (documented as deferred). |
| P1 factor recovery / tree evaluation | **Landed** — `Box::localize`, `Zonotope::localize`, `AdsTree::locateFactors`; covered by `tests/domain/test_localize_evaluate.cpp`. |
| P2 interval hulls / ellipsoid covers | **Landed** — `Zonotope::intervalHull`, `PolynomialZonotope::intervalHull`, `domain/ellipsoid.hpp` (L1 vs L2 distinction kept explicit). |
| P3 image enclosures | **Landed** — `domain/enclosure.hpp` (`intervalHull`, `zonotopeEnclosure` via the even-exponent shift, `zonotopeFrame`, `ImageZonotope::support`); covered by `tests/domain/test_enclosure.cpp`. Math re-checked: correct. |
| P4 constructors / frame utilities | **Landed** — `Zonotope::oriented`, `zonotopeFrame`, `linearPart` / `flowAlignedRotation` / `reorientZonotope` in `domain/reorient.hpp`. |
| P5 2D projection / polygon utilities | **Open** — still example-local (`examples/common/output.hpp`). |
| P6 example hygiene | **Partly open** — `validation.cpp` dead `posErrors` removed; `missed_thrust` vs `missed_thrust_onoff` still ~380 duplicated lines each; CR3BP `rhs` still ×3. |

---

## 2. New findings — `tax::ode`

Ordered by severity. All verified by reading the shipped code; O1 verified
numerically.

### O1. `Fehlberg78Tab` order metadata is swapped (wrong controller exponent)

`detail/fehlberg_tableaus.hpp:32-33` declares `order = 7; order_emb = 8`, but
the shipped `b` is Fehlberg's **8th-order** weight set (zero weights on
stages 1/11, `41/840` on stages 12/13 — NASA TR R-287; the shipped
`b_emb = 2·b₈ − b₇` reproduces the classic error difference). The stepper
therefore propagates at order 8 (local extrapolation, as Boost.Odeint does)
with an O(h⁸) error estimate, but `EmbeddedRKStepper` passes
`Tab::order_emb = 8` to the controllers, which use exponent
`1/(p_emb+1) = 1/9` instead of the correct `1/8`. Effect: systematic
under-adaptation (weaker response to error changes, extra rejections) and
misleading `order_v` metadata. Fix: `order = 8; order_emb = 7` + comment
sweep. Verified numerically: both weight vectors satisfy the quadrature
conditions through k = 7 and fail at k = 8, and the stage-1/11/12/13 pattern
identifies which is which.

### O2. `GridEvent` is single-use — a reused `Integrator` silently records nothing

`events/grid_event.hpp:61`: `cursor_` is only ever incremented and never
reset; there is no start-of-integration hook in `Event`. A second
`integrate()` on the same (const!) `Integrator` finds
`cursor_ == times_.size()`: `nextStop` returns `nullopt`, `onStep` records
nothing — zero grid records, no error. Any loop reusing one configured
integrator over several ICs gets grid output only for the first. Fix: an
`onStart`/`reset` hook invoked by `integrate`, or per-run event instances
(as `AdsDriver::stepLeaf` already does via `clone()` — which is also why the
ADS path is immune). Related hazard: `clone()` copies a spent cursor.

### O3. Rejection retry drops the event `nextStop` clamp — grid points silently skipped

`integrator.hpp:174-179`: on rejection only the `max_step` and `tmax` clamps
are re-applied; the event-stop cap from lines 154-160 is not. The retried
`h` can exceed the cap: (a) the PI controller can *grow* on a rejected step
(see O4), (b) NaN error feeds the max factor (O5), (c)
`std::max(r.h_next, h_min)` with a user `min_step` larger than the gap. When
the retried accepted step overshoots a grid time, `GridEvent::onStep`
(grid_event.hpp:50-55) advances `cursor_` past it while recording only exact
landings — the grid point is consumed unrecorded and never revisited. Fix:
re-apply the full clamp block on retry.

### O4. PI controller can grow the step on a rejected step

`controllers.hpp:70`: `err_prev_` is updated on every call including
rejections, so on a second consecutive rejection
`pow(tol/err_prev_, -β/k)` > 1 can dominate (e.g. DP45: reject at
err = 100·tol, then err = 1.2·tol → net factor 1.27 — the retry is *larger*
than the step that just failed). Wastes the `max_rejects_per_step` budget and
feeds O3. Standard practice (Hairer/Wanner): cap the factor at 1 after a
rejection, or fall back to the I-controller. (`H211b` is safe — its
history exponent has the right sign.)

### O5. NaN error norm is treated as zero error

`detail/embedded_rk_stepper.hpp:63-67`:
`err_for_ctrl = (out.err_norm > 0.0) ? out.err_norm : tol·eps` — NaN fails
the comparison and becomes `tol·eps`, i.e. the controller sees a *perfect*
step and returns the max growth factor while the step is rejected
(`err_norm <= tol` also false). The integrator retries with `h = 5h`
(possibly hopping over a singularity and accepting a wrong step) or burns 16
growing retries into a misleading "rejection cap reached". Note
`if (h < h_min) throw` also never fires for NaN `h`. Fix: treat non-finite
`err_norm` as a hard rejection with a fixed shrink (e.g. h/2), which is the
standard NaN response.

### O6. Zero error norm shrinks the step on the Taylor path

`controllers.hpp:38,59,95` + `detail/adaptive_rk_step.hpp:163`: on
`err_norm == 0` the I controller returns factor 0.9 (shrink), and PI/H211b
substitute the dimensionally arbitrary `denom = 1`. The RK path floors the
error at `tol·eps` first, but `select_taylor_step`'s generic branch passes
the raw value. Failure: `Taylor<N>` with an explicit `I`/`PI` controller on a
polynomial RHS (exact zero tail coefficients) — every step accepted, `h`
decays geometrically, `t` stalls, run dies with "step size below min_step".
Fix: floor the error (as the RK path does) before the controller call.

### O7. Event stops closer than `h_min` throw; `min_step` widens the tmax-landing window

`integrator.hpp:148,154-166`: (a) a grid time within `h_min` of the current
boundary makes `cap < h_min` → spurious "step size below min_step" throw for
a perfectly satisfiable request (notably when the user sets `cfg.min_step`
coarser than the grid spacing; also reachable via the ADS ulp-interaction,
see A1). A stop within `h_min` should be treated as already reached. (b)
`if (tmax - t < h_min) break` uses the user `min_step` as the float-remainder
tolerance, so `min_step = 1e-3` on a unit interval can silently return up to
1e-3 short of `tmax` (`sol.t.back() != tmax`, no flag). The two meanings of
`h_min` should be separated.

### O8. Smaller `tax::ode` items

- **FixedStep + event clamp**: after landing on a grid time, `h = r.h_next`
  where FixedStep's `next_step` returns the *clamped* `h_used` — the user's
  fixed step is permanently replaced by the clamp remainder
  (`controllers.hpp:166`, `adaptive_rk_step.hpp:114-121`,
  `integrator.hpp:228`). Adaptive controllers recover but their history is
  polluted; standard practice is to restore the unclamped h after a forced
  landing.
- **Terminal-event record ordering**: an event registered before the terminal
  one can leave records timestamped after the truncated end of the
  trajectory (`integrator.hpp:194-211`). Undocumented and surprising.
- **Grid time == t0 never recorded**: `nextStop` skips it and `onStep`'s
  sweep discards it (`grid_event.hpp:35-55`).
- **Missed even crossings / dropped Brent roots**: endpoint-sign root
  detection misses double crossings within one (large) step; a bracketed
  root that exhausts `max_iter` is silently dropped instead of falling back
  to the bisection midpoint (`step_evaluator.hpp:56-68`,
  `brent_root.hpp:87`, `events/root_finding_event.hpp:33`).
- **Per-step state copies**: `x = r.x_new` and `record(t, x)` deep-copy the
  state twice per accepted step (`integrator.hpp:215,226`) — measurable for
  DA states; `std::move(r.x_new)` is safe there. `RKStepData work` is
  reconstructed per `step()` call (`embedded_rk_stepper.hpp:51`) — costly
  for the 25/35-stage Feagin methods.
- **DP45 FSAL unused**: the tableau is FSAL but sets `fsal = false`, and the
  flag is read nowhere — ~1 of 7 RHS evaluations wasted on the advertised
  cheap default (`dormand_prince_tableaus.hpp:33`).
- **Taylor coefficient build is O(N²) full-order TE ops**
  (`steppers/taylor.hpp:85-91`): each sweep calls `f` on full order-N
  operands; truncated-degree arithmetic (Jorba–Zou) would cut this ~O(N)×.
  Also the controller exponent uses `p_emb = N−1` → `1/N`, while the
  indicator is dominated by the `h^{N−1}` term.
- **JorbaZou controller breaks for h < 0** (`controllers.hpp:153`,
  `raw/h_used < 0` → permanent min-factor). Unreachable today
  (`integrate` rejects `tmax <= t0`) but a trap if backward integration is
  ever enabled; the half-plumbed `std::abs(span)` at `integrator.hpp:136`
  suggests it was intended.
- **`config.hpp` doc/validation**: `initial_step` comment promises an
  RHS-magnitude heuristic, actual default is `span/100`; negative values
  silently mean "unset"; nothing validates `min_step <= max_step`.

---

## 3. New findings — `tax::ads` + `tax::domain`

### A1. Split landing 1 ulp before a snapshot grid time aborts the whole run

`driver.hpp:118-131` + `grid_event.hpp:35-39` + `integrator.hpp:148-166`:
a child's `tEntry` is `req.t = fl(t + fl(g − t))` of a step the GridEvent
clamped onto grid time `g`; that sum can round to `g − 1 ulp`. The child's
fresh GridEvent clone then returns `nextStop = g` (strict, tolerance-free
`times_[i] > t`), the pre-step cap becomes ~1 ulp < `h_min`, and the
integrator throws "step size below min_step" — killing the entire
propagation (serial and parallel). Fix options: tolerance in `nextStop`
(mirror `onStep`'s landing tolerance), snap `req.t` to the grid time, or
treat `cap < h_min` as "already at the stop" (also fixes O7a). No test
places a grid time at/near a split time.

### A2. `refine()` returns a tree with a stale, fully-populated work queue

`refine.hpp:92,237` + `tree.hpp:64,130-131`: `AdsRefineDriver` drives its own
`std::deque<WorkItem>` and never pops `AdsTree::workQueue_`, which `init` and
every `split` push onto. Verified at runtime: the returned tree has
`empty() == false` and `front()` is the *retired root*. Any consumer using
the documented queue API on a refine result (`tree.empty()` as "complete",
feeding into a driver loop) pops retired/done leaves and trips asserts (or
corrupts state in release). The classic driver drains the queue, so the two
entry points return observably different tree states. Fix: drain/clear the
queue before returning (or stop `split` from enqueuing on a non-driver
path).

### A3. `merge()` evaluates every rejected pair twice per pass and borrows the wrong tolerance

`merge.hpp:73-114`:
- The done-list snapshot contains both children; a merged pair is skipped on
  the second visit by the `retired` guard, but a *rejected* pair passes all
  guards twice — double `substituteAxis` sweeps, double `maxCoeffDiff`, and
  `stats.rejected` double-counts (verified: one rejected pair → `rejected == 2`).
  A `if (sib < li) continue;` halves the work and fixes the stat.
- `diff <= T(crit.tol)` reuses the criterion's `tol` as an absolute
  coefficient-difference bound. For `TruncationCriterion` that is
  dimensionally coherent; for `NliCriterion`, `tol` (default 0.1) is a
  *dimensionless Jacobian-variation ratio* — as an absolute cap it merges
  pairs whose reconstructions differ by 100× the signal in normalized-unit
  problems, or never merges when states are O(1e6). Merge needs its own
  tolerance (or a criterion-supplied `mergeTol`).
- Asymmetry: acceptance tests and keeps only `fromL` — whether a pair merges
  can depend on which child was labeled "left" (`shouldSplit` is not
  Lipschitz in the coefficient bound), and the right half carries the full
  reconstruction error. Checking/storing the average is orientation-independent
  and halves the worst-case error.

### A4. Split at exactly `t1` silently finalizes an out-of-tolerance leaf

`driver.hpp:120-137`: the `atFinal` guard discards a split request firing at
`t1` (children with `tEntry == t1` would make `integrate` throw) and
finalizes the flagged parent — the final partition can contain a leaf the
criterion just declared unacceptable, with no diagnostic, and inconsistently
with a request 1 ulp earlier (which splits; the children then complete via
zero-step integrations). Since `sol.x.back()` at `t1` is a complete
integration, the better fallback is to `tax::ads::split` the payload and
finalize both children directly — no further integration needed. At minimum,
count/flag the suppression on the solution.

### A5. PZ canonical ordering is not total — parallel PZ runs are nondeterministic

`tree.hpp:195-206` (`canonicalizeDone`) and `solution.hpp:349-358` sort by
`domain.center(i)` lexicographically, asserting "disjoint domains have
distinct centers". For `PolynomialZonotope`, `center(dim)` is the constant
term, and a split along a factor that enters only through *even* powers
(possible — even-power monomials carry top-degree mass, so
`TruncationCriterion::splitDim` can pick such an axis) gives both children
identical constant terms on every axis. The comparator ties and `std::sort`
(unstable) orders siblings arbitrarily → `done()` order and `Partition` ids
differ between serial and parallel runs (and run to run). Box/Zonotope are
safe. Fix: tie-break (e.g. on a low-order odd coefficient or the arena
depth/split history), or document + `std::stable_sort` with a deterministic
pre-order. Untested: the PZ propagate test is serial-only.

### A6. `Zonotope::localize`/`contains` falsely reject covered points for dependent generators

`domain/zonotope.hpp:83-113`: `localize` returns the *minimum-L2-norm*
solution of `G·ξ = pt − c` (COD), and `contains` tests `‖ξ‖∞ ≤ 1 + tol` plus
the residual. The min-L2 solution is not the min-L∞ one, so for a
rank-deficient `G` with linearly dependent **non-zero** columns a point
genuinely in the set can localize outside the cube. Verified numerically:
`G = [[0.5, 1.0], [0.2, 0.4]]` (col₂ = 2·col₁), the point
`denormalize((1,1))` — in the set by construction — localizes to
`ξ = (0.6, 1.2)` and `contains` returns **false**. The docstring's claim
that "rank-deficient generator matrices … are handled" holds only for
*zeroed* columns (which is also the only case
`tests/domain/test_localize_evaluate.cpp:87-102` tests). Consequence:
`contains`, `AdsTree::locate/locateFactors`, and `AdsSolution::evaluate`
give an unsound "point not covered" answer on such domains. Fix: solve the
L∞-minimal factor (small LP) or document/assert that rank deficiency is
supported only via zero columns.

### A7. `NliCriterion` fires unconditionally on zero-linear rows with any curvature

`detail/nonlinearity_index.hpp:117-119`: a row with zero linear part and
*any* nonzero curvature mass — even 1e-300 — makes `nonlinearityIndex`
return `+inf`, so `NliCriterion::shouldSplit` fires until `maxDepth`:
unbounded over-splitting driven by numerically-zero coefficients. LOADS
defines the index with matrix norms over the whole Jacobian, which avoids
the per-row 0/0 pathology; alternatively floor the denominator or ignore
rows below a magnitude threshold. (The rest of the formula was verified
numerically: the variation bound dominates sampled Jacobian variations over
the cube, and the tail-block indexing matches graded-lex exactly.)

### A8. Work-pool exception-safety and spawn-failure gaps

`detail/work_pool.hpp:78-112`: only `process()` is inside the try block —
`apply()` (which allocates: `leafSol.resize`, arena `push_back`s, refine's
`WorkItem` fan-out holding dense DA states) and `pop()` run unguarded, so a
`std::bad_alloc` there escapes the worker thread and calls `std::terminate`,
contradicting the header's "first worker exception … rethrown on the calling
thread". Also `pool` holds joinable `std::thread`s (not the `jthread`s
CLAUDE.md claims): if `emplace_back` throws mid-spawn, the vector destructor
terminates. Fixes: widen the catch to the whole locked section (set
`first_err`, `stopping`, notify), and use `std::jthread`/join-guard. The CV
protocol itself is sound (no lost wakeup / premature exit / deadlock — the
throwing-RHS path was verified with a compiled 4-thread repro).

### A9. Smaller `tax::ads`/`tax::domain` items

- **Serial vs parallel retired payloads diverge** (`driver.hpp:150` vs
  `:205`): parallel `pop()` moves the payload out (retired parent left
  moved-from), serial reads by const-ref (parent keeps its full pre-split DA
  state alive — ~44 KB per interior node at P=6, M=D=6). Nothing in the
  library reads retired payloads, but `tree().leaf(i).payload` is public and
  answers differently per `num_threads`. Move in both paths (also fixes the
  serial memory retention).
- **`AdsTree::merge` missing `done` assert** (`tree.hpp:145-170`): comment
  claims the preconditions guarantee the children are in `doneList_`, but
  nothing asserts `done`; called with an active pair, `removeFromDone` uses
  `listPos_` values indexing `activeList_` and corrupts both lists in
  release builds. One assert closes it.
- **`Box::split` on a zero-width axis** produces identical children;
  merge's `splitOrdinate` comparison then ties and the inverse substitutions
  can be applied to the wrong children. Unreachable via the shipped criteria
  (zero-mass axes are never picked) but `split` is public and unguarded —
  `assert(halfWidth(dim) > 0)` makes the invariant explicit.
- **`NliCriterion` tol vs `T{tol}` narrowing** (`split_criteria.hpp:51`):
  braced init from `double` is ill-formed for `T = float` instantiations
  (dormant; everything is `double` today).
- **`detail/nonlinearity_index.hpp:46-66`**: `jacobianVariationBound` scans
  all monomials from k = 0 with a per-monomial degree loop; the graded-lex
  layout makes the `|α| ≥ 2` block contiguous (the `kLo` trick `axisMass`
  already uses). Runs per state row per accepted step — cheap mechanical win.
- **`zonotopeEnclosure` grows generators with `conservativeResize` per
  monomial** (`domain/enclosure.hpp:142-143`) — O(G²) copying; count first
  or reserve.
- **F2 recommendation stands**: heap-box queue payloads
  (`std::unique_ptr<State>`) or spawn workers with a larger stack, then
  remove the PZ example's `num_threads=1` caveat.

### Verified clean (so the next reviewer need not re-plough)

- All seven RK tableaus numerically consistent (except O1 metadata);
  `adaptive_rk_step` stage loop, Brent bracket maintenance, `dir_match`
  half-open no-double-fire semantics, PI/H211b formulas vs literature.
- `substituteAxis` split/merge algebra exact; degree guard unreachable
  (`aTotal − aDim + j ≤ aTotal`); merge shift signs correctly invert the
  split.
- `create` overloads (Box/Zonotope/PZ) build the correct identity DA states;
  center-consistency assert in place.
- `enclosure.hpp` math (interval hull, even-exponent zonotope enclosure,
  support function), `ellipsoid.hpp` (cover for any orthogonal R; exact L2
  hull), `reorient.hpp` composition/SVD/G·R consistency.
- `work_pool` CV/termination protocol; arena-reference discipline in both
  drivers at HEAD; sibling-only merge pairing with correct L/R orientation
  for Box and Zonotope; snapshot reconstruction tiles exactly-once at split
  boundaries (retired parents included, children skip inherited grid times);
  refine loop termination; `propagate.hpp` dispatch/forwarding/event wiring;
  determinism for Box/Zonotope runs.

---

## 4. Test-coverage gaps (both modules)

- **Zero `EXPECT_THROW` in the ODE suite** — none of the seven throw paths
  is exercised; no NaN-RHS test; no rejection-path test at all.
- No test reuses an `Integrator` (O2); GridEvent has one happy-path test —
  nothing near rejections, t0-coincident, sub-`min_step` spacing, or
  duplicate times; FixedStep never combined with events.
- No ADS test: places a grid time at/near a split time (A1); exercises a
  worker exception under `num_threads > 1` (A8); inspects the returned
  tree's queue after `refine()` (A2 — a one-line
  `EXPECT_TRUE(tree.empty())` would catch it); runs parallel over
  Zonotope/PZ (A5 needs exactly that); tests exact `MergeStats::rejected`
  counts (A3 hides behind `EXPECT_GE`); exercises `merge` with
  `NliCriterion` (where A3's unit-mixing lives); exercises `maxDepth`
  end-to-end through the driver; or covers degenerate (zero-width) geometry.
- The only end-to-end PZ propagation test
  (`tests/domain/test_polynomial_zonotope.cpp:210-218`) asserts just
  `done().size() >= 1` and finiteness of the constant terms — a
  wrong-but-finite pipeline passes. An accuracy assertion needs no
  `localize`: sample ξ, map through `pz.denormalize(ξ)`, compare the payload
  eval against a scalar reference propagation. (Run out-of-band during this
  review: 680 samples over 34 leaves, worst |pred − ref| = 1.2e-9 — the
  pipeline is correct, the assertion is just missing.)
- `tests/domain/test_localize_evaluate.cpp:87-102` covers rank deficiency
  only via a *zeroed* column — the dependent-columns regime of A6 is
  exactly the untested case.
- The 2026-07-02 review's suggestion of an in-repo MC containment ctest
  (E4/E8-style: "every MC truth lies inside the located leaf's enclosure")
  is now implementable — `tests/domain/test_enclosure.cpp` covers the
  enclosure primitives, but no end-to-end containment test over an ADS run
  exists yet.

## 5. Suggested landing order

1. **A1 + O7a** (one fix: treat stops within the landing tolerance as
   reached) — crashes healthy runs today.
2. **A2** queue drain + `EXPECT_TRUE(tree.empty())` test — one-liner.
3. **O1** Fehlberg metadata swap — two-line fix, improves every Fehlberg78
   run.
4. **O2/O3** GridEvent reset hook + full re-clamp on retry — silent data
   loss.
5. **A3** merge dedup + explicit merge tolerance; **A4** split-at-`t1`
   payload split.
6. **A6** Zonotope dependent-columns `contains` (unsound rejections —
   decide: L∞ solve or documented assert); **A7** NLI zero-linear-row floor.
7. **O4/O5/O6** controller edge cases (grow-on-reject cap, NaN, zero-error
   floor).
8. **A8** work-pool exception widening + `jthread`; then F2 (heap-boxed
   queue payloads) to unblock PZ × parallel.
9. **A5** PZ ordering tie-break before anyone relies on partition ids.
10. Test-gap backfill (§4), then the deferred P5/P6 example cleanups.
