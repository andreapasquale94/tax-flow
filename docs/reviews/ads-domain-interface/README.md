# Review — `feature/ads-domain-interface`

*Scope: the 26 commits on top of `main` (Domain concept tiers, `Zonotope`,
`PolynomialZonotope`, `reorient`, domain-generic tree/driver/merge), the
examples/tutorials, and an independent Monte-Carlo validation of the enclosure
construction. Reviewed 2026-07-02.*

**Verdict.** The domain interface is well designed and the math is right. The
two-tier `Domain` / `LocatableDomain` split is the correct cut (PZ genuinely
cannot support exact location), the factor-coordinate insight — that all
payload substitutions are domain-agnostic — is exploited cleanly, and the docs
are unusually honest about limitations. An independent Monte-Carlo oracle
(11/11 experiments, see §3) confirms the split/merge substitutions are exact,
domains tile correctly under splitting, and end-to-end ADS enclosures contain
the true flow. The main gaps are (a) a handful of API contracts that are
looser than the docs claim (`locate` on PZ trees, `create`'s center
convention), and (b) a missing *enclosure/query* layer that has forced ~10
copies of the same geometry helpers into the examples — promotion candidates
are listed in §4.

---

## 1. Implementation findings

Ordered by importance. File references are to the feature-branch tree.

### F1. `AdsTree::locate` silently degrades on `PolynomialZonotope` trees

`domain.hpp:8-10` documents `LocatableDomain` as the tier with "EXACT point
location (contains)", and `docs/ads/polynomial_zonotope.md` says exact
membership needs a Box/Zonotope. But:

- the `LocatableDomain` concept (`domain.hpp:39-42`) only requires
  `splitOrdinate` — `contains` is not part of either concept;
- `AdsTree::locate` (`tree.hpp:203-211`) is **not** constrained to
  `LocatableDomain`; it calls `box.contains(pt)` on whatever domain it holds.

Since `PolynomialZonotope::contains` is a conservative interval-hull test,
`tree.locate(pt)` on a PZ tree compiles and returns the **first leaf whose
hull contains the point** — hulls of sibling PZ leaves overlap, so this is
ambiguous and can return a leaf whose actual curved subdomain does not contain
`pt`, with no diagnostic. Callers who then evaluate `leaf.payload` at
hull-recovered coordinates get silently wrong predictions.

**Recommendation:** put `contains` into the `LocatableDomain` concept
(matching the doc comment), and constrain `locate` with
`requires LocatableDomain<Domain>` so PZ trees fail loudly at compile time —
same pattern already used by `merge`. If approximate location on PZ trees is
wanted, expose it under an explicit name (`locateHull`).

### F2. Parallel-driver crash with large DA states (gates the PZ path)

`examples/two_body/polynomial_zonotope.cpp:68-71` works around it:

> `num_threads=1`: the parallel driver may SIGBUS on large DA states; the
> single-threaded path is always safe.

A workaround comment in an example is not a fix; this gates `num_threads > 1`
for exactly the workloads the domain interface targets. Likely mechanism (to
verify): the dense `TaylorExpansion` core is allocation-free, so `State` is
inline storage. The parallel plumbing keeps several `State` objects **by
value on worker stacks** — `AdsDriver::driveParallel`'s `Item` (payload) +
`LeafVerdict` (left/right/leafSol), and `work_pool.hpp:83`'s `Verdict
verdict{}` — several full states per loop iteration. Non-main-thread stacks
default to 512 KiB on macOS (where a stack overflow reports as SIGBUS), so
e.g. P=8, M=6, D=6 states (~144 KiB each) overflow while the main thread
(8 MiB) survives — matching "parallel crashes, serial is safe".

**Recommendation:** heap-box the states in the queue plumbing
(`std::unique_ptr<State>` or small structs holding vectors) or spawn workers
with an explicit larger stack; then remove the example caveat. Worth fixing
before the branch merges, since PZ + parallel is an advertised combination.

### F3. `create(domain, x0)` center convention is a silent-corruption footgun

All three `create` overloads (`da_state.hpp:92/123`,
`polynomial_zonotope.hpp:119`) take the payload center from `x0` and ignore
the domain's own center (`Box::center`, `Zonotope::center`, PZ constant
terms — the PZ overload even *overwrites* the constant term). The tree
geometry, however, uses the domain's center for `locate`,
`canonicalizeDone`, `splitOrdinate` and snapshot ordering. If a caller passes
`ic_center` ≠ domain center, the payload and the geometry disagree: `locate`
recovers factor coordinates from geometry that the polynomial was never
built on. Nothing checks this.

**Recommendation:** in `AdsDriver::run` (or in `create`), assert
`x0.head(M)` matches the domain center to a tolerance — or remove the
redundancy (derive the head of `x0` from the domain and pass only the trailing
components). At minimum document that `x0.head(M)` **must** equal the domain
center.

### F4. The `Domain` concept under-states its real requirements

`domain.hpp:7` says a domain provides "split / denormalize / center /
create", but the concept checks only `center(i)` and `split(dim)`.
`AdsDriver::run` additionally needs an `ads::create(domain, x0)` overload,
`AdsSolution`/`merge` use `splitOrdinate`, and user-side evaluation needs
`denormalize`. A third-party domain that models the concept as written
explodes with deep instantiation errors inside `run`.

**Recommendation:** add `denormalize` to `Domain`, add
`contains` to `LocatableDomain` (F1), and either add a `DomainFactory`-style
requirement for `create` or a `static_assert` in `run` with a clear message.

### F5. Degenerate `Zonotope` generators are unguarded

`Zonotope::contains` (`zonotope.hpp:73-83`) LU-solves `G⁻¹(pt−center)`;
`partialPivLu` on a singular/near-singular `G` silently produces garbage
factors, so `contains`/`locate` return arbitrary results.
`docs/ads/zonotope.md:149-153` acknowledges "not stressed on near-degenerate
matrices". A rank/conditioning guard (or `FullPivLU` + `isInvertible`) with a
documented failure mode would make this a diagnosable error instead of wrong
answers. Same applies to the examples' `predictXY`-style LU solves.

### F6. API-shape nits

- `AdsTree<Payload, M, T, Domain>`: `M` and `T` are redundant — both are
  recoverable as `domain_traits<Domain>::dim/scalar`. `AdsTree<Payload,
  Domain>` (and same for `Leaf`, `AdsSolution`) would remove a
  keep-in-sync burden. Worth doing before the interface calcifies.
- `Leaf::box` / `AdsTree::BoxT` / `AdsDriver::BoxT` name a generic domain
  `box`. Source-compat is noted in `leaf.hpp:20-22`, but this branch already
  renames things (`RefineDriver`→`AdsRefineDriver`, `leaf(pt)`→`locate`) —
  renaming to `domain`/`DomainT` with a deprecated alias fits the theme.
- `Box::contains` is exact-boundary while `Zonotope::contains` takes
  `tol = 1e-12`; boundary points can be located in a Box tree but rejected in
  an equivalent `Zonotope::fromBox` tree (or vice versa). Align the two.
- `refine()`/`AdsRefineDriver` remain Box-only (documented). Fine to defer,
  but the M/T cleanup above would make the later generalization mechanical.

### What was checked and found sound

- `detail::substituteAxis` split/merge algebra — exact (validated to 1e-15,
  §3 E1), including the degree guard and the 2^N amplification asymmetry on
  merge.
- `Zonotope::split/splitOrdinate` — children tile the parent; ordering
  `L < R` holds (E2); `splitOrdinate` difference is `‖g‖²/2 > 0` so sibling
  ordering in `merge` is well-defined for any nonzero generator.
- `merge()` — pass-until-quiescent structure, sibling-pair discovery over the
  done list snapshot, `crit.shouldSplit(fromL, parent_depth)` re-check, and
  MergeStats accounting are all coherent; MC revalidation after merging
  confirms accuracy degrades only to the merge tolerance (E9).
- `work_pool::parallelDrive` — termination (queue drained ∧ nothing in
  flight), the unified notify policy, and first-exception propagation are
  correct under the stated lock discipline; `empty()`/`pop()`/`apply()` are
  only called under the lock as documented.
- `AdsSolution::snapshots` — including retired parents is required and
  correct (children's `GridEvent` clones skip grid times ≤ their entry time
  via the cursor scan in `grid_event.hpp:35-56`, so no duplicates); `x.front()`
  is the initial state even with `save_steps=false`, so `rootPartition` is
  safe.
- `SplitEvent` at accepted-step boundaries + the `atFinal` guard in
  `stepLeaf` (a split that fires exactly at `t1` finalizes instead of queuing
  empty children).
- LOADS `nonlinearityIndex` (∞ when the linear part vanishes but nonlinear
  mass remains) and the unweighted-vs-axis-weighted mass distinction in
  `TruncationCriterion` (the comment at `split_criteria.hpp:88-91` preventing
  a silent `N·tol` rescale is exactly right).

---

## 2. Independent Monte-Carlo validation

The `tax` core is not reachable from this review environment (repo scope), so
the C++ suite could not be compiled here. Instead, `mc_validate.py` (this
directory) transcribes the branch's algorithms into NumPy — a dense
degree-6, 2-factor polynomial algebra plus line-by-line ports of
`substituteAxis`, `create(Box/Zonotope/PZ)`, the three domains, the
`TruncationCriterion`, the serial driver loop (split-at-step-boundary
semantics incl. the `atFinal` guard), `merge()`, and the reorientation
primitives — and checks them against brute-force Monte Carlo with
`scipy.solve_ivp` (DOP853, tol 1e-12) as ground truth. Because it shares no
code with the C++, it is an independent oracle for the *algorithms* (not for
C++-specific defects like F2).

| # | Experiment | Result |
|---|------------|--------|
| E1 | split substitution exact on 200 random polynomials; merge substitution inverts it | PASS (1e-15 / 4e-15) |
| E2 | Zonotope children tile the parent (10 000 MC points); `splitOrdinate` orders L<R | PASS (0 escapees) |
| E3 | PZ split reproduces the parent image exactly; `contains` has no false negatives | PASS |
| E4a | end-to-end ADS, Duffing, box IC: MC pointwise error vs tol=1e-6; interval-hull containment | PASS (max 2.9e-8, 0 misses) |
| E4b | same, 45°-rotated thin **Zonotope** IC (the docs' case) | PASS (max 5.0e-8, 0 misses) |
| E4c | same, curved **PZ** IC (the example's ξ₁² bend): union-of-leaf-hulls coverage of 300 truths | PASS (0 outside) |
| E4d | end-to-end two-body (μ=1), box IC, tol=1e-7 | PASS (max 9.6e-9, 0 misses) |
| E8 | stress: e=0.5 two-body arc through apoapsis, 12-leaf tree, 250 MC | PASS (max 5.1e-7, 0 hull misses) |
| E9 | `merge()` on E8's tree (accept path: 10 merges → 2 leaves), MC revalidation | PASS (err ≤ merge tol regime, 0 hull misses) |
| E5 | ellipsoid-covering recipe: `L·V·cube ⊇ ellipsoid` for random orthogonal V; L2 row-norm box is the ellipsoid's exact interval hull | PASS |
| E6 | zonotope enclosure of a polynomial image via the even-exponent construction (§4 P3): LP containment of 6 000 samples; tighter than the interval hull | PASS |
| E7 | `reorientZonotope`/`create` linear parts commute with R; generic orthogonal R **changes** the represented set (the documented cube-invariance caveat, ~17 % of points leave) | PASS |

Notes for the maintainers:

- **The in-repo examples never test containment** — `validation.cpp`,
  `two_body_mc.cpp`, `refine.cpp`, `missed_thrust*` all measure pointwise
  flow-map *error*, which cannot catch an unsound enclosure that is accurate
  at sampled points but too small elsewhere. E4/E8-style checks ("every MC
  truth lies inside the located leaf's enclosure") belong in `tests/ads/` as
  a ctest once the enclosure ops of §4 exist in the library.
- Reproduce with `python3 mc_validate.py` (needs numpy + scipy; ~6 min).
  Seeded (`20260702`), deterministic.

## 3. Literature cross-check

- The `Zonotope` here is a **parallelotope** (square generator matrix); the
  header/docs say so and correctly frame it as the unconstrained member of
  the constrained-zonotope family (Scott et al. 2016; Kochdumper & Althoff,
  constrained polynomial zonotopes). No issues.
- `PolynomialZonotope` matches the *dependent-factor* part of sparse
  polynomial zonotopes (Kochdumper & Althoff, arXiv:1901.01780): all
  generators share the factor cube, no independent-generator part. That is
  the right choice for ADS (the DA payload is exactly such an object), worth
  one sentence in the docs to orient CORA users.
- The zonotope-enclosure-of-a-PZ operation proposed in §4 (P3) is the
  standard construction (ibid., the zonotope-enclosure proposition): a
  monomial ξ^α with all-even α has range [0,1] over the cube, so its
  coefficient contributes `c/2` to the center and `c/2` as a generator; any
  other monomial contributes its coefficient as a generator. Validated
  empirically in E6.
- Interval hull of a zonotope = center ± Σ|generator components| (L1 row
  norm; Althoff's PhD thesis, zonotope preliminaries) — matches
  `icZonotopeBoundingBox`. The **L2** row norm used in
  `zonotope_adaptive.cpp` / `two_body_mc.cpp` is a different object — the
  exact interval hull of the *ellipsoid* `{L·u : ‖u‖₂ ≤ 1}` — and is correct
  where used, but the two must not be conflated when promoted (P2).
- Split criteria match the cited papers: `TruncationCriterion` is Wittig
  2015's order-N mass test; `NliCriterion` is the LOADS Jacobian-variation
  index (Losacco/Fossà/Armellin 2024).

## 4. Examples review — what should move into the library

The examples are good tutorials, but they contain ~10 re-implementations of
the same enclosure/query geometry, several of which duplicate code that
already exists *inside* the library private paths. The theme: **the library
models and propagates domains but offers no query/measurement layer on the
result**, so every validating example rebuilds one. Proposed promotions, in
priority order:

### P1 — Factor recovery + tree evaluation (biggest duplication)

The single most copied helper is "physical point → leaf-local ξ → evaluate
flow map":

- `two_body/validation.cpp:61-70` (`toLocal`, Box, zero-width-guarded)
- `two_body/refine.cpp:240-246` (inline copy of the same)
- `zonotope/two_body_mc.cpp:109-134` (`predictXY`: LU solve per leaf —
  re-deriving what `Zonotope::contains` computes and throws away)
- `reachability/reachability.cpp:37-66` (inline per-axis scan against a
  `Partition`)

Add to the library:

```cpp
// domains: exact factor recovery (LocatableDomain tier)
VecNT<M,T> Box::localize(pt);        // (pt - center) / halfWidth, 0-width guarded
VecNT<M,T> Zonotope::localize(pt);   // LU solve, shared with contains()
// tree/solution: locate + evaluate in one call
std::optional<Eval> AdsTree::evaluate(pt);       // {leaf idx, xi, payload eval}
std::optional<Eval> Partition::evaluate(pt);
```

`contains` becomes `localize` + a norm check, deleting the duplicated LU.

### P2 — Interval hulls and coverings (three sites, two formulas)

- `Zonotope::intervalHull() -> Box` — L1 row norms
  (`two_body/common.hpp:128-134`).
- `PolynomialZonotope::intervalHull() -> Box` — `c₀ ± Σ|coeff|`; the radius
  sum already lives inside `PolynomialZonotope::contains`
  (`zonotope/representations.cpp:81-94` re-implements it as `boxHull`).
- Ellipsoid covering helpers used by the adaptive recipe
  (`zonotope_adaptive.cpp:107-130`, `two_body_mc.cpp:317-332`):
  `Zonotope::coveringEllipsoid(center, L)` (generators `L`, MC-validated
  E5) and `Box ellipsoidIntervalHull(center, L)` (L2 row norms). Distinct
  names prevent the L1/L2 mix-up.

### P3 — Enclosure of a leaf image (the missing layer)

A leaf's payload over its factor cube *is* a polynomial zonotope, so the
library can provide enclosures of the **image** (reachable set slice) with no
new representation:

```cpp
// interval hull of the image of the cube under a DA state (per component set)
Box<T, K> intervalHull(const State& x, span<const int> comps);
// zonotope (non-square, K x G) over-approximation — even-exponent construction
ZonotopeEnc<T, K> zonotopeEnclosure(const State& x, span<const int> comps);
// exact PZ view of a leaf image
PolynomialZonotope<...> imageOf(const State& x);
```

Validated in E6 (containment via LP over 6 000 samples, strictly tighter
than the interval hull). This is what makes ADS output *consumable* as a
reachability result rather than raw polynomials — and it is what the MC
containment test (§2) needs. Note the enclosure zonotope is a general
zonotope (many generators), not the square-`G` domain `Zonotope`; either a
small separate `ZonotopeEnc` value type or generalizing `Zonotope` to
rectangular `G` (only `contains/split` need squareness).

### P4 — Constructors / frame utilities

- `Zonotope::oriented(center, R, halfWidths)` — the "rotation × diagonal"
  pattern (`two_body/common.hpp:110-123`).
- Reuse of `linearPart`: `zonotope/representations.cpp:57-77`'s `frameOf`
  re-implements it; a `Zonotope fromLinearPart(state)` (center = constant
  terms, G = degree-1 block) would delete `frameOf`, `embed`/`active`/
  `zonoFrom` (`zonotope_adaptive.cpp:45-69`, `two_body_mc.cpp:55-90`) given
  M-factor sub-block helpers.

### P5 — 2D projection & polygon utilities

`evalPolygon`/`boxPolygon` (`examples/common/output.hpp:100-194`) and
`polygonArea` (`refine.cpp:87-93`) are generic geometry re-inlined at least
four more times (`representations.cpp:98-108`, `wsb/ads.cpp:132-139`,
`wsb/taylor.cpp:132`, `validation.cpp:304-313`). Promote as e.g.
`tax::ads::project2d(state, ix, iy, boundarySamples)` + shoelace area; keep
JSON/banner I/O in `examples/common`.

### P6 — Example hygiene (stay in examples, but dedupe)

- CR3BP `rhs` appears verbatim ×3 (`three_body/common.hpp:55-80`,
  `wsb/common.hpp:67-86`, `wsb/wsb_search.cpp:72-91`) → one
  `examples/common/cr3bp.hpp`.
- `wsb/ads.cpp` re-implements `adsThreads` and the JSON dump that
  `examples/common/output.hpp` already provides.
- `two_body/validation.cpp:177-191`: `posErrors` result computed then
  discarded and re-walked — dead code.
- `two_body/refine.cpp:71-84` `leafInit` duplicates library
  `create<P,M>(box, x0)`.
- `missed_thrust` vs `missed_thrust_onoff` are near-identical trees (~600
  duplicated lines) — parameterize one example.

With P1–P3 in place, `validation.cpp`, `two_body_mc.cpp` and
`reachability.cpp` shrink to problem setup + plotting, which is what
tutorials should be.

---

## 5. Suggested landing order

1. F1 (concept/`locate` tightening) + F4 — small, compile-time, prevents
   silent misuse of the new PZ tier.
2. F3 center-consistency assert — one line, kills a silent-corruption class.
3. P1 + P2 + P3 enclosure/query layer (each is small; P1/P2 mostly move
   existing code), then rewrite the examples on top and add the ctest MC
   containment test (§2).
4. F2 parallel-driver root cause (unblocks PZ × parallel).
5. F5/F6 polish with the API rename, before external users pin the names.
