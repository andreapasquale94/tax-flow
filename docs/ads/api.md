# ADS API Reference

All names live in `namespace tax::ads` unless noted. Internal helpers live in
`namespace tax::ads::detail`. The whole surface is brought in by the umbrella
header:

```cpp
#include <tax/ads.hpp>
#include <tax/ode.hpp>   // the steppers ADS drives
```

Throughout, **`M`** is the number of box (expansion) variables, **`D`** the
state dimension, **`P`** the Taylor truncation order, and the per-leaf
`Payload` is the DA flow map `Eigen::Matrix<TaylorExpansion<T, P, M>, D, 1>`.

---

## Domain interface

The IC-set primitives (`Box`, `Zonotope`, `PolynomialZonotope`), the
`Domain` / `LocatableDomain` concept tiers, `create` (domain → identity DA
state), the enclosure/query layer and the reorientation helpers live in the
**[`tax::domain` module](../domain/index.md)** (`#include <tax/domain.hpp>`,
re-exported by `<tax/ads.hpp>`). ADS is generic over any `Domain`; point
location and `merge` additionally require `LocatableDomain`.

---

## Leaf

```cpp
template <class Payload, tax::domain::Domain Domain>
struct Leaf {
    Domain  domain{};
    Payload payload{};        // DA flow map valid on domain
    int  depth      = 0;      // number of splits to reach this leaf
    bool done       = false;  // finalized (propagated to t_final)
    bool retired    = false;  // parent of an active/done sibling pair
    int  parentIdx  = -1;
    int  siblingIdx = -1;
    int  splitDim   = -1;
    T    tEntry     = T{0};    // time at which this leaf's interval starts
};
```

---

## AdsTree

```cpp
template <class Payload, tax::domain::Domain Domain>
class AdsTree {
public:
    // T and M are recovered from the Domain via domain_traits.
    int  init(Domain domain, Payload payload, T tEntry = T{0});  // add root
    std::pair<int, int> split(int idx, int dim,
                              Payload left, Payload right, T tEntry);
    void finalize(int idx);                                      // mark done
    void merge(int leftIdx, int rightIdx, Payload merged);       // collapse a pair

    const Leaf<Payload, Domain>& leaf(int idx) const noexcept;
    std::span<const int> active() const noexcept;               // not-yet-finalized
    std::span<const int> done()   const noexcept;               // finalized leaves
    std::span<const int> roots()  const noexcept;
    void canonicalizeDone();                                 // sort done by centre

    // Point lookup (requires LocatableDomain):
    std::optional<int>      locate(pt) const;         // first leaf containing pt
    std::optional<Location> locateFactors(pt) const;  // {leaf idx, exact ξ}
};
```

A **leaf-only arena tree**: every record is a `Leaf`; the shape is recovered
from `parentIdx` / `siblingIdx`. A split retires the parent in place and
appends two children; a merge revives the parent. `canonicalizeDone()` sorts
the finalized leaves by domain centre so parallel and serial runs agree and
output is reproducible. Drivers populate the tree; users normally only read
`done()`, `leaf(idx)` (by index), `locate(pt)` and `locateFactors(pt)`
(point lookup with exact factor recovery; both require `LocatableDomain`, so
a `PolynomialZonotope` tree refuses them at compile time).

---

## DA state

```cpp
// Identity state on a domain (tax::domain::create; Box/Zonotope/PZ overloads).
template <int P, int M, class Storage = tax::storage::Dense, class T, int D>
State create(const Domain& domain, const Eigen::Matrix<T, D, 1>& x0);

// Re-identify a state on the two halves of its domain along factor `dim`.
template <class T, int N, int M, class Storage, int D>
std::pair<State, State> split(const State& state, int dim);
```

`create` (in `tax::domain`) seeds the propagation; `split` (in `tax::ads`)
carries an existing map onto the children via `ξ_dim → ±0.5 + 0.5 ξ'_dim`.

---

## Split criteria (`propagate`)

```cpp
template <class C, class State>
concept SplitCriterion = requires(C c, const State& x, int depth) {
    { c.shouldSplit(x, depth) } -> std::convertible_to<bool>;
    { c.splitDim(x) }          -> std::convertible_to<int>;
};
```

```cpp
struct TruncationCriterion {   // Wittig 2015
    double tol = 1e-6;         // split when Σ_{|α|=P} |c_α| > tol
    int maxDepth = 30;
};

struct NliCriterion {          // LOADS (Losacco/Fossà/Armellin 2024)
    double tol = 0.1;          // split when the nonlinearity index > tol
    int maxDepth = 30;
};
```

Both pick the split direction as the coordinate contributing the most of the
measured quantity, and stop once `depth >= maxDepth`.

---

## Quality criteria (`refine`)

```cpp
template <class C, class State>
concept QualityCriterion =
    requires(C c, const State& p, std::span<const State> ch,
             std::span<const int> dims, int depth) {
        { c.acceptable(p, ch, dims, depth) } -> std::convertible_to<bool>;
        { c.splitDim(p) }                    -> std::convertible_to<int>;
        { c.maxDepth }                       -> std::convertible_to<int>;
    };
```

`acceptable(parent, children, dims, depth)` returns **true to stop** (the parent
is good enough). `children` are the `2^k` maps from splitting `dims`
(`k = dims.size()`); child `i` is the sub-box whose offset along `dims[j]` is
"+" when bit `j` of `i` is set. See [Refinement](refine.md) for the indices.

```cpp
struct CoefficientMatchCriterion {   // dimension-free
    double tol = 1e-3;               // relative max coefficient mismatch
    int maxDepth = 8;
};

struct VolumeRatioCriterion {        // geometric, any dimension
    double tol = 1e-6;               // |V(parent) / Σ V(child) − 1|
    int maxDepth = 8;
    std::vector<int> axes{};         // active input axes; empty ⇒ all M
    int nQuad = 8;                   // quadrature points per active axis
};
```

---

## propagate

```cpp
template <int P, class Method, class Criterion, class F, class DomainArg, int D>
auto propagate(Method, Criterion crit, F&& rhs,
               const DomainArg& ic_domain,
               const Eigen::Matrix<domain_scalar_t<DomainArg>, D, 1>& ic_center,
               const domain_scalar_t<DomainArg>& t0,
               const domain_scalar_t<DomainArg>& t1,
               tax::ode::IntegratorConfig<domain_scalar_t<DomainArg>> cfg = {},
               int num_threads = 1)
    -> AdsSolution</*Stepper*/, DomainArg>;
```

Classic in-flight ADS. `DomainArg` is any `Domain` type (`Box`, `Zonotope`,
`PolynomialZonotope`, …); `domain_traits` deduces the scalar `T` and factor
dimension `M`. `Method` is an `ode::methods::` tag (`Verner89{}`, `Taylor<N>{}`,
…) selecting the stepper. `num_threads > 1` runs independent boxes on a `jthread`
pool. Returns `AdsSolution`, which holds the `AdsTree` and per-leaf ODE
`Solution` objects; access the tree via `sol.tree()`, and evaluate the
piecewise-polynomial flow map at a physical IC point with `sol.evaluate(pt)`
(exact leaf location + factor recovery + payload evaluation; requires a
`LocatableDomain`). Snapshot partitions expose the same query as
`Partition::evaluate(pt)`.

User events are *not* forwarded — instantiate [`AdsDriver`](#drivers) directly
if you need them.

---

## refine

```cpp
template <int P, class Method, class Quality, class F, class T, int M, int D>
auto refine(Method, Quality quality, F&& rhs,
            const Box<T, M>& ic_box,
            const Eigen::Matrix<T, D, 1>& ic_center,
            const T& t0, const T& t1,
            tax::ode::IntegratorConfig<T> cfg = {},
            int num_threads = 1,
            int split_dirs = 1)
    -> AdsTree<Eigen::Matrix<TaylorExpansion<T, P, M>, D, 1>, Box<T, M>>;
```

Propagate-then-assess refinement. `split_dirs` (default 1, binary) splits the
top-`split_dirs` directions at once into `2^split_dirs` children. The partition
is **independent of `num_threads`**. See [Refinement](refine.md).

---

## Drivers

For full control (user events, reused configuration) instantiate the driver
classes the convenience functions wrap:

```cpp
template <class Stepper, class Criterion> class AdsDriver;         // propagate
template <class Stepper, class Quality>   class AdsRefineDriver;   // refine

AdsDriver(Criterion, Cfg, ExtraEvt extras = {}, int num_threads = 1);
AdsRefineDriver(Quality, Cfg, int num_threads = 1, int split_dirs = 1);

Tree run(F&& rhs, const DomainT& ic_domain, const Eigen::Matrix<T, D, 1>& ic_center, T t0, T t1);
```

`AdsDriver`'s `extras` are extra `std::shared_ptr<tax::ode::Event<State,T>>`
appended to every box's integration (deep-copied per leaf for the parallel
driver via `clone()`).

---

## Merging

```cpp
struct MergeStats { int passes = 0; int merges = 0; int rejected = 0; };

template <class Payload, class Domain, class Criterion>
    requires tax::domain::LocatableDomain<Domain>
MergeStats merge(AdsTree<Payload, Domain>& tree, Criterion crit);
```

A post-pass that collapses sibling pairs whose reconstructed parent still
satisfies `crit` (using `TruncationCriterion`/`NliCriterion`), repeating until
no pair merges. Useful to trim a tree that was split more finely than the final
shape needs.
