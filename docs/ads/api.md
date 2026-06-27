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

```cpp
// include/tax/ads/domains/domain.hpp
template <class D>
struct domain_traits;          // specialised by each primitive

template <class D>
concept Domain = requires(const D d, int dim) {
    typename domain_traits<D>::scalar;
    { domain_traits<D>::dim } -> std::convertible_to<int>;
    { d.center(0) };
    { d.split(dim) } -> std::same_as<std::pair<D, D>>;
};

template <class D>
concept LocatableDomain = Domain<D> && requires(const D d, int dim) {
    { d.splitOrdinate(dim) } -> std::convertible_to<domain_scalar_t<D>>;
};
```

`Domain` is the minimum requirement for `propagate` and `AdsDriver`.
`LocatableDomain` additionally requires `splitOrdinate` (exact affine sibling
ordering) and is required by `merge`. `PolynomialZonotope` models `Domain` but
not `LocatableDomain`; `Box` and `Zonotope` model both.

---

## Box

```cpp
template <class T, int M>
struct Box {
    tax::la::VecNT<M, T> center    = tax::la::VecNT<M, T>::Zero();
    tax::la::VecNT<M, T> halfWidth = tax::la::VecNT<M, T>::Zero();

    bool contains(const Eigen::MatrixBase<Derived>& pt) const noexcept;
    std::pair<Box, Box> split(int dim) const noexcept;            // halve along dim
    T splitOrdinate(int dim) const noexcept;                      // center(dim)
    tax::la::VecNT<M, T> denormalize(const Eigen::MatrixBase<Derived>& d) const noexcept;
};
```

An axis-aligned hyperrectangle of initial conditions. `split(dim)` returns the
left/right halves (centre shifted by `±halfWidth(dim)/2`, that half-width
halved). `denormalize(d)` maps a normalised displacement `d ∈ [-1,1]^M` to
`center + halfWidth ⊙ d`. A zero `halfWidth` component pins that axis.

```cpp
Box<double, 4> b{ {0.5, 0, 0, 1.7}, {0, 1e-3, 0, 1e-3} };   // varies axes 1 and 3
```

---

## Zonotope

```cpp
// include/tax/ads/domains/zonotope.hpp
template <class T, int M>
struct Zonotope {
    tax::la::VecNT<M, T>  center     = tax::la::VecNT<M, T>::Zero();
    tax::la::MatNT<M, T>  generators = tax::la::MatNT<M, T>::Zero();   // M×M

    static Zonotope axisAligned(const VecNT<M,T>& center, const VecNT<M,T>& halfWidth);
    static Zonotope fromBox(const Box<T,M>& b);

    bool contains(const Eigen::MatrixBase<Derived>& pt) const;   // solves G⁻¹(pt-c)
    std::pair<Zonotope, Zonotope> split(int dim) const;          // bisect generator dim
    T splitOrdinate(int dim) const;                              // center(dim)
    tax::la::VecNT<M, T> denormalize(const Eigen::MatrixBase<Derived>& xi) const;
};
```

A parallelotope: `{ center + G · ξ : ξ ∈ [-1,1]^M }`. The oriented generator
matrix `G` lets a single leaf wrap a correlated or rotated IC set with a tighter
fit than the axis-aligned bounding box — see [Zonotope](zonotope.md).
`Zonotope` models `LocatableDomain`; `merge` works on zonotope trees.

---

## PolynomialZonotope

```cpp
// include/tax/ads/domains/polynomial_zonotope.hpp
template <class T, int N, int M, class Storage = tax::storage::Dense>
struct PolynomialZonotope {
    using TE = TaylorExpansion<T, IsotropicScheme<N,M>, Storage>;
    std::array<TE, M> value{};   // value[i](ξ) = i-th physical coordinate

    static PolynomialZonotope fromBox(const Box<T,M>& b);

    T center(int dim) const noexcept;                              // constant term of value[dim]
    tax::la::VecNT<M,T> denormalize(const Eigen::MatrixBase<Derived>& xi) const;
    bool contains(const Eigen::MatrixBase<Derived>& pt, T tol = 1e-12) const;  // conservative
    std::pair<PolynomialZonotope, PolynomialZonotope> split(int dim) const;
    // splitOrdinate NOT provided → models Domain but not LocatableDomain
};
```

The IC set is a polynomial image of the cube. Degree-1 special cases recover
`Box` (`fromBox`) or `Zonotope` (dense linear `value`). `contains` is a
conservative over-approximation. `merge` is disabled at compile time for
polynomial-zonotope trees. See [PolynomialZonotope](polynomial_zonotope.md).

---

## Reorient helpers

```cpp
// include/tax/ads/domains/reorient.hpp
// Compose the flow map with a linear factor change  y(η) = x(R·η):
reorientState(state, R) -> State;
// Extract the D×M linear part  A = ∂x/∂ξ|₀  (the local STM):
linearPart(state)       -> Eigen::Matrix<T, D, M>;
// V from SVD(A) such that A·V has orthogonal columns:
flowAlignedRotation(A)  -> Eigen::Matrix<T, M, M>;
// Keep generators aligned: G → G·R
reorientZonotope(z, R)  -> Zonotope<T, M>;
```

Used to pick the factor frame from the flow before propagation (see
[Zonotope — adaptive orientation](zonotope.md#adaptive-orientation-aligning-the-frame-to-the-flow)).

---

## Leaf

```cpp
template <class Payload, int M, class T = double, class Domain = Box<T, M>>
struct Leaf {
    Domain  box{};
    Payload payload{};        // DA flow map valid on box
    int  depth      = 0;      // number of splits to reach this leaf
    bool done       = false;  // finalized (propagated to t_final)
    bool retired    = false;  // parent of an active/done sibling pair
    int  parentIdx  = -1;
    int  siblingIdx = -1;
    int  splitDim   = -1;
    T    splitValue = T{0};
    T    tEntry     = T{0};    // time at which this leaf's interval starts
};
```

---

## AdsTree

```cpp
template <class Payload, int M, class T = double, class Domain = Box<T, M>>
class AdsTree {
public:
    int  init(Domain box, Payload payload, T tEntry = T{0});    // add root
    std::pair<int, int> split(int idx, int dim,
                              Payload left, Payload right, T tEntry);
    void finalize(int idx);                                      // mark done
    void merge(int leftIdx, int rightIdx, Payload merged);       // collapse a pair

    const Leaf<Payload, M, T, Domain>& leaf(int idx) const noexcept;
    std::span<const int> active() const noexcept;               // not-yet-finalized
    std::span<const int> done()   const noexcept;               // finalized leaves
    std::span<const int> roots()  const noexcept;
    void canonicalizeDone();                                     // sort done by box centre

    std::optional<int> locate(const Eigen::MatrixBase<Derived>& pt) const;  // point lookup
};
```

A **leaf-only arena tree**: every record is a `Leaf`; the shape is recovered
from `parentIdx` / `siblingIdx`. A split retires the parent in place and
appends two children; a merge revives the parent. `canonicalizeDone()` sorts
the finalized leaves by domain centre so parallel and serial runs agree and
output is reproducible. Drivers populate the tree; users normally only read
`done()`, `leaf(idx)` (by index) and `locate(pt)` (point lookup, returns the
index of the first finalized leaf whose domain contains `pt`, or `nullopt`).

---

## DA state

```cpp
// Identity state on a box: comp_i = x0_i + halfWidth_i · ξ_i  (ξ ∈ [-1,1]^M).
template <int P, int M, class Storage = tax::storage::Dense, class T, int D>
Eigen::Matrix<TaylorExpansion<T, P, M, Storage>, D, 1>
create(const Box<T, M>& box, const Eigen::Matrix<T, D, 1>& x0);

// Re-identify a state on the two halves of its parent box along `dim`.
template <class T, int N, int M, class Storage, int D>
std::pair<State, State> split(const State& state, const Box<T, M>& parent_box, int dim);
```

`create` seeds the propagation; `split` carries an existing map onto child
sub-boxes via the substitution `ξ_dim → ±0.5 + 0.5 ξ'_dim`.

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
    -> AdsSolution</*Stepper*/, domain_dim_v<DomainArg>, DomainArg>;
```

Classic in-flight ADS. `DomainArg` is any `Domain` type (`Box`, `Zonotope`,
`PolynomialZonotope`, …); `domain_traits` deduces the scalar `T` and factor
dimension `M`. `Method` is an `ode::methods::` tag (`Verner89{}`, `Taylor<N>{}`,
…) selecting the stepper. `num_threads > 1` runs independent boxes on a `jthread`
pool. Returns `AdsSolution`, which holds the `AdsTree` and per-leaf ODE
`Solution` objects; access the tree via `sol.tree()`.

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
    -> AdsTree<Eigen::Matrix<TaylorExpansion<T, P, M>, D, 1>, M, T>;
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

Tree run(F&& rhs, const BoxT& ic_box, const Eigen::Matrix<T, D, 1>& ic_center, T t0, T t1);
```

`AdsDriver`'s `extras` are extra `std::shared_ptr<tax::ode::Event<State,T>>`
appended to every box's integration (deep-copied per leaf for the parallel
driver via `clone()`).

---

## Merging

```cpp
struct MergeStats { int passes = 0; int merges = 0; int rejected = 0; };

template <class Payload, int M, class T, class Criterion>
MergeStats merge(AdsTree<Payload, M, T>& tree, Criterion crit);
```

A post-pass that collapses sibling pairs whose reconstructed parent still
satisfies `crit` (using `TruncationCriterion`/`NliCriterion`), repeating until
no pair merges. Useful to trim a tree that was split more finely than the final
shape needs.
