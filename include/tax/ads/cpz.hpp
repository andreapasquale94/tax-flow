// include/tax/ads/cpz.hpp
//
// ConstrainedPolyZonotope — a polynomial zonotope plus polynomial equality
// constraints on its factors (Kochdumper & Althoff 2023).
//
// A leaf payload x(ξ) is already a polynomial zonotope over ξ ∈ [-1,1]^M. A
// *constrained* polynomial zonotope restricts the valid factors to the zero
// set of one or more polynomials g_j(ξ):
//
//     { x(ξ)  :  ξ ∈ [-1,1]^M ,  g_j(ξ) = 0  ∀j }.
//
// The constraints carve a lower-dimensional sub-manifold out of the box — e.g.
// "the initial state lies anywhere in this uncertainty box, but on a fixed
// energy / semi-major-axis level set". Because the constraints are polynomials
// in the SAME factors ξ, they:
//
//   * split exactly like the value map (the per-axis substituteAxis of
//     da_state.hpp re-expands a constraint on each child sub-box), and
//   * give a cheap emptiness test: over ξ ∈ [-1,1]^M every monomial obeys
//     |ξ^α| ≤ 1, so g ranges within [g₀ - r, g₀ + r] with r = Σ_{α≠0}|g_α|.
//     If 0 ∉ that interval the sub-box cannot satisfy g = 0 and the whole
//     leaf is **infeasible** — it is pruned, collapsing the box onto the
//     constrained set.
//
// This is the "split-by-constraint" lever: subdivide, prune infeasible
// children, and only the band straddling g = 0 survives.

#pragma once

#include <cmath>
#include <cstddef>
#include <tax/ads/box.hpp>
#include <tax/ads/da_state.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/la/types.hpp>
#include <utility>
#include <vector>

namespace tax::ads
{

template < class T, int N, int M, class Storage = tax::storage::Dense, int D = Eigen::Dynamic >
struct ConstrainedPolyZonotope
{
    using TE = tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >;

    // The polynomial-zonotope value map (the leaf payload).
    Eigen::Matrix< TE, D, 1 > value{};
    // Equality constraints; the set requires every g_j(ξ) = 0 over the box.
    std::vector< TE > constraints{};
};

// Range half-width of a constraint over ξ ∈ [-1,1]^M: r = Σ_{α≠0} |g_α|.
template < class T, int N, int M, class Storage >
[[nodiscard]] T intervalRadius(
    const tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >& g ) noexcept
{
    T r{ 0 };
    constexpr std::size_t Ncoef = tax::numMonomials( N, M );
    for ( std::size_t k = 1; k < Ncoef; ++k ) r += std::abs( g[k] );
    return r;
}

// Can a single constraint vanish somewhere in the box? 0 ∈ [g₀ - r, g₀ + r].
template < class T, int N, int M, class Storage >
[[nodiscard]] bool constraintFeasible(
    const tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >& g, T tol = T{ 0 } )
{
    return std::abs( g[0] ) <= intervalRadius( g ) + tol;
}

// A CPZ leaf is feasible iff every constraint can vanish in the box.
template < class T, int N, int M, class Storage, int D >
[[nodiscard]] bool feasible( const ConstrainedPolyZonotope< T, N, M, Storage, D >& z,
                             T tol = T{ 0 } )
{
    for ( const auto& g : z.constraints )
        if ( !constraintFeasible( g, tol ) ) return false;
    return true;
}

// Bisect a CPZ along factor `dim`: split the value map (da_state::split) and
// re-expand every constraint on each half with the same substitution
// ξ_dim → ±0.5 + 0.5·ξ'_dim, so the children's constraints are correct in
// their own local coordinates. `box` is the parent's geometric domain.
template < class T, int N, int M, class Storage, int D, class Domain >
[[nodiscard]] std::pair< ConstrainedPolyZonotope< T, N, M, Storage, D >,
                         ConstrainedPolyZonotope< T, N, M, Storage, D > >
split( const ConstrainedPolyZonotope< T, N, M, Storage, D >& z, const Domain& domain, int dim )
{
    using CPZ = ConstrainedPolyZonotope< T, N, M, Storage, D >;
    auto vpr = tax::ads::split( z.value, domain, dim );
    CPZ L, R;
    L.value = std::move( vpr.first );
    R.value = std::move( vpr.second );
    L.constraints.reserve( z.constraints.size() );
    R.constraints.reserve( z.constraints.size() );
    for ( const auto& g : z.constraints )
    {
        L.constraints.push_back( detail::substituteAxis( g, dim, T{ -0.5 }, T{ 0.5 } ) );
        R.constraints.push_back( detail::substituteAxis( g, dim, T{ 0.5 }, T{ 0.5 } ) );
    }
    return { std::move( L ), std::move( R ) };
}

// Re-express a root constraint on a descendant leaf's sub-box. The leaf box
// (centre c, half-width h) sits inside the root box (centre C, half-width H) at
// normalized position ξ_root = (c-C)/H + (h/H)·ξ_local per axis, so the
// constraint is restricted by the per-axis substitution substituteAxis(shift =
// (c-C)/H, scale = h/H). Lets a constraint be carried onto the leaves of an
// ordinary (constraint-unaware) ADS run for post-hoc pruning.
template < class T, int N, int M, class Storage >
[[nodiscard]] tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage > restrictConstraint(
    const tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >& g_root,
    const Box< T, M >& root, const Box< T, M >& leaf )
{
    auto out = g_root;
    for ( int dim = 0; dim < M; ++dim )
    {
        const T H = root.halfWidth( dim );
        if ( H == T{ 0 } ) continue;  // pinned axis carries no factor
        const T shift = ( leaf.center( dim ) - root.center( dim ) ) / H;
        const T scale = leaf.halfWidth( dim ) / H;
        out = detail::substituteAxis( out, dim, shift, scale );
    }
    return out;
}

}  // namespace tax::ads
