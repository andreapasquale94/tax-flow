// include/tax/ads/domains/zonotope.hpp
//
// Zonotope<T, M> — a parallelotope (the unconstrained member of the
// constrained-zonotope family of Scott et al. 2016 / Kochdumper & Althoff
// 2020) used as an alternative geometric primitive for the ADS tree.
//
// Where Box<T, M> is an axis-aligned hyperrectangle
//
//     { center + halfWidth ⊙ ξ : ξ ∈ [-1, 1]^M },
//
// a Zonotope carries a full M×M generator matrix G instead of a diagonal
// of half-widths:
//
//     { center + G · ξ : ξ ∈ [-1, 1]^M }.
//
// A Box is exactly the special case G = diag(halfWidth) — recoverable via
// Zonotope::fromBox / Zonotope::axisAligned. Allowing G to be a general
// (oriented) matrix lets a single leaf wrap a correlated / rotated initial
// set that an axis-aligned box would have to cover with a much larger
// bounding rectangle — and a larger domain carries more truncation mass,
// so it splits more. Oriented domains are the lever for *fewer* leaves on
// anisotropic problems.
//
// Everything downstream of the domain (the DA flow-map payload and the
// split/merge polynomial substitutions in da_state.hpp / merge.hpp) works
// in the normalised factor coordinates ξ ∈ [-1, 1]^M and is therefore
// independent of whether those factors are scaled by a diagonal (Box) or a
// dense generator matrix (Zonotope). The only domain-aware operations are
// the ones below: contains / denormalize / split / splitOrdinate.

#pragma once

#include <Eigen/Dense>  // partialPivLu for contains()
#include <tax/ads/domains/box.hpp>
#include <tax/ads/domains/domain.hpp>
#include <tax/la/types.hpp>
#include <utility>

namespace tax::ads
{

template < class T, int M >
struct Zonotope
{
    static_assert( M >= 1, "Zonotope dimension must be at least 1" );

    tax::la::VecNT< M, T > center = tax::la::VecNT< M, T >::Zero();
    // Column j is the generator spanning factor ξ_j; the leaf domain is the
    // image of the cube [-1, 1]^M under ξ ↦ center + generators · ξ.
    tax::la::MatNT< M, T > generators = tax::la::MatNT< M, T >::Zero();

    // Axis-aligned parallelotope with the given per-axis half-widths — the
    // Box-equivalent constructor (generators = diag(halfWidth)).
    [[nodiscard]] static Zonotope axisAligned( const tax::la::VecNT< M, T >& center,
                                               const tax::la::VecNT< M, T >& halfWidth ) noexcept
    {
        Zonotope z;
        z.center = center;
        z.generators = halfWidth.asDiagonal();
        return z;
    }

    // Lift a Box into the Zonotope representation (no orientation).
    [[nodiscard]] static Zonotope fromBox( const Box< T, M >& b ) noexcept
    {
        return axisAligned( b.center, b.halfWidth );
    }

    // Is pt inside the parallelotope? Recover the factor coordinates
    // ξ = G⁻¹ (pt - center) and test ‖ξ‖∞ ≤ 1. tol guards the boundary
    // against round-off (matches Box's inclusive [-halfWidth, halfWidth]).
    template < class Derived >
    [[nodiscard]] bool contains( const Eigen::MatrixBase< Derived >& pt, T tol = T{ 1e-12 } ) const
    {
        const tax::la::VecNT< M, T > rhs = pt - center;
        const tax::la::VecNT< M, T > xi = generators.partialPivLu().solve( rhs );
        for ( int i = 0; i < M; ++i )
        {
            if ( xi( i ) > T{ 1 } + tol ) return false;
            if ( xi( i ) < T{ -1 } - tol ) return false;
        }
        return true;
    }

    // Bisect along generator `dim`: halve that column and shift the centre
    // by ±half-column. The two children are themselves parallelotopes whose
    // factor ξ_dim covers the left / right half of the parent's, matching
    // the ξ_dim → ±0.5 + 0.5·ξ'_dim substitution applied to the payload.
    [[nodiscard]] std::pair< Zonotope, Zonotope > split( int dim ) const noexcept
    {
        Zonotope L = *this;
        Zonotope R = *this;
        const tax::la::VecNT< M, T > half = generators.col( dim ) * T{ 0.5 };
        L.generators.col( dim ) = half;
        R.generators.col( dim ) = half;
        L.center = center - half;
        R.center = center + half;
        return { L, R };
    }

    // Map ξ ∈ [-1, 1]^M to physical coordinates: center + G · ξ.
    template < class Derived >
    [[nodiscard]] tax::la::VecNT< M, T > denormalize(
        const Eigen::MatrixBase< Derived >& d ) const noexcept
    {
        return center + generators * d;
    }

    // Scalar position of the centre along generator `dim`, used by merge() to
    // order a sibling pair (left = smaller). center · g_dim is monotone in the
    // split offset ±½‖g_dim‖² and reduces to center(dim)·halfWidth(dim) (hence
    // the same order as Box's center(dim)) in the axis-aligned case.
    [[nodiscard]] T splitOrdinate( int dim ) const noexcept
    {
        return center.dot( generators.col( dim ) );
    }
};

template < class T, int M >
struct domain_traits< Zonotope< T, M > >
{
    using scalar = T;
    static constexpr int dim = M;
};

}  // namespace tax::ads
