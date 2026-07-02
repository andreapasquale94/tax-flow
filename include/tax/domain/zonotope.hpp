// include/tax/domain/zonotope.hpp
//
// Zonotope<T, M> — a parallelotope (the unconstrained member of the
// constrained-zonotope family of Scott et al. 2016 / Kochdumper & Althoff
// 2020) used as an oriented alternative to Box.
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
// (oriented) matrix lets a single ADS leaf wrap a correlated / rotated
// initial set that an axis-aligned box would have to cover with a much
// larger bounding rectangle — and a larger domain carries more truncation
// mass, so it splits more. Oriented domains are the lever for *fewer*
// leaves on anisotropic problems.
//
// Everything downstream of the domain (the DA flow-map payload and the
// split/merge polynomial substitutions in tax::ads) works in the normalised
// factor coordinates ξ ∈ [-1, 1]^M and is therefore independent of whether
// those factors are scaled by a diagonal (Box) or a dense generator matrix
// (Zonotope). The only domain-aware operations are the ones below:
// localize / contains / denormalize / split / splitOrdinate / intervalHull.

#pragma once

#include <Eigen/Dense>  // completeOrthogonalDecomposition for localize()
#include <cmath>
#include <tax/domain/box.hpp>
#include <tax/domain/domain.hpp>
#include <tax/la/types.hpp>
#include <utility>

namespace tax::domain
{

template < class T, int M >
struct Zonotope
{
    static_assert( M >= 1, "Zonotope dimension must be at least 1" );

    tax::la::VecNT< M, T > center = tax::la::VecNT< M, T >::Zero();
    // Column j is the generator spanning factor ξ_j; the domain is the
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

    // Oriented parallelotope R · diag(halfWidth): the "rotation × diagonal"
    // pattern of a rotated-rectangle uncertainty set.
    template < class Derived >
    [[nodiscard]] static Zonotope oriented( const tax::la::VecNT< M, T >& center,
                                            const Eigen::MatrixBase< Derived >& R,
                                            const tax::la::VecNT< M, T >& halfWidth )
    {
        Zonotope z;
        z.center = center;
        z.generators = R * halfWidth.asDiagonal();
        return z;
    }

    // Factor coordinates of pt: the minimum-norm ξ with G·ξ = pt - center.
    // Uses a rank-revealing decomposition, so rank-deficient generator
    // matrices (e.g. deliberately zeroed inactive axes) are handled: the
    // null-space component of ξ is 0, mirroring Box::localize on zero-width
    // axes. Whether pt actually lies on the (possibly degenerate) set is
    // contains()'s job — it additionally checks the reconstruction residual.
    template < class Derived >
    [[nodiscard]] tax::la::VecNT< M, T > localize( const Eigen::MatrixBase< Derived >& pt ) const
    {
        const tax::la::VecNT< M, T > rhs = pt - center;
        return generators.completeOrthogonalDecomposition().solve( rhs );
    }

    // Is pt inside the parallelotope? Recover ξ = localize(pt) and test
    // ‖ξ‖∞ ≤ 1 plus the reconstruction residual (nonzero iff pt is off the
    // span of a rank-deficient G). tol guards the boundary against round-off
    // (matches Box's inclusive [-halfWidth, halfWidth]).
    template < class Derived >
    [[nodiscard]] bool contains( const Eigen::MatrixBase< Derived >& pt, T tol = T{ 1e-12 } ) const
    {
        const tax::la::VecNT< M, T > xi = localize( pt );
        for ( int i = 0; i < M; ++i )
        {
            if ( xi( i ) > T{ 1 } + tol ) return false;
            if ( xi( i ) < T{ -1 } - tol ) return false;
        }
        const tax::la::VecNT< M, T > res = generators * xi - ( pt - center );
        const T scale = T{ 1 } + center.template lpNorm< Eigen::Infinity >() +
                        generators.template lpNorm< Eigen::Infinity >();
        return res.template lpNorm< Eigen::Infinity >() <= tol * scale;
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

    // Tightest axis-aligned Box covering the parallelotope: per axis i the
    // extent is the L1 norm of generator row i (the support of the cube).
    [[nodiscard]] Box< T, M > intervalHull() const noexcept
    {
        tax::la::VecNT< M, T > hw;
        for ( int i = 0; i < M; ++i ) hw( i ) = generators.row( i ).cwiseAbs().sum();
        return Box< T, M >{ center, hw };
    }
};

template < class T, int M >
struct domain_traits< Zonotope< T, M > >
{
    using scalar = T;
    static constexpr int dim = M;
};

}  // namespace tax::domain
