// include/tax/domain/polynomial_zonotope.hpp
//
// PolynomialZonotope<T,N,M,Storage> — a polynomial zonotope: the set is the
// image of the cube [-1,1]^M under a polynomial map
//   x(ξ) = value(ξ),   value : an M-vector of degree-<=N expansions over ξ.
// Box and Zonotope are the degree-1 special cases (diagonal / dense linear
// generators); allowing higher-degree generators lets a single set wrap a
// curved region. This is the dependent-factor part of the sparse polynomial
// zonotopes of Kochdumper & Althoff 2020 (no independent-generator block).
// Splits and merges are the SAME per-axis substitution as Box/Zonotope
// (substituteAxis), because they act in factor coordinates ξ.
//
// A polynomial image has no closed-form inverse, so this models Domain but NOT
// LocatableDomain: contains() is a conservative bounding-box test (kept off
// the concept tiers on purpose) and there is no localize/splitOrdinate, so
// point location (AdsTree::locate) and merge() refuse poly zonotopes at
// compile time while propagate / AdsDriver (core Domain only) accept them —
// the intended lower tier of the domain interface.
#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <tax/core/multi_index.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/domain/box.hpp>
#include <tax/domain/create.hpp>
#include <tax/domain/detail/substitute_axis.hpp>
#include <tax/domain/domain.hpp>
#include <tax/la/types.hpp>
#include <utility>

namespace tax::domain
{
template < class T, int N, int M, class Storage = tax::storage::Dense >
struct PolynomialZonotope
{
    static_assert( M >= 1, "PolynomialZonotope dimension must be at least 1" );
    using TE = tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >;

    // value[i](ξ) is the i-th physical coordinate as a polynomial in ξ.
    std::array< TE, static_cast< std::size_t >( M ) > value{};

    // Axis-aligned box as a degree-1 poly zonotope (parity with Zonotope::fromBox).
    [[nodiscard]] static PolynomialZonotope fromBox( const Box< T, M >& b )
    {
        PolynomialZonotope z;
        for ( int i = 0; i < M; ++i )
        {
            TE c{};
            c[0] = b.center( i );
            tax::MultiIndex< M > e{};
            e[static_cast< std::size_t >( i )] = 1;
            c[tax::flatIndex< M >( e )] = b.halfWidth( i );
            z.value[static_cast< std::size_t >( i )] = c;
        }
        return z;
    }

    // Per-axis anchor: the constant term of value[dim] (the image of ξ=0).
    [[nodiscard]] T center( int dim ) const noexcept
    {
        return value[static_cast< std::size_t >( dim )][0];
    }

    // Map ξ ∈ [-1,1]^M to physical coordinates by evaluating each component.
    template < class Derived >
    [[nodiscard]] tax::la::VecNT< M, T > denormalize( const Eigen::MatrixBase< Derived >& xi ) const
    {
        tax::la::VecNT< M, T > out;
        for ( int i = 0; i < M; ++i ) out( i ) = value[static_cast< std::size_t >( i )].eval( xi );
        return out;
    }

    // Tightest axis-aligned Box the coefficients certify: component i ranges
    // within c_i ± Σ_{α≠0} |coeff(α)| over the cube.
    [[nodiscard]] Box< T, M > intervalHull() const noexcept
    {
        constexpr std::size_t Ncoef = tax::numMonomials( N, M );
        Box< T, M > out;
        for ( int i = 0; i < M; ++i )
        {
            const auto& f = value[static_cast< std::size_t >( i )];
            T r{ 0 };
            for ( std::size_t k = 1; k < Ncoef; ++k ) r += std::abs( f[k] );
            out.center( i ) = f[0];
            out.halfWidth( i ) = r;
        }
        return out;
    }

    // Conservative containment: test the axis-aligned interval hull.
    // Over-approximate (may return true for points just outside the curved
    // set) — deliberately NOT the LocatableDomain::contains contract.
    template < class Derived >
    [[nodiscard]] bool contains( const Eigen::MatrixBase< Derived >& pt, T tol = T{ 1e-12 } ) const
    {
        constexpr std::size_t Ncoef = tax::numMonomials( N, M );
        for ( int i = 0; i < M; ++i )
        {
            const auto& f = value[static_cast< std::size_t >( i )];
            T r{ 0 };
            for ( std::size_t k = 1; k < Ncoef; ++k ) r += std::abs( f[k] );
            const T d = pt( i ) - f[0];
            if ( d > r + tol || d < -r - tol ) return false;
        }
        return true;
    }

    // Bisect along factor `dim`: substitute ξ_dim → ∓0.5 + 0.5·ξ'_dim per child.
    [[nodiscard]] std::pair< PolynomialZonotope, PolynomialZonotope > split( int dim ) const
    {
        PolynomialZonotope L, R;
        for ( int i = 0; i < M; ++i )
        {
            const auto& f = value[static_cast< std::size_t >( i )];
            L.value[static_cast< std::size_t >( i )] =
                detail::substituteAxis( f, dim, T{ -0.5 }, T{ 0.5 } );
            R.value[static_cast< std::size_t >( i )] =
                detail::substituteAxis( f, dim, T{ 0.5 }, T{ 0.5 } );
        }
        return { L, R };
    }
};

template < class T, int N, int M, class Storage >
struct domain_traits< PolynomialZonotope< T, N, M, Storage > >
{
    using scalar = T;
    static constexpr int dim = M;
};

// create<P, M>(polyZono, x0): build the identity DA state seeded by a poly
// zonotope. Component i in [0, M) keeps the zonotope's polynomial generators
// but overwrites its constant term with the authoritative IC center x0(i)
// (which MUST agree with the zonotope's own center, see create.hpp);
// components i >= M are constant TEs equal to x0(i). P and M are fixed by the
// zonotope's own degree/dimension.
template < int P, int M, class Storage, class T, int D >
[[nodiscard]] Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< P, M >, Storage >, D,
                             1 >
create( const PolynomialZonotope< T, P, M, Storage >& z, const Eigen::Matrix< T, D, 1 >& x0 )
{
    static_assert( D == Eigen::Dynamic || D >= M,
                   "tax::domain::create(): state dimension D must be >= M (every factor axis must "
                   "map to a state component)." );
    using TE = tax::TaylorExpansion< T, tax::IsotropicScheme< P, M >, Storage >;
    Eigen::Matrix< TE, D, 1 > out;
    if constexpr ( D == Eigen::Dynamic ) out.resize( x0.size() );
    for ( Eigen::Index i = 0; i < x0.size(); ++i )
    {
        TE comp{};
        if ( i < M )
        {
            assert( detail::centerMatches( z.center( static_cast< int >( i ) ), x0( i ) ) &&
                    "tax::domain::create(): x0.head(M) must equal the domain center" );
            comp = z.value[static_cast< std::size_t >( i )];
        }
        comp[0] = x0( i );
        out( i ) = std::move( comp );
    }
    return out;
}
}  // namespace tax::domain
