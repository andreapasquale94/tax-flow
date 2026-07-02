// include/tax/domain/box.hpp
//
// Box<T, M> — axis-aligned hyperrectangle in M-dimensional space:
//
//   { center + halfWidth ⊙ ξ : ξ ∈ [-1, 1]^M }.
//
// The default Domain primitive: every leaf of an ADS tree owns one describing
// the subdomain of initial conditions for which its payload (typically a
// DA-valued flow map) is valid. Models both Domain and LocatableDomain.
//
// Storage is tax::la::VecNT<M, T> (fixed-size Eigen vector) on both
// center and halfWidth — matches the rest of the tax pipeline, which
// is Eigen-native. The aggregate stays brace-initialisable thanks to
// Eigen 3.4's list-init for fixed-size vectors:
//
//   Box<double, 4> b{ {0.5, 0, 0, 1.7}, {1e-3, 1e-3, 1e-3, 1e-3} };

#pragma once

#include <tax/domain/domain.hpp>
#include <tax/la/types.hpp>
#include <utility>

namespace tax::domain
{

template < class T, int M >
struct Box
{
    static_assert( M >= 1, "Box dimension must be at least 1" );

    tax::la::VecNT< M, T > center = tax::la::VecNT< M, T >::Zero();
    tax::la::VecNT< M, T > halfWidth = tax::la::VecNT< M, T >::Zero();

    template < class Derived >
    [[nodiscard]] bool contains( const Eigen::MatrixBase< Derived >& pt ) const noexcept
    {
        for ( int i = 0; i < M; ++i )
        {
            const T d = pt( i ) - center( i );
            if ( d > halfWidth( i ) ) return false;
            if ( d < -halfWidth( i ) ) return false;
        }
        return true;
    }

    // Exact inverse of denormalize: ξ_i = (pt_i - c_i) / h_i. A zero-width
    // axis has no factor extent, so its coordinate is defined as 0 (whether
    // the point lies ON the degenerate axis is contains()'s job).
    template < class Derived >
    [[nodiscard]] tax::la::VecNT< M, T > localize(
        const Eigen::MatrixBase< Derived >& pt ) const noexcept
    {
        tax::la::VecNT< M, T > xi;
        for ( int i = 0; i < M; ++i )
            xi( i ) = halfWidth( i ) > T{ 0 } ? ( pt( i ) - center( i ) ) / halfWidth( i ) : T{ 0 };
        return xi;
    }

    [[nodiscard]] std::pair< Box, Box > split( int dim ) const noexcept
    {
        Box L = *this;
        Box R = *this;
        const T h = halfWidth( dim ) * T{ 0.5 };
        L.halfWidth( dim ) = h;
        R.halfWidth( dim ) = h;
        L.center( dim ) = center( dim ) - h;
        R.center( dim ) = center( dim ) + h;
        return { L, R };
    }

    // Scalar position of the centre along axis `dim`. Mirrors Zonotope::splitOrdinate
    // so merge() can order a sibling pair the same way for either domain.
    [[nodiscard]] T splitOrdinate( int dim ) const noexcept { return center( dim ); }

    // Map d ∈ [-1, 1]^M to box coordinates: center + halfWidth ⊙ d.
    template < class Derived >
    [[nodiscard]] tax::la::VecNT< M, T > denormalize(
        const Eigen::MatrixBase< Derived >& d ) const noexcept
    {
        tax::la::VecNT< M, T > out;
        for ( int i = 0; i < M; ++i ) out( i ) = center( i ) + halfWidth( i ) * d( i );
        return out;
    }
};

template < class T, int M >
struct domain_traits< Box< T, M > >
{
    using scalar = T;
    static constexpr int dim = M;
};

}  // namespace tax::domain
