// include/tax/ads/box.hpp
//
// Box<T, M> — axis-aligned hyperrectangle in M-dimensional space.
// Used as the geometric primitive of the ADS tree: every leaf owns a
// Box describing the subdomain of initial conditions for which its
// payload (typically a DA-valued flow map) is valid.
//
// Storage is tax::la::VecNT<M, T> (fixed-size Eigen vector) on both
// center and halfWidth — matches the rest of the tax pipeline, which
// is Eigen-native. The aggregate stays brace-initialisable thanks to
// Eigen 3.4's list-init for fixed-size vectors:
//
//   Box<double, 4> b{ {0.5, 0, 0, 1.7}, {1e-3, 1e-3, 1e-3, 1e-3} };

#pragma once

#include <tax/la/types.hpp>
#include <utility>

namespace tax::ads
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

}  // namespace tax::ads
