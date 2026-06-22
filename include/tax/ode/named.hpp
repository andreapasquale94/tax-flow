// include/tax/ode/named.hpp
//
// VectorOps specialization that lets a named expansion (tax::named) act as
// an ODE state scalar, so Eigen::Matrix< NE<...>, D, 1 > can be integrated
// directly. Covers both tax::named::NamedTaylorExpansion and
// tax::named::MixedTaylorExpansion via a single NamedExpansion concept
// constraint. The operations run on the underlying coefficient array, exactly
// like the anonymous TaylorExpansion specialization.

#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <tax/core/mixed_named.hpp>
#include <tax/core/named.hpp>
#include <tax/ode/vector_ops.hpp>

namespace tax::ode
{

// A named/mixed expansion wrapper: exposes a dense inner TaylorExpansion via
// .inner() and a compile-time coefficient count. Both
// tax::named::NamedTaylorExpansion and tax::named::MixedTaylorExpansion match.
template < class S >
concept NamedExpansion = requires( S s, const S cs ) {
    typename S::Inner;
    { S::Inner::nCoefficients } -> std::convertible_to< std::size_t >;
    { cs.inner()[std::size_t{ 0 }] };
};

template < NamedExpansion S >
struct VectorOps< S >
{
    static constexpr std::size_t nc = S::Inner::nCoefficients;
    using T = typename S::Inner::scalar_type;

    [[nodiscard]] static double norm( const S& x ) noexcept
    {
        double n = 0.0;
        for ( std::size_t i = 0; i < nc; ++i )
            n = std::max( n, std::abs( static_cast< double >( x.inner()[i] ) ) );
        return n;
    }

    static void axpy( S& y, double a, const S& x ) noexcept
    {
        const T s = static_cast< T >( a );
        for ( std::size_t i = 0; i < nc; ++i ) y.inner()[i] += s * x.inner()[i];
    }

    static void scale_assign( S& y, double a, const S& x ) noexcept
    {
        const T s = static_cast< T >( a );
        for ( std::size_t i = 0; i < nc; ++i ) y.inner()[i] = s * x.inner()[i];
    }

    [[nodiscard]] static double diff_norm( const S& a, const S& b ) noexcept
    {
        double n = 0.0;
        for ( std::size_t i = 0; i < nc; ++i )
            n = std::max( n, std::abs( static_cast< double >( a.inner()[i] - b.inner()[i] ) ) );
        return n;
    }
};

}  // namespace tax::ode
