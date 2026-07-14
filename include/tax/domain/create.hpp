// include/tax/domain/create.hpp
//
// create<P, M[, Storage]>(domain, x0) — build the identity DA state seeded by
// a domain: the state vector whose component i spans the domain's factor axis
// i as ξ runs over [-1, 1]^M (components i >= M stay constant at x0(i)).
//
// x0 supplies ALL D components of the initial condition; its leading M
// components MUST agree with the domain's own center (asserted in debug
// builds). Downstream consumers (point location, canonical ordering, merge)
// read the center from the domain's geometry while the payload's constant
// term comes from x0 — if the two disagreed, locate/evaluate would silently
// recover factor coordinates the polynomial was never built on.

#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <tax/core/multi_index.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/domain/box.hpp>
#include <tax/domain/zonotope.hpp>
#include <tax/la/types.hpp>
#include <utility>

namespace tax::domain
{

namespace detail
{
template < class T >
[[nodiscard]] inline bool centerMatches( T domainCenter, T x0i ) noexcept
{
    const T scale = T{ 1 } + std::abs( domainCenter ) + std::abs( x0i );
    return std::abs( domainCenter - x0i ) <= T{ 1e-9 } * scale;
}
}  // namespace detail

// create<P, M[, Storage]>(box, x0): build the identity DA state on box.
template < int P, int M, class Storage = tax::storage::Dense, class T, int D >
[[nodiscard]] Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< P, M >, Storage >, D,
                             1 >
create( const Box< T, M >& box, const Eigen::Matrix< T, D, 1 >& x0 )
{
    // Each box axis i in [0, M) seeds the identity direction of state component i.
    // If D < M, axes [D, M) would silently get no direction (the box would carry
    // dead uncertainty axes), so require the state to cover every axis.
    static_assert( D == Eigen::Dynamic || D >= M,
                   "tax::domain::create(): state dimension D must be >= M (every box axis must "
                   "map to a state component)." );
    Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< P, M >, Storage >, D, 1 > out;
    if constexpr ( D == Eigen::Dynamic ) out.resize( x0.size() );
    for ( Eigen::Index i = 0; i < x0.size(); ++i )
    {
        tax::TaylorExpansion< T, tax::IsotropicScheme< P, M >, Storage > comp{};
        comp[0] = x0( i );
        if ( i < M )
        {
            assert( detail::centerMatches( box.center( i ), x0( i ) ) &&
                    "tax::domain::create(): x0.head(M) must equal the domain center" );
            tax::MultiIndex< M > alpha{};
            alpha[static_cast< std::size_t >( i )] = 1;
            comp[tax::flatIndex< M >( alpha )] = box.halfWidth( i );
        }
        out( i ) = std::move( comp );
    }
    return out;
}

// create<P, M[, Storage]>(zono, x0): identity DA state on a Zonotope. State
// component i (i < M) gets the i-th generator row, so x_i(ξ)=x0_i+Σ_j G_ij ξ_j;
// rows i >= M stay constant at x0_i, as the Box overload.
template < int P, int M, class Storage = tax::storage::Dense, class T, int D >
[[nodiscard]] Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< P, M >, Storage >, D,
                             1 >
create( const Zonotope< T, M >& zono, const Eigen::Matrix< T, D, 1 >& x0 )
{
    static_assert( D == Eigen::Dynamic || D >= M,
                   "tax::domain::create(): state dimension D must be >= M (every zonotope factor "
                   "must map to a state component)." );
    Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< P, M >, Storage >, D, 1 > out;
    if constexpr ( D == Eigen::Dynamic ) out.resize( x0.size() );
    for ( Eigen::Index i = 0; i < x0.size(); ++i )
    {
        tax::TaylorExpansion< T, tax::IsotropicScheme< P, M >, Storage > comp{};
        comp[0] = x0( i );
        if ( i < M )
        {
            assert( detail::centerMatches( zono.center( i ), x0( i ) ) &&
                    "tax::domain::create(): x0.head(M) must equal the domain center" );
            for ( int j = 0; j < M; ++j )
            {
                const T g = zono.generators( static_cast< Eigen::Index >( i ), j );
                if ( g == T{ 0 } ) continue;
                tax::MultiIndex< M > alpha{};
                alpha[static_cast< std::size_t >( j )] = 1;
                comp[tax::flatIndex< M >( alpha )] = g;
            }
        }
        out( i ) = std::move( comp );
    }
    return out;
}

}  // namespace tax::domain
