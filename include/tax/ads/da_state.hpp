// include/tax/ads/da_state.hpp
//
// create        — build a DA-valued state vector from a Box and a
//                 center initial condition. Each component is
//                 x0_i + halfWidth_i · ξ_i, so the state spans the box
//                 as ξ runs over [-1, 1]^M.
//
// split         — re-identify a DA state's domain on the two halves of
//                 its parent box along `dim`. Substitution:
//                   ξ_dim  →  -0.5 + 0.5 · ξ'_dim   (left half)
//                   ξ_dim  →  +0.5 + 0.5 · ξ'_dim   (right half)
//                 so the children carry polynomials in their own local
//                 [-1, 1] coordinates.

#pragma once

#include <array>
#include <cstddef>
#include <tax/ads/domains/box.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/la/types.hpp>
#include <utility>

namespace tax::ads
{

namespace detail
{
// Binomial coefficient C(n, k).
[[nodiscard]] inline constexpr double binom( int n, int k ) noexcept
{
    if ( k < 0 || k > n ) return 0.0;
    double r = 1.0;
    for ( int i = 0; i < k; ++i ) r = r * double( n - i ) / double( i + 1 );
    return r;
}

// Substitute ξ_dim → shift + scale · ξ_dim in a single TE coefficient by
// coefficient via the binomial expansion of (shift + scale·ξ_dim)^a_dim.
// scale = 0.5 is the split substitution; scale = 2 its inverse (merge).
template < class T, int N, int M, class Storage >
[[nodiscard]] tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage > substituteAxis(
    const tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >& f, int dim, T shift,
    T scale ) noexcept
{
    // Power tables: shift_pow[p] = shift^p, scale_pow[p] = scale^p
    // (exponents never exceed N — avoids std::pow in the inner loop).
    std::array< T, static_cast< std::size_t >( N ) + 1 > shift_pow{};
    std::array< T, static_cast< std::size_t >( N ) + 1 > scale_pow{};
    shift_pow[0] = T{ 1 };
    scale_pow[0] = T{ 1 };
    for ( int p = 1; p <= N; ++p )
    {
        shift_pow[static_cast< std::size_t >( p )] =
            shift_pow[static_cast< std::size_t >( p - 1 )] * shift;
        scale_pow[static_cast< std::size_t >( p )] =
            scale_pow[static_cast< std::size_t >( p - 1 )] * scale;
    }

    tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage > out{};
    constexpr std::size_t Ncoef = tax::numMonomials( N, M );
    for ( std::size_t k = 0; k < Ncoef; ++k )
    {
        const T cval = f[k];
        if ( cval == T{ 0 } ) continue;
        const auto alpha = tax::unflatIndex< M >( k );
        const int aDim = alpha[static_cast< std::size_t >( dim )];
        int aTotal = 0;
        for ( int q = 0; q < M; ++q ) aTotal += alpha[static_cast< std::size_t >( q )];
        // Distribute (shift + scale·ξ_dim)^aDim into ξ_dim^j terms.
        for ( int j = 0; j <= aDim; ++j )
        {
            if ( aTotal - aDim + j > N ) break;  // degrees only grow with j
            tax::MultiIndex< M > beta = alpha;
            beta[static_cast< std::size_t >( dim )] = j;
            const T coef = cval * T( detail::binom( aDim, j ) ) *
                           shift_pow[static_cast< std::size_t >( aDim - j )] *
                           scale_pow[static_cast< std::size_t >( j )];
            out[tax::flatIndex< M >( beta )] += coef;
        }
    }
    return out;
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
                   "ads::create(): state dimension D must be >= M (every box axis must "
                   "map to a state component)." );
    Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< P, M >, Storage >, D, 1 > out;
    if constexpr ( D == Eigen::Dynamic ) out.resize( x0.size() );
    for ( Eigen::Index i = 0; i < x0.size(); ++i )
    {
        tax::TaylorExpansion< T, tax::IsotropicScheme< P, M >, Storage > comp{};
        comp[0] = x0( i );
        if ( i < M )
        {
            tax::MultiIndex< M > alpha{};
            alpha[static_cast< std::size_t >( i )] = 1;
            comp[tax::flatIndex< M >( alpha )] = box.halfWidth( i );
        }
        out( i ) = std::move( comp );
    }
    return out;
}

// split(state, dim): produce the left/right halves.
// Deduces Storage from the input.
template < class T, int N, int M, class Storage, int D >
[[nodiscard]] std::pair<
    Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 >,
    Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 > >
split( const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D,
                            1 >& state,
       int dim )
{
    using State =
        Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 >;
    State L{ state.size() };
    State R{ state.size() };
    for ( Eigen::Index i = 0; i < state.size(); ++i )
    {
        L( i ) = detail::substituteAxis( state( i ), dim, T{ -0.5 }, T{ 0.5 } );
        R( i ) = detail::substituteAxis( state( i ), dim, T{ 0.5 }, T{ 0.5 } );
    }
    return { std::move( L ), std::move( R ) };
}

}  // namespace tax::ads
