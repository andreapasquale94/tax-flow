// include/tax/domain/detail/substitute_axis.hpp
//
// substituteAxis — the per-axis affine re-identification underlying every
// domain split and merge: ξ_dim → shift + scale · ξ_dim applied to a
// TaylorExpansion coefficient by coefficient via the binomial expansion of
// (shift + scale·ξ_dim)^a_dim. scale = 0.5 is the split substitution;
// scale = 2 its inverse (merge).

#pragma once

#include <array>
#include <cstddef>
#include <tax/core/multi_index.hpp>
#include <tax/core/taylor_expansion.hpp>

namespace tax::domain::detail
{
// Binomial coefficient C(n, k).
[[nodiscard]] inline constexpr double binom( int n, int k ) noexcept
{
    if ( k < 0 || k > n ) return 0.0;
    double r = 1.0;
    for ( int i = 0; i < k; ++i ) r = r * double( n - i ) / double( i + 1 );
    return r;
}

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
            const T coef = cval * T( binom( aDim, j ) ) *
                           shift_pow[static_cast< std::size_t >( aDim - j )] *
                           scale_pow[static_cast< std::size_t >( j )];
            out[tax::flatIndex< M >( beta )] += coef;
        }
    }
    return out;
}
}  // namespace tax::domain::detail
