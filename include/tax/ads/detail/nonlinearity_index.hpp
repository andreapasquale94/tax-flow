// include/tax/ads/detail/nonlinearity_index.hpp
//
// LOADS nonlinearity-index math. All helpers live in tax::ads::detail
// and operate on TaylorExpansion<T, N, M, Storage> over the normalized
// box [-1, 1]^M (the basis in which DA states built from a Box live).
//
//   linRowBound(f)            = Σ_{|α|=1} |coeff(α)|
//   jacobianVariationBound(f) = vector v ∈ R^M with
//                                v[j] = Σ_{|α|≥2} |coeff(α)| · α_j
//   nonlinearityIndex(F)      = max_i ||v_i||_1 / linRowBound(F_i)
//   nliSplitDim(F)            = argmax_j Σ_i v_i[j]
//   axisMass(f)               = vector m ∈ R^M with
//                                m[j] = Σ_{|α|=N} |coeff(α)| · α_j
//   topDegreeMass(f)          = Σ_{|α|=N} |coeff(α)|  (unweighted)

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <tax/core/multi_index.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/la/types.hpp>

namespace tax::ads::detail
{

// Sum of |coefficients| at total degree 1.
template < class T, int N, int M, class Storage >
[[nodiscard]] T linRowBound(
    const tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >& f ) noexcept
{
    static_assert( N >= 1, "linRowBound requires N >= 1" );
    T acc{ 0 };
    for ( int j = 0; j < M; ++j )
    {
        tax::MultiIndex< M > alpha{};
        alpha[static_cast< std::size_t >( j )] = 1;
        acc += std::abs( f.coeff( alpha ) );
    }
    return acc;
}

// Per-coordinate j: Σ over total-degree-≥2 monomials of |coeff| * α_j.
template < class T, int N, int M, class Storage >
[[nodiscard]] std::array< T, M > jacobianVariationBound(
    const tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >& f ) noexcept
{
    std::array< T, M > bound{};
    constexpr std::size_t Ncoef = tax::numMonomials( N, M );
    for ( std::size_t k = 0; k < Ncoef; ++k )
    {
        const auto alpha = tax::unflatIndex< M >( k );
        int total = 0;
        for ( int j = 0; j < M; ++j ) total += alpha[static_cast< std::size_t >( j )];
        if ( total < 2 ) continue;
        const T mag = std::abs( f[k] );
        for ( int j = 0; j < M; ++j )
        {
            const int aj = alpha[static_cast< std::size_t >( j )];
            bound[static_cast< std::size_t >( j )] += mag * T( aj );
        }
    }
    return bound;
}

// Per-axis degree-N coefficient mass: m[j] = Σ_{|α|=N} |f[α]| · α_j.
// Graded-lex layout: the degree-N monomials are exactly the contiguous tail
// block [numMonomials(N-1, M), numMonomials(N, M)). The N>0 guard keeps kLo at 0
// for a degree-0 expansion (where numMonomials(N-1, M) would underflow).
template < class T, int N, int M, class Storage >
[[nodiscard]] std::array< T, M > axisMass(
    const tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >& f ) noexcept
{
    std::array< T, M > mass{};
    constexpr std::size_t kLo = ( N > 0 ) ? tax::numMonomials( N - 1, M ) : 0;
    constexpr std::size_t Ncoef = tax::numMonomials( N, M );
    for ( std::size_t k = kLo; k < Ncoef; ++k )
    {
        const T a = std::abs( f[k] );
        if ( a == T{ 0 } ) continue;
        const auto alpha = tax::unflatIndex< M >( k );
        for ( int j = 0; j < M; ++j )
            mass[static_cast< std::size_t >( j )] +=
                a * T( alpha[static_cast< std::size_t >( j )] );
    }
    return mass;
}

// Unweighted degree-N tail-block mass: Σ_{|α|=N} |f[α]|.
template < class T, int N, int M, class Storage >
[[nodiscard]] T topDegreeMass(
    const tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >& f ) noexcept
{
    T s{ 0 };
    constexpr std::size_t kLo = ( N > 0 ) ? tax::numMonomials( N - 1, M ) : 0;
    constexpr std::size_t Ncoef = tax::numMonomials( N, M );
    for ( std::size_t k = kLo; k < Ncoef; ++k ) s += std::abs( f[k] );
    return s;
}

// LOADS nonlinearity index over a vector of TE rows.
template < class T, int N, int M, class Storage, int D >
[[nodiscard]] double nonlinearityIndex(
    const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 >&
        f )
{
    constexpr std::size_t Ncoef = tax::numMonomials( N, M );
    double best = 0.0;
    for ( Eigen::Index i = 0; i < f.size(); ++i )
    {
        const auto& row = f( i );
        const auto var = jacobianVariationBound( row );
        T var_l1{ 0 };
        for ( int j = 0; j < M; ++j ) var_l1 += var[static_cast< std::size_t >( j )];
        const T lin = linRowBound( row );

        // Floor both terms by the row's coefficient scale so numerically-zero
        // coefficients don't dominate the ratio. Without this a singular
        // Jacobian row (lin == 0) with any nonzero curvature — even 1e-300
        // round-off — reported an INFINITE index and forced splitting all the
        // way to maxDepth. Curvature below the floor is treated as noise (ratio
        // 0); genuine curvature with a vanishing linear part still yields a
        // large finite ratio and splits when it should.
        T rowScale{ 0 };
        for ( std::size_t k = 0; k < Ncoef; ++k )
            rowScale = std::max( rowScale, std::abs( row[k] ) );
        const T floor = rowScale * std::numeric_limits< T >::epsilon() * T{ 16 };

        if ( var_l1 <= floor ) continue;  // no meaningful nonlinearity in this row
        const T linEff = std::max( lin, floor );
        const double ratio = static_cast< double >( var_l1 ) / static_cast< double >( linEff );
        if ( ratio > best ) best = ratio;
    }
    return best;
}

// Split dimension: argmax over j of Σ_i v_i[j].
template < class T, int N, int M, class Storage, int D >
[[nodiscard]] int nliSplitDim(
    const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 >&
        f )
{
    std::array< T, M > totals{};
    for ( Eigen::Index i = 0; i < f.size(); ++i )
    {
        const auto v = jacobianVariationBound( f( i ) );
        for ( int j = 0; j < M; ++j )
            totals[static_cast< std::size_t >( j )] += v[static_cast< std::size_t >( j )];
    }
    int best_j = 0;
    T best_val = totals[0];
    for ( int j = 1; j < M; ++j )
    {
        if ( totals[static_cast< std::size_t >( j )] > best_val )
        {
            best_val = totals[static_cast< std::size_t >( j )];
            best_j = j;
        }
    }
    return best_j;
}

}  // namespace tax::ads::detail
