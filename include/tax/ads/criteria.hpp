// include/tax/ads/criteria.hpp
//
// SplitCriterion concept + two implementations.
//
//   TruncationCriterion — Wittig 2015. Sum the |coefficient| values at
//     total degree N (the truncation degree of the TE). If the mass
//     exceeds `tol`, split along the coordinate contributing most of it.
//
//   NliCriterion        — Losacco/Fossà/Armellin 2024 (LOADS). Use the
//     nonlinearity index built from the Jacobian variation bound; split
//     along the coordinate whose nonlinear contribution dominates.
//
// Both criteria honour a `maxDepth` cap: shouldSplit() returns false
// once depth >= maxDepth, regardless of state magnitude.

#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <tax/ads/detail/nonlinearity_index.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/la/types.hpp>

namespace tax::ads
{

template < class C, class State >
concept SplitCriterion = requires( C c, const State& x, int depth ) {
    { c.shouldSplit( x, depth ) } -> std::convertible_to< bool >;
    { c.splitDim( x ) } -> std::convertible_to< int >;
    // The split/merge tolerance. merge() reads this directly, so it is part of
    // the criterion contract rather than an implementation detail of each type.
    { c.tol } -> std::convertible_to< double >;
};

struct TruncationCriterion
{
    double tol = 1e-6;
    int maxDepth = 30;

    template < class T, int N, int M, class Storage, int D >
    [[nodiscard]] bool shouldSplit(
        const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D,
                             1 >& f,
        int depth ) const
    {
        if ( depth >= maxDepth ) return false;
        return totalTopDegreeMass( f ) > T{ tol };
    }

    template < class T, int N, int M, class Storage, int D >
    [[nodiscard]] int splitDim(
        const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D,
                             1 >& f ) const
    {
        // Coordinate j with the largest sum_{|α|=N, α_j>0} |coeff(α)| · α_j.
        // Graded-lex layout: the degree-N monomials are exactly the
        // contiguous tail block [numMonomials(N-1, M), numMonomials(N, M)).
        std::array< T, M > totals{};
        constexpr std::size_t kLo = ( N > 0 ) ? tax::numMonomials( N - 1, M ) : 0;
        constexpr std::size_t Ncoef = tax::numMonomials( N, M );
        for ( Eigen::Index i = 0; i < f.size(); ++i )
        {
            const auto& row = f( i );
            for ( std::size_t k = kLo; k < Ncoef; ++k )
            {
                const T mag = std::abs( row[k] );
                if ( mag == T{ 0 } ) continue;
                const auto alpha = tax::unflatIndex< M >( k );
                for ( int j = 0; j < M; ++j )
                {
                    const int aj = alpha[static_cast< std::size_t >( j )];
                    totals[static_cast< std::size_t >( j )] += mag * T( aj );
                }
            }
        }
        int best = 0;
        T bestVal = totals[0];
        for ( int j = 1; j < M; ++j )
        {
            if ( totals[static_cast< std::size_t >( j )] > bestVal )
            {
                bestVal = totals[static_cast< std::size_t >( j )];
                best = j;
            }
        }
        return best;
    }

   private:
    template < class T, int N, int M, class Storage, int D >
    static T totalTopDegreeMass(
        const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D,
                             1 >& f )
    {
        // Graded-lex layout: the degree-N monomials are exactly the
        // contiguous tail block [numMonomials(N-1, M), numMonomials(N, M)).
        T acc{ 0 };
        constexpr std::size_t kLo = ( N > 0 ) ? tax::numMonomials( N - 1, M ) : 0;
        constexpr std::size_t Ncoef = tax::numMonomials( N, M );
        for ( Eigen::Index i = 0; i < f.size(); ++i )
        {
            const auto& row = f( i );
            for ( std::size_t k = kLo; k < Ncoef; ++k ) acc += std::abs( row[k] );
        }
        return acc;
    }
};

struct NliCriterion
{
    double tol = 0.1;
    int maxDepth = 30;

    template < class T, int N, int M, class Storage, int D >
    [[nodiscard]] bool shouldSplit(
        const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D,
                             1 >& f,
        int depth ) const
    {
        if ( depth >= maxDepth ) return false;
        return tax::ads::detail::nonlinearityIndex( f ) > tol;
    }

    template < class T, int N, int M, class Storage, int D >
    [[nodiscard]] int splitDim(
        const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D,
                             1 >& f ) const
    {
        return tax::ads::detail::nliSplitDim( f );
    }
};

}  // namespace tax::ads
