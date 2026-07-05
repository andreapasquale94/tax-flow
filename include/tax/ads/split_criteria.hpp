// include/tax/ads/split_criteria.hpp
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
#include <tax/ads/detail/model_state.hpp>
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
        // static_cast, not T{tol}: braced init from double is ill-formed (narrows)
        // for a float instantiation of T.
        return totalTopDegreeMass( f ) > static_cast< T >( tol );
    }

    template < class T, int N, int M, class Storage, int D >
    [[nodiscard]] int splitDim(
        const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D,
                             1 >& f ) const
    {
        // Coordinate j with the largest sum_{|α|=N, α_j>0} |coeff(α)| · α_j,
        // summed over all rows. detail::axisMass computes the per-row degree-N
        // mass (graded-lex tail block); accumulate it across rows.
        std::array< T, M > totals{};
        for ( Eigen::Index i = 0; i < f.size(); ++i )
        {
            const auto rowMass = tax::ads::detail::axisMass( f( i ) );
            for ( int j = 0; j < M; ++j )
                totals[static_cast< std::size_t >( j )] += rowMass[static_cast< std::size_t >( j )];
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

    // Taylor-model states (duck-typed): the criterion reads the POLYNOMIAL
    // parts over the top TWO degrees (N-1 and N). The Taylor-model
    // antiderivation harvests the order-N ξ-block into the remainder on every
    // step, so a propagated payload's degree-N mass is structurally zero and
    // its truncation frontier sits at degree N-1; including the (usually
    // empty) N block also covers user-built states that do carry it. The
    // remainder itself is validated integration error, not splittable
    // structure — a custom criterion may key on it via state(i).remainder().
    template < detail::ModelValuedState ModelState >
    [[nodiscard]] bool shouldSplit( const ModelState& f, int depth ) const
    {
        if ( depth >= maxDepth ) return false;
        using T = typename ModelState::Scalar::scalar_type;
        using Scheme = typename ModelState::Scalar::scheme;
        constexpr std::size_t kLo = tailLo< ModelState >();
        T acc{ 0 };
        for ( Eigen::Index i = 0; i < f.size(); ++i )
        {
            const auto& poly = f( i ).polynomial();
            for ( std::size_t k = kLo; k < Scheme::nCoeff; ++k ) acc += std::abs( poly[k] );
        }
        return acc > static_cast< T >( tol );
    }

    template < detail::ModelValuedState ModelState >
    [[nodiscard]] int splitDim( const ModelState& f ) const
    {
        using T = typename ModelState::Scalar::scalar_type;
        using Scheme = typename ModelState::Scalar::scheme;
        constexpr int M = ModelState::Scalar::vars_v;
        constexpr std::size_t kLo = tailLo< ModelState >();
        std::array< T, M > totals{};
        for ( Eigen::Index i = 0; i < f.size(); ++i )
        {
            const auto& poly = f( i ).polynomial();
            for ( std::size_t k = kLo; k < Scheme::nCoeff; ++k )
            {
                const T a = std::abs( poly[k] );
                if ( a == T{ 0 } ) continue;
                const auto alpha = Scheme::multiOf( k );
                for ( int j = 0; j < M; ++j )
                    totals[static_cast< std::size_t >( j )] +=
                        a * T( alpha[static_cast< std::size_t >( j )] );
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
    // First flat index of the degree >= N-1 tail block (graded-lex layout);
    // degenerate low orders start at 0.
    template < class ModelState >
    [[nodiscard]] static constexpr std::size_t tailLo() noexcept
    {
        constexpr int N = ModelState::Scalar::order_v;
        constexpr int M = ModelState::Scalar::vars_v;
        return ( N >= 2 ) ? tax::numMonomials( N - 2, M ) : 0;
    }

    template < class T, int N, int M, class Storage, int D >
    static T totalTopDegreeMass(
        const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D,
                             1 >& f )
    {
        // UNWEIGHTED degree-N tail-block mass summed over rows:
        // Σ_i Σ_{|α|=N} |coeff|. This must stay unweighted — sum(axisMass)
        // would equal N·topDegreeMass and silently rescale `tol` by N.
        T acc{ 0 };
        for ( Eigen::Index i = 0; i < f.size(); ++i )
            acc += tax::ads::detail::topDegreeMass( f( i ) );
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

    // Taylor-model states (duck-typed): the LOADS index reads the Jacobian
    // structure of the POLYNOMIAL parts (copied into a plain TE vector).
    template < detail::ModelValuedState ModelState >
    [[nodiscard]] bool shouldSplit( const ModelState& f, int depth ) const
    {
        if ( depth >= maxDepth ) return false;
        return tax::ads::detail::nonlinearityIndex( polynomialsOf( f ) ) > tol;
    }

    template < detail::ModelValuedState ModelState >
    [[nodiscard]] int splitDim( const ModelState& f ) const
    {
        return tax::ads::detail::nliSplitDim( polynomialsOf( f ) );
    }

   private:
    template < class ModelState >
    [[nodiscard]] static auto polynomialsOf( const ModelState& f )
    {
        using Poly = std::decay_t< decltype( f( 0 ).polynomial() ) >;
        Eigen::Matrix< Poly, ModelState::RowsAtCompileTime, 1 > polys{ f.size() };
        for ( Eigen::Index i = 0; i < f.size(); ++i ) polys( i ) = f( i ).polynomial();
        return polys;
    }
};

}  // namespace tax::ads
