// include/tax/ads/domains/reorient.hpp
//
// Re-orientation of the factor frame for the Zonotope domain — the primitive
// behind *adaptive* oriented ADS.
//
// A leaf carries a DA flow map x(ξ), ξ ∈ [-1, 1]^M, and a Zonotope whose
// generators G map the same ξ to the physical initial condition. Splitting is
// axis-aligned in ξ, so the *factor frame* determines which physical
// directions a split can cut. Choosing that frame to align with the flow's
// dominant stretching makes splits efficient (fewer leaves) — but the choice
// is geometrically constrained:
//
//   A linear change of factor variables ξ = R·η maps the cube [-1, 1]^M to
//   R·[-1, 1]^M, which is a cube again ONLY when R is a signed permutation.
//   So a generic re-orientation either (a) changes the represented set, or
//   (b) must over-approximate back to a cube.
//
// Re-orientation is therefore exact in two settings: when the uncertainty is
// an ELLIPSOID (the covering parallelotope is free to orient — pick its frame
// from flowAlignedRotation, see examples/two_body/zonotope_adaptive.cpp), or
// when one re-wraps a leaf with a controlled over-approximation.
//
// This header provides the building blocks, frame-agnostic:
//   * reorientState(x, R)       — compose the flow map: y(η) = x(R·η)
//   * linearPart(x)             — A = ∂x/∂ξ|₀  (D×M), the local STM
//   * flowAlignedRotation(A)    — V from SVD(A): A·V has orthogonal columns
//   * reorientZonotope(z, R)    — keep the generators in step: G → G·R

#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <array>
#include <cstddef>
#include <tax/ads/domains/zonotope.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/la/types.hpp>
#include <tax/operators/arithmetic.hpp>
#include <utility>

namespace tax::ads
{

// y(η) = x(R·η): substitute ξ_k = Σ_j R(k,j) η_j into every component of the
// flow map. This is a full polynomial composition (not the per-axis binomial
// trick of da_state::split), so it costs O(numMonomials · M) TE multiplies —
// use it sparingly (a handful of re-orientations, not per step). R is M×M.
template < class T, int N, int M, class Storage, int D, class Derived >
[[nodiscard]] Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D,
                             1 >
reorientState( const Eigen::Matrix<
                   tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 >& state,
               const Eigen::MatrixBase< Derived >& R )
{
    using TE = tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >;

    // Linear forms ℓ_k(η) = Σ_j R(k,j) η_j as degree-1 expansions.
    std::array< TE, static_cast< std::size_t >( M ) > lin{};
    for ( int k = 0; k < M; ++k )
    {
        TE f{};
        for ( int j = 0; j < M; ++j )
        {
            const T r = T( R( k, j ) );
            if ( r == T{ 0 } ) continue;
            tax::MultiIndex< M > e{};
            e[static_cast< std::size_t >( j )] = 1;
            f[tax::flatIndex< M >( e )] = r;
        }
        lin[static_cast< std::size_t >( k )] = f;
    }

    // Power tables pw[k][p] = ℓ_k^p, truncated at degree N by TE multiply.
    std::array< std::array< TE, static_cast< std::size_t >( N ) + 1 >,
                static_cast< std::size_t >( M ) >
        pw{};
    for ( int k = 0; k < M; ++k )
    {
        TE one{};
        one[0] = T{ 1 };
        pw[static_cast< std::size_t >( k )][0] = one;
        for ( int p = 1; p <= N; ++p )
            pw[static_cast< std::size_t >( k )][static_cast< std::size_t >( p )] =
                pw[static_cast< std::size_t >( k )][static_cast< std::size_t >( p - 1 )] *
                lin[static_cast< std::size_t >( k )];
    }

    using State = Eigen::Matrix< TE, D, 1 >;
    State out{ state.size() };
    constexpr std::size_t Ncoef = tax::numMonomials( N, M );
    for ( Eigen::Index i = 0; i < state.size(); ++i )
    {
        TE acc{};
        for ( std::size_t k = 0; k < Ncoef; ++k )
        {
            const T c = state( i )[k];
            if ( c == T{ 0 } ) continue;
            const auto alpha = tax::unflatIndex< M >( k );
            TE prod{};
            prod[0] = T{ 1 };
            for ( int dim = 0; dim < M; ++dim )
            {
                const int a = alpha[static_cast< std::size_t >( dim )];
                if ( a == 0 ) continue;
                prod =
                    prod * pw[static_cast< std::size_t >( dim )][static_cast< std::size_t >( a )];
            }
            acc += prod * c;
        }
        out( i ) = std::move( acc );
    }
    return out;
}

// A = ∂x/∂ξ|₀ — the local state-transition matrix (D×M): the degree-1
// coefficient of factor ξ_j in component i.
template < class T, int N, int M, class Storage, int D >
[[nodiscard]] Eigen::Matrix< T, D, M > linearPart(
    const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 >&
        state )
{
    Eigen::Matrix< T, D, M > A;
    if constexpr ( D == Eigen::Dynamic ) A.resize( state.size(), M );
    for ( Eigen::Index i = 0; i < state.size(); ++i )
        for ( int j = 0; j < M; ++j )
        {
            tax::MultiIndex< M > e{};
            e[static_cast< std::size_t >( j )] = 1;
            A( i, j ) = state( i )[tax::flatIndex< M >( e )];
        }
    return A;
}

// V from the thin SVD A = U Σ Vᵀ. Right-multiplying the factor frame by V
// (i.e. choosing generators G·V) makes the propagated linear map A·V = U Σ
// have orthogonal columns, ordered by stretch — the "flow-aligned" frame.
template < class Derived >
[[nodiscard]] auto flowAlignedRotation( const Eigen::MatrixBase< Derived >& A )
{
    Eigen::JacobiSVD< typename Derived::PlainObject > svd( A, Eigen::ComputeThinV );
    return svd.matrixV().eval();
}

// Keep a Zonotope's generators consistent with a factor re-orientation ξ = R·η:
// the physical map center + G·ξ = center + (G·R)·η, so G → G·R.
template < class T, int M, class Derived >
[[nodiscard]] Zonotope< T, M > reorientZonotope( const Zonotope< T, M >& z,
                                                 const Eigen::MatrixBase< Derived >& R )
{
    Zonotope< T, M > out = z;
    out.generators = z.generators * R;
    return out;
}

}  // namespace tax::ads
