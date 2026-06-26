// include/tax/ads/refine_criteria.hpp
//
// Quality criteria for the "propagate-then-assess" ADS refinement
// (tax::ads::refine, see refine.hpp). Unlike the in-flight SplitCriterion
// of the classic driver — which inspects a single flow map mid-integration
// — a QualityCriterion judges a box by comparing the flow map of the parent
// against the flow maps of its two children once *all three* have been
// propagated all the way to the final time. The verdict answers a single
// question: "does splitting this box change the answer?" If not, the parent
// is accepted as-is; if so, the children are kept and refined further.
//
// Two indices are provided:
//
//   CoefficientMatchCriterion — dimension-free. Re-identify the parent map
//     on each half-domain (the same substitution ADS uses to split) and
//     compare it, coefficient by coefficient, to the independently
//     propagated child map. While the parent is accurate the two agree;
//     once it has diverged the mismatch blows up. Normalised by the child
//     magnitude, so `tol` is a relative tolerance.
//
//   VolumeRatioCriterion — geometric, dimension-general. Measure the
//     m-volume of the image of the box face under each flow map (the integral
//     of sqrt(det(JᵀJ)) over the active axes) and compare the parent volume
//     to the sum of the two child volumes. When the parent polynomial is well
//     shaped the children tile it and the ratio is ~1; stretching or folding
//     past its radius of convergence drives the ratio away from 1. Reduces to
//     an image area for two active axes.
//
// Both honour a `maxDepth` cap: acceptable() returns true (stop) once
// depth >= maxDepth regardless of the index value.

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <span>
#include <tax/ads/da_state.hpp>
#include <tax/ads/detail/nonlinearity_index.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <vector>

namespace tax::ads
{

// A quality criterion drives tax::ads::refine. splitDim picks the coordinate
// to bisect from a flow map; acceptable compares a parent map to the 2^k
// propagated children produced by splitting `dims` (k = dims.size(); 1 dim /
// 2 children is the binary case) and reports whether the parent is good
// enough (true => stop, do not split). Child i corresponds to the sub-box
// whose offset along dims[j] is "+" when bit j of i is set, "-" otherwise.
template < class C, class State >
concept QualityCriterion = requires( C c, const State& p, std::span< const State > ch,
                                     std::span< const int > dims, int depth ) {
    { c.acceptable( p, ch, dims, depth ) } -> std::convertible_to< bool >;
    { c.splitDim( p ) } -> std::convertible_to< int >;
    { c.maxDepth } -> std::convertible_to< int >;
};

namespace detail
{

// Coordinate j carrying the largest order-N coefficient mass — the same
// split-direction heuristic as TruncationCriterion. Graded-lex layout puts
// the degree-N monomials in the contiguous tail block.
template < class T, int N, int M, class Storage, int D >
[[nodiscard]] int topDegreeSplitDim(
    const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 >&
        f )
{
    std::array< T, M > totals{};
    for ( Eigen::Index i = 0; i < f.size(); ++i )
    {
        const auto rowMass = axisMass( f( i ) );
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

// The k coordinates carrying the most order-N coefficient mass (k = K), in
// descending order, skipping coordinates with no mass. Returns fewer than K
// when fewer than K coordinates contribute — so a box that is no longer
// nonlinear in some axis is not split there.
template < class T, int N, int M, class Storage, int D >
[[nodiscard]] std::vector< int > topKDims(
    const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 >&
        f,
    int K )
{
    std::array< T, M > totals{};
    for ( Eigen::Index i = 0; i < f.size(); ++i )
    {
        const auto rowMass = axisMass( f( i ) );
        for ( int j = 0; j < M; ++j )
            totals[static_cast< std::size_t >( j )] += rowMass[static_cast< std::size_t >( j )];
    }
    std::array< int, M > order{};
    for ( int j = 0; j < M; ++j ) order[static_cast< std::size_t >( j )] = j;
    std::sort( order.begin(), order.end(), [&]( int a, int b ) {
        return totals[static_cast< std::size_t >( a )] > totals[static_cast< std::size_t >( b )];
    } );
    std::vector< int > out;
    for ( int j : order )
    {
        if ( static_cast< int >( out.size() ) >= K ) break;
        if ( totals[static_cast< std::size_t >( j )] > T{ 0 } ) out.push_back( j );
    }
    return out;
}

// Relative coefficient mismatch between the parent map re-identified on the
// sub-box of child `combo` (ξ_{dims[j]} → ±0.5 + 0.5 ξ', sign = bit j of
// combo) and the independently propagated child.
template < class T, int N, int M, class Storage, int D >
[[nodiscard]] T childMismatch(
    const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 >&
        parent,
    const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 >&
        child,
    std::span< const int > dims, std::size_t combo )
{
    T maxDiff{ 0 };
    T maxMag{ 0 };
    constexpr std::size_t Ncoef = tax::numMonomials( N, M );
    for ( Eigen::Index i = 0; i < parent.size(); ++i )
    {
        auto restricted = parent( i );
        for ( std::size_t j = 0; j < dims.size(); ++j )
        {
            const T shift = ( ( combo >> j ) & 1u ) ? T{ 0.5 } : T{ -0.5 };
            restricted = tax::ads::detail::substituteAxis( restricted, dims[j], shift, T{ 0.5 } );
        }
        for ( std::size_t k = 0; k < Ncoef; ++k )
        {
            const T diff = std::abs( restricted[k] - child( i )[k] );
            if ( diff > maxDiff ) maxDiff = diff;
            const T mag = std::abs( child( i )[k] );
            if ( mag > maxMag ) maxMag = mag;
        }
    }
    return maxMag > T{ 0 } ? maxDiff / maxMag : maxDiff;
}

}  // namespace detail

// Dimension-free quality index: accept the parent when re-identifying it on
// each half reproduces the independently propagated child to a relative
// tolerance `tol`.
struct CoefficientMatchCriterion
{
    double tol = 1e-3;
    int maxDepth = 8;

    template < class T, int N, int M, class Storage, int D >
    [[nodiscard]] int splitDim(
        const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D,
                             1 >& f ) const
    {
        return detail::topDegreeSplitDim( f );
    }

    template < class T, int N, int M, class Storage, int D >
    [[nodiscard]] bool acceptable(
        const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D,
                             1 >& parent,
        std::span< const Eigen::Matrix<
            tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 > >
            children,
        std::span< const int > dims, int depth ) const
    {
        if ( depth >= maxDepth ) return true;
        T worst{ 0 };
        for ( std::size_t i = 0; i < children.size(); ++i )
            worst = std::max( worst, detail::childMismatch( parent, children[i], dims, i ) );
        return worst <= T{ tol };
    }
};

// Geometric quality index, dimension-general: ratio of the parent image
// "volume" to the summed child image volumes. Accept when |ratio - 1| <= tol.
//
// The image of an m-dimensional box face under the flow map is an
// m-dimensional manifold in output space; its m-volume is
//
//     V = ∫_{[-1,1]^m} sqrt( det( JᵀJ ) ) dξ ,
//
// where J is the Jacobian of the (all D) output components with respect to
// the active input axes, evaluated by a tensor-product midpoint grid of
// `nQuad` points per active axis. `axes` lists the active input axes (those
// with nonzero half-width); empty means all M. When the parent map is
// accurate its children tile it and V(parent) ≈ V(left) + V(right); both
// stretching and folding drive the ratio away from 1 (unlike a signed area,
// |det| does not cancel over a fold). For m = 2 this is an image area, so it
// generalises the planar area-ratio idea to any state-space dimension.
struct VolumeRatioCriterion
{
    double tol = 1e-6;
    int maxDepth = 8;
    std::vector< int > axes{};  // active input axes; empty ⇒ all M
    int nQuad = 8;              // quadrature points per active axis

    template < class T, int N, int M, class Storage, int D >
    [[nodiscard]] int splitDim(
        const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D,
                             1 >& f ) const
    {
        return detail::topDegreeSplitDim( f );
    }

    template < class T, int N, int M, class Storage, int D >
    [[nodiscard]] bool acceptable(
        const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D,
                             1 >& parent,
        std::span< const Eigen::Matrix<
            tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 > >
            children,
        std::span< const int > /*dims*/, int depth ) const
    {
        if ( depth >= maxDepth ) return true;
        const double vp = imageVolume( parent );
        double denom = 0.0;
        for ( const auto& c : children ) denom += imageVolume( c );
        if ( !( denom > 0.0 ) ) return true;
        return std::abs( vp / denom - 1.0 ) <= tol;
    }

    template < class T, int N, int M, class Storage, int D >
    [[nodiscard]] double imageVolume(
        const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D,
                             1 >& f ) const
    {
        // Active axes (default: all M).
        std::vector< int > ax = axes;
        if ( ax.empty() )
        {
            ax.resize( static_cast< std::size_t >( M ) );
            for ( int j = 0; j < M; ++j ) ax[static_cast< std::size_t >( j )] = j;
        }
        const int m = static_cast< int >( ax.size() );
        const int n = nQuad > 0 ? nQuad : 1;
        const Eigen::Index R = f.size();

        // Derivative polynomials dpoly[i*m + a] = ∂f_i / ∂ξ_{ax[a]} (constant
        // in ξ, so compute once and evaluate at every quadrature point).
        std::vector< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage > > dpoly;
        dpoly.reserve( static_cast< std::size_t >( R ) * static_cast< std::size_t >( m ) );
        for ( Eigen::Index i = 0; i < R; ++i )
            for ( int a = 0; a < m; ++a )
                dpoly.push_back( f( i ).deriv( ax[static_cast< std::size_t >( a )] ) );

        std::vector< double > mids( static_cast< std::size_t >( n ) );
        for ( int k = 0; k < n; ++k )
            mids[static_cast< std::size_t >( k )] = -1.0 + ( 2.0 * k + 1.0 ) / n;

        long total = 1;
        for ( int a = 0; a < m; ++a ) total *= n;

        std::vector< double > g( static_cast< std::size_t >( R ) *
                                 static_cast< std::size_t >( m ) );
        Eigen::MatrixXd G( m, m );
        double acc = 0.0;
        for ( long c = 0; c < total; ++c )
        {
            std::array< T, M > xi{};
            long cc = c;
            for ( int a = 0; a < m; ++a )
            {
                const int k = static_cast< int >( cc % n );
                cc /= n;
                xi[static_cast< std::size_t >( ax[static_cast< std::size_t >( a )] )] =
                    static_cast< T >( mids[static_cast< std::size_t >( k )] );
            }
            for ( Eigen::Index i = 0; i < R; ++i )
                for ( int a = 0; a < m; ++a )
                    g[static_cast< std::size_t >( i ) * static_cast< std::size_t >( m ) +
                      static_cast< std::size_t >( a )] =
                        static_cast< double >( dpoly[static_cast< std::size_t >( i ) *
                                                         static_cast< std::size_t >( m ) +
                                                     static_cast< std::size_t >( a )]
                                                   .eval( xi ) );
            for ( int a = 0; a < m; ++a )
                for ( int b = a; b < m; ++b )
                {
                    double s = 0.0;
                    for ( Eigen::Index i = 0; i < R; ++i )
                        s += g[static_cast< std::size_t >( i ) * static_cast< std::size_t >( m ) +
                               static_cast< std::size_t >( a )] *
                             g[static_cast< std::size_t >( i ) * static_cast< std::size_t >( m ) +
                               static_cast< std::size_t >( b )];
                    G( a, b ) = s;
                    G( b, a ) = s;
                }
            double dg = G.determinant();
            if ( dg < 0.0 ) dg = 0.0;
            acc += std::sqrt( dg );
        }

        // ∫_{[-1,1]^m} ... dξ ≈ (2^m / n^m) Σ (midpoint rule).
        return acc * std::pow( 2.0, m ) / static_cast< double >( total );
    }
};

}  // namespace tax::ads
