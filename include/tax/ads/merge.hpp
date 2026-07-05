// include/tax/ads/merge.hpp
//
// Bottom-up merge: scan done leaves for sibling pairs whose payloads,
// after the inverse of the create/split substitution, agree on the
// parent's coordinates within criterion tolerance. Collapse each
// accepted pair back onto the parent via AdsTree::merge.
//
// Inverse substitution (vs. da_state.hpp): ξ_dim → shift + 2·ξ_dim.
// shift = +1 for the left child, shift = -1 for the right child.
//
// The merge loop runs in passes until a pass makes no changes.

#pragma once

#include <cmath>
#include <cstddef>
#include <optional>
#include <tax/ads/tree.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/domain/detail/substitute_axis.hpp>
#include <tax/domain/domain.hpp>
#include <tax/la/types.hpp>
#include <utility>
#include <vector>

namespace tax::ads
{

struct MergeStats
{
    int passes = 0;
    int merges = 0;
    int rejected = 0;
};

namespace detail
{
// The inverse of the split substitution is ξ_dim → shift + 2·ξ_dim:
// tax::domain::detail::substituteAxis with scale = 2.
// shift = +1 for the left child, shift = -1 for the right child.

template < class T, int N, int M, class Storage, int D >
[[nodiscard]] T maxCoeffDiff(
    const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 >&
        a,
    const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 >&
        b ) noexcept
{
    T worst{ 0 };
    constexpr std::size_t Ncoef = tax::numMonomials( N, M );
    for ( Eigen::Index i = 0; i < a.size(); ++i )
        for ( std::size_t k = 0; k < Ncoef; ++k )
        {
            const T d = std::abs( a( i )[k] - b( i )[k] );
            if ( d > worst ) worst = d;
        }
    return worst;
}
}  // namespace detail

// merge(tree, crit[, mergeTol]): collapse sibling pairs whose reconstructed
// parents agree within `mergeTol` (an ABSOLUTE bound on the max per-coefficient
// difference, in payload/state units) and that the criterion would not re-split.
// mergeTol defaults to crit.tol — correct when tol IS a coefficient-mass bound
// (TruncationCriterion), but pass an explicit value for criteria whose tol is a
// different quantity (e.g. NliCriterion's dimensionless nonlinearity ratio),
// where reusing it as a coefficient tolerance is a unit mismatch.
template < class Payload, class Domain, class Criterion >
    requires tax::domain::LocatableDomain< Domain >
MergeStats merge( AdsTree< Payload, Domain >& tree, Criterion crit,
                  std::optional< tax::domain::domain_scalar_t< Domain > > mergeTol = std::nullopt )
{
    using T = tax::domain::domain_scalar_t< Domain >;
    const T tol = mergeTol.value_or( static_cast< T >( crit.tol ) );
    MergeStats stats{};

    while ( true )
    {
        ++stats.passes;
        bool changed = false;

        std::vector< int > snapshot( tree.done().begin(), tree.done().end() );
        for ( std::size_t i = 0; i < snapshot.size(); ++i )
        {
            const int li = snapshot[i];
            if ( tree.leaf( li ).retired ) continue;
            const int sib = tree.leaf( li ).siblingIdx;
            if ( sib < 0 ) continue;
            if ( !tree.leaf( sib ).done ) continue;
            if ( tree.leaf( sib ).retired ) continue;
            // The snapshot holds BOTH children of every done pair; process each
            // pair once (from its lower-indexed member) so the expensive
            // reconstruction and stats.rejected are not double-counted.
            if ( sib < li ) continue;

            const int dim = tree.leaf( li ).splitDim;

            // Determine which of the pair is left and which is right by
            // comparing splitOrdinate (reduces to center(dim) for a Box;
            // generalizes to the oriented split position for a Zonotope):
            // the child with the lower splitOrdinate is the left child
            // (shift = +1 for left, -1 for right).
            const int leftIdx = ( tree.leaf( li ).domain.splitOrdinate( dim ) <
                                  tree.leaf( sib ).domain.splitOrdinate( dim ) )
                                    ? li
                                    : sib;
            const int rightIdx = ( leftIdx == li ) ? sib : li;

            // Reconstruct parent by inverting the split substitution
            // (fill the components directly — no payload copy first).
            const Payload& pl = tree.leaf( leftIdx ).payload;
            const Payload& pr = tree.leaf( rightIdx ).payload;
            Payload fromL{ pl.size() };
            Payload fromR{ pr.size() };
            for ( Eigen::Index r = 0; r < fromL.size(); ++r )
            {
                fromL( r ) = tax::domain::detail::substituteAxis( pl( r ), dim, T{ 1 }, T{ 2 } );
                fromR( r ) = tax::domain::detail::substituteAxis( pr( r ), dim, T{ -1 }, T{ 2 } );
            }

            const T diff = detail::maxCoeffDiff( fromL, fromR );
            const int parent_depth = tree.leaf( tree.leaf( li ).parentIdx ).depth;

            // Symmetric merged payload: the midpoint of the two reconstructions.
            // With the pair within tolerance the halves agree, so this halves
            // the worst-case per-coefficient error versus keeping only fromL and
            // makes the result independent of the left/right labeling.
            Payload merged{ fromL.size() };
            for ( Eigen::Index r = 0; r < merged.size(); ++r )
                merged( r ) = ( fromL( r ) + fromR( r ) ) * T{ 0.5 };

            const bool flagged = crit.shouldSplit( merged, parent_depth );

            if ( !flagged && diff <= tol )
            {
                tree.merge( leftIdx, rightIdx, std::move( merged ) );
                ++stats.merges;
                changed = true;
            } else
            {
                ++stats.rejected;
            }
        }
        if ( !changed ) break;
    }
    return stats;
}

}  // namespace tax::ads
