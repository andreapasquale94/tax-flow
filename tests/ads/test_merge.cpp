// tests/ads/test_merge.cpp
//
// Build an over-split tree with two sibling leaves whose payloads
// reconstruct the parent exactly; verify merge collapses them. Then
// perturb one child's payload and verify merge rejects.

#include <gtest/gtest.h>

#include <tax/ads/domains/box.hpp>
#include <tax/ads/split_criteria.hpp>
#include <tax/ads/da_state.hpp>
#include <tax/ads/merge.hpp>
#include <tax/ads/tree.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>
#include <utility>

using tax::ads::AdsTree;
using tax::ads::Box;
using tax::ads::create;
using tax::ads::merge;
using tax::ads::MergeStats;
using tax::ads::split;
using tax::ads::TruncationCriterion;

namespace
{
constexpr int P = 4;
constexpr int M = 2;
constexpr int D = 2;
using TE = tax::TE< P, M >;
using State = tax::la::VecNT< D, TE >;
using Tree = AdsTree< State, M, double >;
}  // namespace

TEST( AdsMerge, CollapsesUnnecessarySplit )
{
    Box< double, M > parent{ { 0.0, 0.0 }, { 1.0, 1.0 } };
    tax::la::VecNT< D, double > x0;
    x0( 0 ) = 0.0;
    x0( 1 ) = 0.0;
    State F = create< P, M >( parent, x0 );

    Tree tree;
    const int root = tree.init( parent, F, /*t=*/0.0 );
    (void)tree.popFront();
    auto child_states = split( F, /*dim=*/0 );
    auto pr = tree.split( root, /*dim=*/0, std::move( child_states.first ),
                          std::move( child_states.second ),
                          /*tEntry=*/0.0 );
    (void)tree.popFront();
    tree.finalize( pr.first );
    (void)tree.popFront();
    tree.finalize( pr.second );

    TruncationCriterion crit{ /*tol=*/1e-10 };
    const auto stats = merge( tree, crit );

    EXPECT_GE( stats.merges, 1 );
    EXPECT_GE( stats.passes, 1 );
    EXPECT_FALSE( tree.leaf( root ).retired );
    EXPECT_TRUE( tree.leaf( root ).done );
    EXPECT_TRUE( tree.leaf( pr.first ).retired );
    EXPECT_TRUE( tree.leaf( pr.second ).retired );
}

TEST( AdsMerge, RejectsWhenChildrenDoNotMatch )
{
    Box< double, M > parent{ { 0.0, 0.0 }, { 1.0, 1.0 } };
    tax::la::VecNT< D, double > x0;
    x0( 0 ) = 0.0;
    x0( 1 ) = 0.0;
    State F = create< P, M >( parent, x0 );

    Tree tree;
    const int root = tree.init( parent, F, 0.0 );
    (void)tree.popFront();
    auto cs = split( F, /*dim=*/0 );
    cs.second( 0 )[0] += 1.0;  // perturb right child's constant
    auto pr = tree.split( root, 0, std::move( cs.first ), std::move( cs.second ), 0.0 );
    (void)tree.popFront();
    tree.finalize( pr.first );
    (void)tree.popFront();
    tree.finalize( pr.second );

    TruncationCriterion crit{ /*tol=*/1e-10 };
    const auto stats = merge( tree, crit );
    EXPECT_EQ( stats.merges, 0 );
    EXPECT_GE( stats.rejected, 1 );
    EXPECT_FALSE( tree.leaf( root ).done );    // not revived
    EXPECT_TRUE( tree.leaf( root ).retired );  // still retired
}
