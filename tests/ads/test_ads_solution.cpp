// tests/ads/test_ads_solution.cpp
//
// Unit tests for AdsSolution::snapshots()/final() built from hand-constructed
// trees and per-leaf Solutions (no integration) — pins the bucketing,
// bracketing, clustering, and canonical ordering.

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <tax/ads/box.hpp>
#include <tax/ads/da_state.hpp>
#include <tax/ads/solution.hpp>
#include <tax/ads/tree.hpp>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <tax/ode/solution.hpp>
#include <tax/tax.hpp>
#include <utility>
#include <vector>

using tax::ads::AdsSolution;
using tax::ads::AdsTree;
using tax::ads::Box;
using tax::ads::create;
using tax::ads::kSnapshotLabel;
using tax::ads::split;
using tax::ode::Verner89Stepper;

namespace
{
constexpr int P = 2;
constexpr int M = 2;
constexpr int D = 2;
using TE = tax::TE< P, M >;
using DAState = tax::la::VecNT< D, TE >;
using Stepper = Verner89Stepper< DAState >;
using Tree = AdsTree< DAState, M, double >;
using LeafSol = tax::ode::Solution< Stepper, DAState >;
using Sol = AdsSolution< Stepper, M >;

struct Built
{
    Tree tree;
    std::vector< LeafSol > leafSol;
    int root, c0, c1;
};

// Root box split once along dim 0 at t=0.4; both children finalized at t1=1.0.
// Each child carries an "ads:grid" event at every time in childGridTimes.
Built buildSplitOnce( const std::vector< double >& childGridTimes )
{
    Box< double, M > parent{ tax::la::VecNT< M, double >{ 0.0, 0.0 },
                             tax::la::VecNT< M, double >{ 1.0, 1.0 } };
    tax::la::VecNT< D, double > x0;
    x0 << 0.0, 0.0;
    DAState rootState = create< P, M >( parent, x0 );

    Tree tree;
    const int root = tree.init( parent, rootState, /*t=*/0.0 );
    (void)tree.popFront();
    auto cs = split( rootState, parent, /*dim=*/0 );
    DAState c0entry = cs.first;
    DAState c1entry = cs.second;
    auto pr = tree.split( root, /*dim=*/0, std::move( cs.first ), std::move( cs.second ),
                          /*tEntry=*/0.4 );
    (void)tree.popFront();
    tree.leaf( pr.first ).payload = c0entry;  // stand-in final flow map
    tree.finalize( pr.first );
    (void)tree.popFront();
    tree.leaf( pr.second ).payload = c1entry;
    tree.finalize( pr.second );
    tree.canonicalizeDone();

    std::vector< LeafSol > leafSol( 3 );
    leafSol[static_cast< std::size_t >( root )].t = { 0.0 };
    leafSol[static_cast< std::size_t >( root )].x = { rootState };
    for ( int ci : { pr.first, pr.second } )
    {
        DAState entry = ( ci == pr.first ) ? c0entry : c1entry;
        auto& ls = leafSol[static_cast< std::size_t >( ci )];
        ls.t = { 0.4, 1.0 };
        ls.x = { entry, entry };
        for ( double tg : childGridTimes )
            ls.events.push_back( { std::string{ kSnapshotLabel }, tg, entry } );
    }
    return Built{ std::move( tree ), std::move( leafSol ), root, pr.first, pr.second };
}
}  // namespace

TEST( AdsSolution, SnapshotsBracketAndBucket )
{
    auto b = buildSplitOnce( /*childGridTimes=*/{ 0.5, 1.0 } );
    Sol sol{ std::move( b.tree ), std::move( b.leafSol ), /*t0=*/0.0, /*t1=*/1.0 };

    auto snaps = sol.snapshots();
    ASSERT_EQ( snaps.size(), 3u );  // t0 bracket, t=0.5 cluster, t1 bracket
    EXPECT_DOUBLE_EQ( snaps[0].time(), 0.0 );
    EXPECT_EQ( snaps[0].size(), 1u );
    EXPECT_DOUBLE_EQ( snaps[1].time(), 0.5 );
    EXPECT_EQ( snaps[1].size(), 2u );
    EXPECT_DOUBLE_EQ( snaps[2].time(), 1.0 );
    EXPECT_EQ( snaps[2].size(), 2u );

    EXPECT_DOUBLE_EQ( snaps[0][0].box.center( 0 ), 0.0 );
    EXPECT_DOUBLE_EQ( snaps[0][0].box.halfWidth( 0 ), 1.0 );

    EXPECT_DOUBLE_EQ( snaps[1][0].box.center( 0 ), -0.5 );
    EXPECT_DOUBLE_EQ( snaps[1][1].box.center( 0 ), 0.5 );
    EXPECT_DOUBLE_EQ( snaps[1][0].box.halfWidth( 0 ), 0.5 );
    EXPECT_EQ( snaps[1][0].id, 0 );
    EXPECT_EQ( snaps[1][1].id, 1 );

    EXPECT_EQ( snaps[0][0].depth, 0 );
    EXPECT_EQ( snaps[1][0].depth, 1 );
    EXPECT_EQ( snaps[1][1].depth, 1 );
}

TEST( AdsSolution, FinalMatchesLastSnapshot )
{
    auto b = buildSplitOnce( { 1.0 } );
    Sol sol{ std::move( b.tree ), std::move( b.leafSol ), 0.0, 1.0 };
    auto fin = sol.final();
    EXPECT_DOUBLE_EQ( fin.time(), 1.0 );
    EXPECT_EQ( fin.size(), 2u );
    auto snaps = sol.snapshots();
    ASSERT_EQ( snaps.back().size(), fin.size() );
    for ( std::size_t i = 0; i < fin.size(); ++i )
    {
        EXPECT_DOUBLE_EQ( snaps.back()[i].box.center( 0 ), fin[i].box.center( 0 ) );
        EXPECT_EQ( snaps.back()[i].id, fin[i].id );
    }
}

TEST( AdsSolution, ClustersNearbyTimes )
{
    // Two leaves record the same grid time off by 1e-12 (cross-leaf rounding):
    // they must collapse into ONE snapshot of size 2.
    auto b = buildSplitOnce( {} );
    auto& ls0 = b.leafSol[static_cast< std::size_t >( b.c0 )];
    auto& ls1 = b.leafSol[static_cast< std::size_t >( b.c1 )];
    ls0.events.push_back( { std::string{ kSnapshotLabel }, 0.5, ls0.x.front() } );
    ls1.events.push_back( { std::string{ kSnapshotLabel }, 0.5 + 1e-12, ls1.x.front() } );
    Sol sol{ std::move( b.tree ), std::move( b.leafSol ), 0.0, 1.0 };

    auto snaps = sol.snapshots();
    ASSERT_EQ( snaps.size(), 3u );
    EXPECT_EQ( snaps[1].size(), 2u );
}

TEST( AdsSolution, NoGridGivesEndpointsOnly )
{
    auto b = buildSplitOnce( /*childGridTimes=*/{} );
    Sol sol{ std::move( b.tree ), std::move( b.leafSol ), 0.0, 1.0 };
    auto snaps = sol.snapshots();
    ASSERT_EQ( snaps.size(), 2u );
    EXPECT_DOUBLE_EQ( snaps[0].time(), 0.0 );
    EXPECT_EQ( snaps[0].size(), 1u );
    EXPECT_DOUBLE_EQ( snaps[1].time(), 1.0 );
    EXPECT_EQ( snaps[1].size(), 2u );
}

TEST( AdsSolution, RetiredParentGridRecordIncluded )
{
    // A grid record on the (later retired) root at t=0.2 — before the split at
    // 0.4 — must appear as the sole partition member (the parent box) at t=0.2,
    // proving retired leaves' pre-split records are not dropped.
    auto b = buildSplitOnce( /*childGridTimes=*/{ 0.5 } );
    auto& root = b.leafSol[static_cast< std::size_t >( b.root )];
    root.events.push_back( { std::string{ kSnapshotLabel }, 0.2, root.x.front() } );
    Sol sol{ std::move( b.tree ), std::move( b.leafSol ), 0.0, 1.0 };

    auto snaps = sol.snapshots();
    ASSERT_EQ( snaps.size(), 4u );  // t0(0.0), parent(0.2), children(0.5), t1(1.0)
    EXPECT_DOUBLE_EQ( snaps[1].time(), 0.2 );
    ASSERT_EQ( snaps[1].size(), 1u );
    EXPECT_DOUBLE_EQ( snaps[1][0].box.center( 0 ), 0.0 );  // parent box, not a child
    EXPECT_DOUBLE_EQ( snaps[1][0].box.halfWidth( 0 ), 1.0 );
}
