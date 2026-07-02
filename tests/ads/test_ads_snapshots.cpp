// tests/ads/test_ads_snapshots.cpp
//
// End-to-end: one ADS propagation with a snapshot grid yields, per grid time,
// a partition that tiles the IC box; serial and parallel agree; t0 is the IC
// box; save_steps=false still produces snapshots; the no-grid overload yields
// just the endpoints.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <tax/ads.hpp>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <tax/tax.hpp>
#include <vector>

using tax::ads::TruncationCriterion;
using tax::domain::Box;
using tax::ode::IntegratorConfig;
using tax::ode::methods::Verner89;

namespace
{
constexpr int P = 6;
constexpr int M = 2;
constexpr int D = 2;

auto rhs()
{
    return []( const auto& x, double ) {
        using S = std::decay_t< decltype( x ) >;
        S out{ x.size() };
        out( 0 ) = x( 1 );
        out( 1 ) = -x( 0 ) - 0.1 * x( 0 ) * x( 0 ) * x( 0 );
        return out;
    };
}

Box< double, M > icBox()
{
    return Box< double, M >{ tax::la::VecNT< M, double >{ 1.0, 0.0 },
                             tax::la::VecNT< M, double >{ 0.5, 0.5 } };
}

tax::la::VecNT< D, double > icCenter()
{
    tax::la::VecNT< D, double > c;
    c << 1.0, 0.0;
    return c;
}

// Sum of leaf-box volumes; equals the IC-box volume iff the partition tiles it.
template < class Partition >
double totalVolume( const Partition& part )
{
    double v = 0.0;
    for ( const auto& leaf : part )
    {
        double vol = 1.0;
        for ( int k = 0; k < M; ++k ) vol *= 2.0 * leaf.domain.halfWidth( k );
        v += vol;
    }
    return v;
}

constexpr double kIcVolume = ( 2.0 * 0.5 ) * ( 2.0 * 0.5 );  // (2*hw0)*(2*hw1)
}  // namespace

TEST( AdsSnapshots, EachPartitionTilesTheBox )
{
    const double t1 = 2.0 * M_PI;
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    std::vector< double > grid{ 0.0, t1 / 3.0, 2.0 * t1 / 3.0, t1 };
    auto sol = tax::ads::propagate< P >( Verner89{}, TruncationCriterion{ 1e-5, 8 }, rhs(), icBox(),
                                         icCenter(), 0.0, t1, grid, cfg );

    auto snaps = sol.snapshots();
    ASSERT_GE( snaps.size(), 2u );
    for ( const auto& part : snaps )
    {
        EXPECT_GE( part.size(), 1u );
        EXPECT_NEAR( totalVolume( part ), kIcVolume, 1e-9 ) << "at t=" << part.time();
    }
    EXPECT_DOUBLE_EQ( snaps.front().time(), 0.0 );
    EXPECT_EQ( snaps.front().size(), 1u );  // t0 is the single IC box
}

TEST( AdsSnapshots, SerialMatchesParallel )
{
    const double t1 = 2.0 * M_PI;
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    std::vector< double > grid{ 0.0, t1 / 2.0, t1 };

    auto s = tax::ads::propagate< P >( Verner89{}, TruncationCriterion{ 1e-6, 8 }, rhs(), icBox(),
                                       icCenter(), 0.0, t1, grid, cfg, /*threads=*/1 );
    auto p = tax::ads::propagate< P >( Verner89{}, TruncationCriterion{ 1e-6, 8 }, rhs(), icBox(),
                                       icCenter(), 0.0, t1, grid, cfg, /*threads=*/4 );

    auto ss = s.snapshots();
    auto ps = p.snapshots();
    ASSERT_EQ( ss.size(), ps.size() );
    for ( std::size_t i = 0; i < ss.size(); ++i )
    {
        ASSERT_EQ( ss[i].size(), ps[i].size() ) << "snapshot " << i;
        for ( std::size_t l = 0; l < ss[i].size(); ++l )
            for ( int k = 0; k < M; ++k )
                EXPECT_DOUBLE_EQ( ss[i][l].domain.center( k ), ps[i][l].domain.center( k ) );
    }
}

TEST( AdsSnapshots, SaveStepsFalseStillSnapshots )
{
    const double t1 = M_PI;
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    cfg.save_steps = false;
    std::vector< double > grid{ 0.0, t1 / 2.0, t1 };

    auto sol = tax::ads::propagate< P >( Verner89{}, TruncationCriterion{ 1e-5, 8 }, rhs(), icBox(),
                                         icCenter(), 0.0, t1, grid, cfg );
    auto snaps = sol.snapshots();
    EXPECT_GE( snaps.size(), 2u );
    for ( const auto& part : snaps ) EXPECT_NEAR( totalVolume( part ), kIcVolume, 1e-9 );
}

TEST( AdsSnapshots, NoGridOverloadGivesEndpoints )
{
    const double t1 = M_PI;
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    auto sol = tax::ads::propagate< P >( Verner89{}, TruncationCriterion{ 1e-5, 8 }, rhs(), icBox(),
                                         icCenter(), 0.0, t1, cfg );
    auto snaps = sol.snapshots();
    ASSERT_EQ( snaps.size(), 2u );
    EXPECT_DOUBLE_EQ( snaps.front().time(), 0.0 );
    EXPECT_DOUBLE_EQ( snaps.back().time(), t1 );
}
