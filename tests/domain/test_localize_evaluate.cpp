// tests/domain/test_localize_evaluate.cpp
//
// The query layer: exact factor recovery (Box/Zonotope::localize), robust
// point location on an ADS tree (locateFactors), and evaluation of the
// piecewise-polynomial flow map at physical IC points
// (AdsSolution::evaluate / Partition::evaluate) — checked against
// high-accuracy scalar references on the test_driver oscillator.

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <tax/ads/propagate.hpp>
#include <tax/ads/split_criteria.hpp>
#include <tax/domain/box.hpp>
#include <tax/domain/zonotope.hpp>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <tax/tax.hpp>

using tax::ads::TruncationCriterion;
using tax::domain::Box;
using tax::domain::Zonotope;
using tax::ode::IntegratorConfig;

namespace
{
constexpr int P = 6;
constexpr int M = 2;
constexpr int D = 2;

using V2 = tax::la::VecNT< M, double >;
using Mat2 = tax::la::MatNT< M, double >;
using ScState = tax::la::VecNT< D, double >;

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

ScState scalarReference( ScState x0, double t1 )
{
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    tax::ode::Verner89< ScState > integ{ rhs(), cfg };
    auto sol = integ.integrate( x0, 0.0, t1 );
    return sol.x.back();
}
}  // namespace

// ---------------------------------------------------------------------------
// localize
// ---------------------------------------------------------------------------

TEST( Localize, BoxRoundTripAndZeroWidthGuard )
{
    Box< double, M > b{ V2{ 1.0, -2.0 }, V2{ 0.5, 0.0 } };  // axis 1 degenerate
    const V2 xi{ 0.3, 0.0 };
    const V2 p = b.denormalize( xi );
    const V2 back = b.localize( p );
    EXPECT_NEAR( back( 0 ), 0.3, 1e-15 );
    EXPECT_DOUBLE_EQ( back( 1 ), 0.0 );  // zero-width axis maps to 0
}

TEST( Localize, ZonotopeRoundTrip )
{
    Zonotope< double, M > z;
    z.center = V2{ 1.0, 2.0 };
    z.generators << 0.4, -0.3, 0.1, 0.5;
    std::mt19937 gen{ 21 };
    std::uniform_real_distribution< double > u{ -1.0, 1.0 };
    for ( int k = 0; k < 200; ++k )
    {
        const V2 xi{ u( gen ), u( gen ) };
        const V2 back = z.localize( z.denormalize( xi ) );
        EXPECT_NEAR( back( 0 ), xi( 0 ), 1e-12 );
        EXPECT_NEAR( back( 1 ), xi( 1 ), 1e-12 );
    }
}

TEST( Localize, RankDeficientZonotope )
{
    // Second generator zeroed: the set is a segment along the first column.
    Zonotope< double, M > z;
    z.center = V2{ 0.0, 0.0 };
    z.generators << 0.5, 0.0, 0.2, 0.0;
    const V2 on = z.denormalize( V2{ 0.6, 0.0 } );
    EXPECT_TRUE( z.contains( on ) );
    const V2 xi = z.localize( on );
    EXPECT_NEAR( xi( 0 ), 0.6, 1e-12 );
    EXPECT_NEAR( xi( 1 ), 0.0, 1e-12 );  // null-space component pinned to 0
    // A point off the segment's span must be rejected even though its
    // least-squares factors are small.
    const V2 off{ 0.0, 0.3 };
    EXPECT_FALSE( z.contains( off, 1e-9 ) );
}

// ---------------------------------------------------------------------------
// locateFactors / evaluate on an ADS run (Box IC)
// ---------------------------------------------------------------------------

TEST( Evaluate, BoxTreeMatchesScalarReference )
{
    using namespace tax::ode::methods;
    const Box< double, M > ic{ V2{ 1.0, 0.0 }, V2{ 0.4, 0.4 } };
    const ScState x0{ 1.0, 0.0 };
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    const double t1 = 2.0;

    auto sol = tax::ads::propagate< P >( Verner89{}, TruncationCriterion{ 1e-8, 8 }, rhs(), ic, x0,
                                         0.0, t1, cfg );
    ASSERT_GT( sol.tree().done().size(), 1u );  // forced at least one split

    std::mt19937 gen{ 33 };
    std::uniform_real_distribution< double > u{ -1.0, 1.0 };
    for ( int k = 0; k < 50; ++k )
    {
        const V2 xi{ u( gen ), u( gen ) };
        const V2 pt = ic.denormalize( xi );

        const auto loc = sol.tree().locateFactors( pt );
        ASSERT_TRUE( loc.has_value() );
        EXPECT_TRUE( sol.tree().leaf( loc->idx ).domain.contains( pt ) );
        EXPECT_LE( loc->xi.template lpNorm< Eigen::Infinity >(), 1.0 + 1e-9 );

        const auto pred = sol.evaluate( pt );
        ASSERT_TRUE( pred.has_value() );
        const ScState ref = scalarReference( ScState{ pt( 0 ), pt( 1 ) }, t1 );
        EXPECT_NEAR( ( *pred )( 0 ), ref( 0 ), 1e-6 );
        EXPECT_NEAR( ( *pred )( 1 ), ref( 1 ), 1e-6 );
    }

    // A point outside the IC domain is not claimed by any leaf.
    EXPECT_FALSE( sol.evaluate( V2{ 2.0, 2.0 } ).has_value() );
    EXPECT_FALSE( sol.tree().locateFactors( V2{ 2.0, 2.0 } ).has_value() );

    // Partition::evaluate on the final partition agrees with AdsSolution.
    const auto fin = sol.final();
    const V2 probe = ic.denormalize( V2{ 0.25, -0.75 } );
    const auto pa = fin.evaluate( probe );
    const auto pb = sol.evaluate( probe );
    ASSERT_TRUE( pa.has_value() );
    ASSERT_TRUE( pb.has_value() );
    EXPECT_NEAR( ( *pa )( 0 ), ( *pb )( 0 ), 1e-14 );
    EXPECT_NEAR( ( *pa )( 1 ), ( *pb )( 1 ), 1e-14 );
}

// ---------------------------------------------------------------------------
// evaluate on an oriented Zonotope run (the predictXY use case)
// ---------------------------------------------------------------------------

TEST( Evaluate, ZonotopeTreeMatchesScalarReference )
{
    using namespace tax::ode::methods;
    const double c = std::cos( M_PI / 4.0 ), s = std::sin( M_PI / 4.0 );
    Mat2 rot;
    rot << c, -s, s, c;
    const auto ic = Zonotope< double, M >::oriented( V2{ 1.0, 0.0 }, rot, V2{ 0.5, 0.05 } );
    const ScState x0{ 1.0, 0.0 };
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    const double t1 = 2.0;

    auto sol = tax::ads::propagate< P >( Verner89{}, TruncationCriterion{ 1e-8, 8 }, rhs(), ic, x0,
                                         0.0, t1, cfg );

    std::mt19937 gen{ 55 };
    std::uniform_real_distribution< double > u{ -1.0, 1.0 };
    for ( int k = 0; k < 50; ++k )
    {
        const V2 xi{ u( gen ), u( gen ) };
        const V2 pt = ic.denormalize( xi );
        const auto pred = sol.evaluate( pt );
        ASSERT_TRUE( pred.has_value() );
        const ScState ref = scalarReference( ScState{ pt( 0 ), pt( 1 ) }, t1 );
        EXPECT_NEAR( ( *pred )( 0 ), ref( 0 ), 1e-6 );
        EXPECT_NEAR( ( *pred )( 1 ), ref( 1 ), 1e-6 );
    }
}
