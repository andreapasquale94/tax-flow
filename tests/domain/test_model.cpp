// tests/domain/test_model.cpp
//
// tax::domain ↔ tax::model bridge:
//   createModel (Box / Zonotope / PolynomialZonotope) — identity Taylor-model
//   states over ξ ∈ [-1,1]^M; the rigorous enclosure layer (intervalHull,
//   zonotopeEnclosure with remainder generators, zonotopeFrame) and the PZ
//   round-trip.
//
// Requires the tax core's tax::model module (tax PR "feat(model)").

#include <gtest/gtest.h>

#if __has_include( <tax/model.hpp>)

#include <array>
#include <cmath>
#include <tax/domain.hpp>
#include <tax/la/types.hpp>

using tax::model::Interval;

namespace
{
constexpr int P = 4;
constexpr int M = 2;
constexpr int D = 2;
using TM = tax::model::TaylorModel< double, P, M >;
using State = Eigen::Matrix< TM, D, 1 >;
using V2 = tax::la::VecNT< 2, double >;
}  // namespace

TEST( DomainModel, CreateModelBoxMatchesDenormalize )
{
    tax::domain::Box< double, M > box{ { 1.0, -0.5 }, { 0.3, 0.7 } };
    Eigen::Matrix< double, D, 1 > x0;
    x0 << 1.0, -0.5;
    const State s = tax::domain::createModel< P >( box, x0 );

    for ( double a : { -1.0, -0.4, 0.0, 0.6, 1.0 } )
        for ( double b : { -1.0, 0.0, 0.5, 1.0 } )
        {
            const V2 phys = box.denormalize( V2{ a, b } );
            const std::array< double, M > xi{ a, b };
            EXPECT_TRUE( s( 0 ).eval( xi ).contains( phys( 0 ) ) );
            EXPECT_TRUE( s( 1 ).eval( xi ).contains( phys( 1 ) ) );
        }
    // Identity models are exact: zero remainder (width() is outward-rounded,
    // so compare the endpoints).
    EXPECT_EQ( s( 0 ).remainder().lower(), 0.0 );
    EXPECT_EQ( s( 0 ).remainder().upper(), 0.0 );
    EXPECT_EQ( s( 1 ).remainder().lower(), 0.0 );
    EXPECT_EQ( s( 1 ).remainder().upper(), 0.0 );
}

TEST( DomainModel, CreateModelZonotopeMatchesDenormalize )
{
    tax::domain::Zonotope< double, M > z;
    z.center = V2{ 1.0, 2.0 };
    z.generators << 0.3, 0.1,  //
        -0.2, 0.4;
    Eigen::Matrix< double, D, 1 > x0;
    x0 << 1.0, 2.0;
    const State s = tax::domain::createModel< P >( z, x0 );

    for ( double a : { -1.0, 0.0, 0.7 } )
        for ( double b : { -0.5, 0.0, 1.0 } )
        {
            const V2 phys = z.denormalize( V2{ a, b } );
            const std::array< double, M > xi{ a, b };
            EXPECT_TRUE( s( 0 ).eval( xi ).contains( phys( 0 ) ) );
            EXPECT_TRUE( s( 1 ).eval( xi ).contains( phys( 1 ) ) );
        }
}

TEST( DomainModel, IntervalHullCoversBoundAndRemainder )
{
    tax::domain::Box< double, M > box{ { 1.0, 0.0 }, { 0.25, 0.5 } };
    Eigen::Matrix< double, D, 1 > x0;
    x0 << 1.0, 0.0;
    State s = tax::domain::createModel< P >( box, x0 );

    // Inflate component 0's remainder and check the hull covers it.
    s( 0 ).remainder() += Interval< double >{ -0.05, 0.05 };

    const auto hull = tax::domain::intervalHull( s );
    EXPECT_LE( hull.center( 0 ) - hull.halfWidth( 0 ), 1.0 - 0.25 - 0.05 );
    EXPECT_GE( hull.center( 0 ) + hull.halfWidth( 0 ), 1.0 + 0.25 + 0.05 );
    EXPECT_LE( hull.center( 1 ) - hull.halfWidth( 1 ), -0.5 );
    EXPECT_GE( hull.center( 1 ) + hull.halfWidth( 1 ), 0.5 );
}

TEST( DomainModel, ZonotopeEnclosureCoversSamplesAndRemainder )
{
    tax::domain::Box< double, M > box{ { 0.0, 0.0 }, { 1.0, 1.0 } };
    Eigen::Matrix< double, D, 1 > x0;
    x0 << 0.0, 0.0;
    State s = tax::domain::createModel< P >( box, x0 );

    // Curve component 0 (ξ1² term) and give it a remainder.
    tax::MultiIndex< M > q{};
    q[0] = 2;
    s( 0 ).polynomial()[TM::scheme::flatOf( q )] = 0.3;
    s( 0 ).remainder() += Interval< double >{ -0.02, 0.02 };

    const auto zen = tax::domain::zonotopeEnclosure( s, std::array< int, 2 >{ 0, 1 } );

    // Support-function containment over a direction fan for sampled points of
    // the set (polynomial image + remainder corner).
    for ( double a : { -1.0, -0.5, 0.0, 0.5, 1.0 } )
        for ( double b : { -1.0, 0.0, 1.0 } )
            for ( double rc : { -0.02, 0.02 } )
            {
                const Eigen::Vector2d p{ a + 0.3 * a * a + rc, b };
                for ( double ang = 0.0; ang < 6.28; ang += 0.39 )
                {
                    const Eigen::Vector2d dvec{ std::cos( ang ), std::sin( ang ) };
                    EXPECT_LE( dvec.dot( p ), zen.support( dvec ) + 1e-12 );
                }
            }
}

TEST( DomainModel, ZonotopeFrameReadsTheLinearPart )
{
    tax::domain::Zonotope< double, M > z;
    z.center = V2{ 0.5, -1.0 };
    z.generators << 0.2, 0.0,  //
        0.1, 0.3;
    Eigen::Matrix< double, D, 1 > x0;
    x0 << 0.5, -1.0;
    const State s = tax::domain::createModel< P >( z, x0 );

    const auto frame = tax::domain::zonotopeFrame( s );
    EXPECT_DOUBLE_EQ( frame.center( 0 ), 0.5 );
    EXPECT_DOUBLE_EQ( frame.center( 1 ), -1.0 );
    for ( int i = 0; i < M; ++i )
        for ( int j = 0; j < M; ++j )
            EXPECT_DOUBLE_EQ( frame.generators( i, j ), z.generators( i, j ) );
}

TEST( DomainModel, PolynomialZonotopeRoundTrip )
{
    // PZ → TM state → PZ preserves the polynomial map (zero remainders).
    using PZ = tax::domain::PolynomialZonotope< double, P, M >;
    tax::domain::Box< double, M > box{ { 1.0, 2.0 }, { 0.5, 0.25 } };
    PZ pz = PZ::fromBox( box );
    // Bend it: ξ0² into component 1.
    tax::MultiIndex< M > q{};
    q[0] = 2;
    pz.value[1][TM::scheme::flatOf( q )] = 0.1;

    Eigen::Matrix< double, D, 1 > x0;
    x0 << 1.0, 2.0;
    const State s = tax::domain::createModel< P >( pz, x0 );
    const PZ back = tax::domain::toPolynomialZonotope( s );

    for ( double a : { -1.0, 0.0, 0.8 } )
        for ( double b : { -1.0, 0.3, 1.0 } )
        {
            const V2 orig = pz.denormalize( V2{ a, b } );
            const V2 rt = back.denormalize( V2{ a, b } );
            EXPECT_NEAR( rt( 0 ), orig( 0 ), 1e-14 );
            EXPECT_NEAR( rt( 1 ), orig( 1 ), 1e-14 );
        }
}

#else  // tax core without tax::model

#include <gtest/gtest.h>
TEST( DomainModel, SkippedTaxCoreLacksModelModule ) { GTEST_SKIP(); }

#endif
