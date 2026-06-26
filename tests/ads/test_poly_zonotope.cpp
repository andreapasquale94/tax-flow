// tests/ads/test_poly_zonotope.cpp
//
// PolyZonotope domain: the curved initial-condition map, its geometry
// operations (denormalize / split / contains), and an end-to-end ADS
// propagation of a bent initial set checked against scalar references.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <tax/ads/criteria.hpp>
#include <tax/ads/da_state.hpp>
#include <tax/ads/poly_zonotope.hpp>
#include <tax/ads/propagate.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <tax/tax.hpp>

using tax::ads::TruncationCriterion;
using tax::ode::IntegratorConfig;

namespace
{
constexpr int P = 6;
constexpr int M = 2;
constexpr int D = 2;

using TE = tax::TE< P, M >;
using DAState = tax::la::VecNT< D, TE >;
using ScState = tax::la::VecNT< D, double >;
using V2 = tax::la::VecNT< 2, double >;
using PZ = tax::ads::PolyZonotope< double, P, M, tax::storage::Dense, D >;

// f(x, v) = (v, -x - 0.1 x^3) — same problem as test_driver.cpp.
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

// A small but genuinely curved initial set: x(ξ) bends quadratically in ξ0.
DAState curvedMap()
{
    const TE xi0 = TE::variable( 0.0, 0 );
    const TE xi1 = TE::variable( 0.0, 1 );
    DAState m;
    m( 0 ) = 1.0 + 0.05 * xi0 + 0.02 * xi0 * xi0;
    m( 1 ) = 0.05 * xi1 + 0.015 * xi0 * xi1;
    return m;
}

ScState scalarReference( ScState x0, double t1 )
{
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    tax::ode::Verner89< ScState > integ{ rhs(), cfg };
    return integ.integrate( x0, 0.0, t1 ).x.back();
}
}  // namespace

TEST( PolyZonotope, DenormalizeIsTheMap )
{
    const PZ z = PZ::fromMap( curvedMap() );
    EXPECT_DOUBLE_EQ( z.center( 0 ), 1.0 );  // map(0) constant term
    EXPECT_DOUBLE_EQ( z.center( 1 ), 0.0 );

    const std::array< V2, 3 > ds{ { V2{ 0.0, 0.0 }, V2{ 0.7, -0.4 }, V2{ -1.0, 1.0 } } };
    for ( const auto& d : ds )
    {
        const auto phys = z.denormalize( d );
        EXPECT_DOUBLE_EQ( phys( 0 ), 1.0 + 0.05 * d( 0 ) + 0.02 * d( 0 ) * d( 0 ) );
        EXPECT_DOUBLE_EQ( phys( 1 ), 0.05 * d( 1 ) + 0.015 * d( 0 ) * d( 1 ) );
    }
}

TEST( PolyZonotope, SplitReExpandsTheCurvedMap )
{
    const PZ z = PZ::fromMap( curvedMap() );
    auto [L, R] = z.split( /*dim=*/0 );

    // Right child covers ξ0 ∈ [0,1]: local ξ0' ↦ parent ξ0 = 0.5 + 0.5 ξ0'.
    const std::array< V2, 4 > pts{
        { V2{ 0.0, 0.0 }, V2{ 0.5, -0.6 }, V2{ -0.4, 0.8 }, V2{ 1.0, 1.0 } } };
    for ( const auto& p : pts )
    {
        const V2 parent{ 0.5 + 0.5 * p( 0 ), p( 1 ) };
        const auto child = R.denormalize( p );
        const auto ref = z.denormalize( parent );
        EXPECT_NEAR( child( 0 ), ref( 0 ), 1e-12 );
        EXPECT_NEAR( child( 1 ), ref( 1 ), 1e-12 );
    }
    EXPECT_NEAR( R.center( 0 ), z.denormalize( V2{ 0.5, 0.0 } )( 0 ), 1e-12 );
}

TEST( PolyZonotope, ContainsTheCentre )
{
    const PZ z = PZ::fromMap( curvedMap() );
    EXPECT_TRUE( z.contains( z.center ) );
    EXPECT_TRUE( z.contains( z.denormalize( V2{ 0.6, -0.5 } ) ) );
    EXPECT_FALSE( z.contains( z.center + V2{ 10.0, 10.0 } ) );
}

// End-to-end: propagate the curved set (a single leaf) and confirm the leaf
// flow map reproduces scalar references taken at the curved initial states.
TEST( PolyZonotope, PropagatesCurvedSet )
{
    const double t1 = 2.0 * M_PI;
    const PZ z = PZ::fromMap( curvedMap() );
    V2 center{ 1.0, 0.0 };

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    auto tree = tax::ads::propagate< P >( tax::ode::methods::Verner89{},
                                          TruncationCriterion{ /*tol=*/1.0, /*maxDepth=*/0 }, rhs(),
                                          z, center, 0.0, t1, cfg );
    ASSERT_EQ( tree.done().size(), 1u );
    const auto& payload = tree.leaf( tree.done().front() ).payload;

    const std::array< V2, 5 > xis{
        { V2{ 0.0, 0.0 }, V2{ 0.8, 0.5 }, V2{ -0.7, 0.4 }, V2{ 0.5, -0.9 }, V2{ -1.0, -0.6 } } };
    for ( const auto& xi : xis )
    {
        const ScState ic = z.denormalize( xi );
        const ScState ref = scalarReference( ic, t1 );
        EXPECT_NEAR( payload( 0 ).eval( xi ), ref( 0 ), 1e-4 );
        EXPECT_NEAR( payload( 1 ).eval( xi ), ref( 1 ), 1e-4 );
    }
}
