// tests/ads/test_zonotope.cpp
//
// Zonotope (parallelotope) domain: geometry primitives.
// The end-to-end propagation, seeding, and merge tests depend on Task 3.x
// (create(zono, x0), generic propagate over Zonotope) and are deferred.
//
// TODO(Phase 3): create/propagate/merge zonotope tests added in Tasks 3.3/3.7

#include <gtest/gtest.h>

#include <cmath>
#include <tax/ads/domains/zonotope.hpp>
#include <tax/la/types.hpp>
#include <utility>

// Permanent compile-time guards: Zonotope must model both domain tiers.
static_assert( tax::ads::Domain< tax::ads::Zonotope< double, 2 > >, "Zonotope must model Domain" );
static_assert( tax::ads::LocatableDomain< tax::ads::Zonotope< double, 2 > >,
               "Zonotope must model LocatableDomain" );

namespace
{
constexpr int M = 2;

using V2 = tax::la::VecNT< M, double >;
using Mat2 = tax::la::MatNT< M, double >;

// A 45°-rotated parallelotope: long along one diagonal, thin across it.
tax::ads::Zonotope< double, M > orientedThinZonotope()
{
    const double c = std::cos( M_PI / 4.0 );
    const double s = std::sin( M_PI / 4.0 );
    Mat2 rot;
    rot << c, -s, s, c;
    Mat2 scale;
    scale << 0.8, 0.0, 0.0, 0.03;  // long axis 0.8, thin axis 0.03
    tax::ads::Zonotope< double, M > z;
    z.center = V2{ 1.0, 0.0 };
    z.generators = rot * scale;
    return z;
}
}  // namespace

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

TEST( Zonotope, AxisAlignedMatchesBox )
{
    auto z = tax::ads::Zonotope< double, M >::axisAligned( V2{ 1.0, 2.0 }, V2{ 0.5, 0.25 } );
    EXPECT_DOUBLE_EQ( z.generators( 0, 0 ), 0.5 );
    EXPECT_DOUBLE_EQ( z.generators( 1, 1 ), 0.25 );
    EXPECT_DOUBLE_EQ( z.generators( 0, 1 ), 0.0 );
    EXPECT_DOUBLE_EQ( z.generators( 1, 0 ), 0.0 );

    // denormalize agrees with the Box mapping center + halfWidth ⊙ d.
    const V2 d{ 0.5, -1.0 };
    const V2 mapped = z.denormalize( d );
    EXPECT_DOUBLE_EQ( mapped( 0 ), 1.0 + 0.5 * 0.5 );
    EXPECT_DOUBLE_EQ( mapped( 1 ), 2.0 + 0.25 * -1.0 );
}

TEST( Zonotope, ContainsOrientedSet )
{
    auto z = orientedThinZonotope();
    // Corners ξ = (±1, ±1) lie on the boundary (inside up to tol).
    for ( double a : { -1.0, 1.0 } )
        for ( double b : { -1.0, 1.0 } ) EXPECT_TRUE( z.contains( z.denormalize( V2{ a, b } ) ) );

    // A point well outside the thin set: step off the long axis by the full
    // long half-length along the thin-axis normal.
    EXPECT_FALSE(
        z.contains( z.center + 0.45 * z.generators.col( 0 ).normalized() + V2{ 0.45, -0.45 } ) );
}

TEST( Zonotope, SplitBisectsGenerator )
{
    auto z = orientedThinZonotope();
    auto [L, R] = z.split( 0 );

    // The split generator is halved; the other is untouched.
    EXPECT_TRUE( L.generators.col( 0 ).isApprox( 0.5 * z.generators.col( 0 ) ) );
    EXPECT_TRUE( R.generators.col( 0 ).isApprox( 0.5 * z.generators.col( 0 ) ) );
    EXPECT_TRUE( L.generators.col( 1 ).isApprox( z.generators.col( 1 ) ) );

    // Centres shift by ∓ half the (old) generator column.
    const V2 half = 0.5 * z.generators.col( 0 );
    EXPECT_TRUE( L.center.isApprox( z.center - half ) );
    EXPECT_TRUE( R.center.isApprox( z.center + half ) );

    // Children tile the parent: parent ξ=+0.5 maps to the same physical point
    // as the right child's local ξ'=0.
    EXPECT_TRUE( z.denormalize( V2{ 0.5, 0.0 } ).isApprox( R.denormalize( V2{ 0.0, 0.0 } ) ) );

    // splitOrdinate orders the pair left < right along the split direction.
    EXPECT_LT( L.splitOrdinate( 0 ), R.splitOrdinate( 0 ) );
}
