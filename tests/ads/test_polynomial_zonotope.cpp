// tests/ads/test_polynomial_zonotope.cpp
//
// PolynomialZonotope (curved IC set) domain: geometry primitives.
//
// PolynomialZonotope is the lower tier of the domain interface: it models the
// core Domain concept (center / split / denormalize + traits) but NOT
// LocatableDomain (no exact split-ordinate / inverse). The two static_asserts
// below are the permanent tier gate. The end-to-end propagation path is
// exercised separately via create(polyZono, x0).

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <tax/ads/da_state.hpp>
#include <tax/ads/domains/box.hpp>
#include <tax/ads/domains/polynomial_zonotope.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/la/types.hpp>

// Permanent compile-time guards (the tier gate):
//   PolynomialZonotope MUST model Domain but MUST NOT model LocatableDomain.
static_assert( tax::ads::Domain< tax::ads::PolynomialZonotope< double, 4, 2 > >,
               "PolynomialZonotope must model the core Domain concept" );
static_assert( !tax::ads::LocatableDomain< tax::ads::PolynomialZonotope< double, 4, 2 > >,
               "PolynomialZonotope must NOT model LocatableDomain (no exact split-ordinate)" );

namespace
{
constexpr int N = 4;  // generator degree
constexpr int M = 2;

using V2 = tax::la::VecNT< M, double >;
using PZ = tax::ads::PolynomialZonotope< double, N, M >;
using TE = PZ::TE;

// Set coefficient of monomial ξ0^a0 · ξ1^a1 in a degree-N expansion.
void setCoeff( TE& f, int a0, int a1, double v )
{
    tax::MultiIndex< M > e{};
    e[0] = a0;
    e[1] = a1;
    f[tax::flatIndex< M >( e )] = v;
}

// A genuinely curved poly zonotope (quadratic + cross terms).
PZ curvedPZ()
{
    PZ z;
    // value[0] = 1.0 + 0.5 ξ0 + 0.1 ξ0^2 + 0.2 ξ1
    setCoeff( z.value[0], 0, 0, 1.0 );
    setCoeff( z.value[0], 1, 0, 0.5 );
    setCoeff( z.value[0], 2, 0, 0.1 );
    setCoeff( z.value[0], 0, 1, 0.2 );
    // value[1] = -0.3 + 0.4 ξ1 + 0.05 ξ0 ξ1
    setCoeff( z.value[1], 0, 0, -0.3 );
    setCoeff( z.value[1], 0, 1, 0.4 );
    setCoeff( z.value[1], 1, 1, 0.05 );
    return z;
}
}  // namespace

// ---------------------------------------------------------------------------
// 1. fromBox reduces to the Box map (degree-1 special case).
// ---------------------------------------------------------------------------
TEST( PolynomialZonotope, FromBoxMatchesBox )
{
    tax::ads::Box< double, M > b{ { 1.0, -0.5 }, { 0.3, 0.7 } };
    auto z = PZ::fromBox( b );

    for ( double a : { -1.0, -0.4, 0.0, 0.6, 1.0 } )
        for ( double c : { -1.0, -0.25, 0.0, 0.5, 1.0 } )
        {
            const V2 d{ a, c };
            const V2 box = b.denormalize( d );
            const V2 pz = z.denormalize( d );
            EXPECT_NEAR( pz( 0 ), box( 0 ), 1e-12 );
            EXPECT_NEAR( pz( 1 ), box( 1 ), 1e-12 );
        }
}

// ---------------------------------------------------------------------------
// 2. Children tile the parent: L/R denormalize over their local cube reproduce
//    the parent over the corresponding half (same per-axis substitution).
// ---------------------------------------------------------------------------
TEST( PolynomialZonotope, SplitTilesParent )
{
    auto parent = curvedPZ();
    auto [L, R] = parent.split( 0 );

    for ( double xp : { -1.0, -0.5, 0.0, 0.5, 1.0 } )
        for ( double y : { -1.0, 0.0, 0.7, 1.0 } )
        {
            const V2 local{ xp, y };
            // ξ0 → ∓0.5 + 0.5·ξ0' on the parent.
            const V2 pl = parent.denormalize( V2{ -0.5 + 0.5 * xp, y } );
            const V2 pr = parent.denormalize( V2{ 0.5 + 0.5 * xp, y } );
            const V2 cl = L.denormalize( local );
            const V2 cr = R.denormalize( local );
            EXPECT_NEAR( cl( 0 ), pl( 0 ), 1e-12 );
            EXPECT_NEAR( cl( 1 ), pl( 1 ), 1e-12 );
            EXPECT_NEAR( cr( 0 ), pr( 0 ), 1e-12 );
            EXPECT_NEAR( cr( 1 ), pr( 1 ), 1e-12 );
        }
}

// ---------------------------------------------------------------------------
// 3. contains() is a conservative superset: every image point is inside,
//    a far-away point is outside.
// ---------------------------------------------------------------------------
TEST( PolynomialZonotope, ContainsIsConservativeSuperset )
{
    auto z = curvedPZ();
    for ( double a = -1.0; a <= 1.0 + 1e-9; a += 0.25 )
        for ( double c = -1.0; c <= 1.0 + 1e-9; c += 0.25 )
            EXPECT_TRUE( z.contains( z.denormalize( V2{ a, c } ) ) );

    // A point 100 units away on each axis is far outside (range radius < 1).
    EXPECT_FALSE( z.contains( V2{ z.center( 0 ) + 100.0, z.center( 1 ) + 100.0 } ) );
}

// ---------------------------------------------------------------------------
// 4. center(i) is the constant term of value[i].
// ---------------------------------------------------------------------------
TEST( PolynomialZonotope, CenterIsConstantTerm )
{
    auto z = curvedPZ();
    for ( int i = 0; i < M; ++i ) EXPECT_DOUBLE_EQ( z.center( i ), z.value[i][0] );
}

// ---------------------------------------------------------------------------
// 5. create(polyZono, x0) seeds an identity DA state whose constant terms are
//    the authoritative IC center x0 and whose linear part matches the box
//    half-widths — equivalent to create(box, x0) for the fromBox case.
// ---------------------------------------------------------------------------
TEST( PolynomialZonotope, CreateMatchesBoxSeed )
{
    tax::ads::Box< double, M > b{ { 1.0, -0.5 }, { 0.3, 0.7 } };
    auto z = PZ::fromBox( b );
    const V2 x0{ 2.0, 3.0 };

    auto sPz = tax::ads::create< N, M >( z, x0 );
    auto sBox = tax::ads::create< N, M >( b, x0 );

    constexpr std::size_t Ncoef = tax::numMonomials( N, M );
    for ( int i = 0; i < M; ++i )
    {
        // Constant term is the IC center; linear part is the box half-width.
        EXPECT_DOUBLE_EQ( sPz( i )[0], x0( i ) );
        tax::MultiIndex< M > e{};
        e[static_cast< std::size_t >( i )] = 1;
        EXPECT_DOUBLE_EQ( sPz( i )[tax::flatIndex< M >( e )], b.halfWidth( i ) );
        // Whole expansion matches the Box seed coefficient by coefficient.
        for ( std::size_t k = 0; k < Ncoef; ++k ) EXPECT_DOUBLE_EQ( sPz( i )[k], sBox( i )[k] );
    }
}
