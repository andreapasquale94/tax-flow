// tests/domain/test_soundness.cpp
//
// Enclosure-soundness property sweep for the tax::domain primitives: every
// operation that claims to represent or bound a set must OVER-approximate —
// no operation may silently drop part of the set it claims to cover.
//
//   * split children TILE the parent (Box, Zonotope, PolynomialZonotope):
//     every parent point lies in a child, and every child point lies in the
//     parent — the geometric contract the whole ADS pipeline rests on;
//   * create() and split commute: splitting the identity DA state equals
//     seeding the split domain, so the payload substitution and the domain
//     geometry can never drift apart;
//   * the member interval hulls contain sampled set points (Zonotope, PZ);
//   * reorientation is a MAP-preserving, deliberately NOT a set-preserving
//     operation — asserted both ways so nobody mistakes it for an enclosure.
//
// (The enclosure/query layer over DA states — intervalHull, zonotopeEnclosure,
// ellipsoid coverings — has its own sampled-containment tests in
// test_enclosure.cpp.)

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <random>
#include <tax/ads/da_state.hpp>
#include <tax/domain.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>

using tax::domain::Box;
using tax::domain::PolynomialZonotope;
using tax::domain::Zonotope;
using V2 = tax::la::VecNT< 2, double >;

namespace
{
constexpr int M = 2;

std::vector< V2 > cubeSamples( int perAxis )
{
    std::vector< V2 > pts;
    for ( int a = 0; a < perAxis; ++a )
        for ( int b = 0; b < perAxis; ++b )
            pts.push_back(
                V2{ -1.0 + 2.0 * a / ( perAxis - 1 ), -1.0 + 2.0 * b / ( perAxis - 1 ) } );
    return pts;
}
}  // namespace

// ---------------------------------------------------------------------------
// Split tiling: children cover the parent, children stay inside the parent.
// ---------------------------------------------------------------------------

TEST( Soundness, BoxSplitChildrenTileParent )
{
    const Box< double, M > parent{ { 0.3, -1.2 }, { 0.5, 0.25 } };
    for ( int dim = 0; dim < M; ++dim )
    {
        const auto [l, r] = parent.split( dim );
        for ( const V2& xi : cubeSamples( 21 ) )
        {
            // Cover: every parent point lies in at least one child.
            const V2 p = parent.denormalize( xi );
            EXPECT_TRUE( l.contains( p ) || r.contains( p ) )
                << "gap at parent xi = (" << xi( 0 ) << ", " << xi( 1 ) << "), dim " << dim;
            // No spill: every child point lies in the parent.
            EXPECT_TRUE( parent.contains( l.denormalize( xi ) ) );
            EXPECT_TRUE( parent.contains( r.denormalize( xi ) ) );
        }
    }
}

TEST( Soundness, ZonotopeSplitChildrenTileParent )
{
    Zonotope< double, M > parent;
    parent.center = V2{ 1.0, -0.5 };
    parent.generators << 0.4, 0.15,  //
        -0.2, 0.3;                   // oriented, correlated
    for ( int dim = 0; dim < M; ++dim )
    {
        const auto [l, r] = parent.split( dim );
        for ( const V2& xi : cubeSamples( 21 ) )
        {
            const V2 p = parent.denormalize( xi );
            EXPECT_TRUE( l.contains( p ) || r.contains( p ) )
                << "gap at parent xi = (" << xi( 0 ) << ", " << xi( 1 ) << "), dim " << dim;
            EXPECT_TRUE( parent.contains( l.denormalize( xi ) ) );
            EXPECT_TRUE( parent.contains( r.denormalize( xi ) ) );
        }
    }
}

TEST( Soundness, PolynomialZonotopeSplitImagesTileParentImage )
{
    // Curved set: the split children must reproduce the parent's image on
    // each half EXACTLY (the substitution is exact polynomial algebra), so
    // the union of child images equals the parent image — no gap, no spill.
    using PZ = PolynomialZonotope< double, 4, M >;
    PZ parent = PZ::fromBox( Box< double, M >{ { 0.0, 0.0 }, { 1.0, 0.5 } } );
    tax::MultiIndex< M > q{};
    q[0] = 2;
    parent.value[1][tax::IsotropicScheme< 4, M >::flatOf( q )] = 0.3;  // bend

    for ( int dim = 0; dim < M; ++dim )
    {
        const auto [l, r] = parent.split( dim );
        for ( const V2& xi : cubeSamples( 9 ) )
        {
            // Child local ξ' maps onto the parent half: ξ_dim = ∓0.5 + 0.5 ξ'.
            V2 xl = xi, xr = xi;
            xl( dim ) = -0.5 + 0.5 * xi( dim );
            xr( dim ) = 0.5 + 0.5 * xi( dim );
            EXPECT_TRUE( l.denormalize( xi ).isApprox( parent.denormalize( xl ), 1e-13 ) );
            EXPECT_TRUE( r.denormalize( xi ).isApprox( parent.denormalize( xr ), 1e-13 ) );
        }
    }
}

// ---------------------------------------------------------------------------
// create() / split commutation: the payload substitution and the domain
// geometry describe the SAME set after a split.
// ---------------------------------------------------------------------------

TEST( Soundness, CreateSplitCommutesForBox )
{
    constexpr int P = 4, D = 2;
    const Box< double, M > box{ { 1.0, -2.0 }, { 0.3, 0.7 } };
    Eigen::Matrix< double, D, 1 > x0;
    x0 << 1.0, -2.0;
    const auto parentState = tax::domain::create< P, M >( box, x0 );

    for ( int dim = 0; dim < M; ++dim )
    {
        const auto [dl, dr] = box.split( dim );
        const auto [sl, sr] = tax::ads::split( parentState, dim );
        Eigen::Matrix< double, D, 1 > xl, xr;
        xl << dl.center( 0 ), dl.center( 1 );
        xr << dr.center( 0 ), dr.center( 1 );
        const auto seededL = tax::domain::create< P, M >( dl, xl );
        const auto seededR = tax::domain::create< P, M >( dr, xr );
        for ( int i = 0; i < D; ++i )
            for ( std::size_t k = 0; k < tax::numMonomials( P, M ); ++k )
            {
                EXPECT_NEAR( sl( i )[k], seededL( i )[k], 1e-14 );
                EXPECT_NEAR( sr( i )[k], seededR( i )[k], 1e-14 );
            }
    }
}

TEST( Soundness, CreateSplitCommutesForZonotope )
{
    constexpr int P = 4, D = 2;
    Zonotope< double, M > z;
    z.center = V2{ 0.5, 1.5 };
    z.generators << 0.2, 0.05,  //
        -0.1, 0.25;
    Eigen::Matrix< double, D, 1 > x0;
    x0 << 0.5, 1.5;
    const auto parentState = tax::domain::create< P, M >( z, x0 );

    for ( int dim = 0; dim < M; ++dim )
    {
        const auto [dl, dr] = z.split( dim );
        const auto [sl, sr] = tax::ads::split( parentState, dim );
        Eigen::Matrix< double, D, 1 > xl, xr;
        xl << dl.center( 0 ), dl.center( 1 );
        xr << dr.center( 0 ), dr.center( 1 );
        const auto seededL = tax::domain::create< P, M >( dl, xl );
        const auto seededR = tax::domain::create< P, M >( dr, xr );
        for ( int i = 0; i < D; ++i )
            for ( std::size_t k = 0; k < tax::numMonomials( P, M ); ++k )
            {
                EXPECT_NEAR( sl( i )[k], seededL( i )[k], 1e-14 );
                EXPECT_NEAR( sr( i )[k], seededR( i )[k], 1e-14 );
            }
    }
}

// ---------------------------------------------------------------------------
// Member interval hulls contain the sampled set.
// ---------------------------------------------------------------------------

TEST( Soundness, ZonotopeIntervalHullContainsSampledSet )
{
    Zonotope< double, M > z;
    z.center = V2{ -0.3, 0.8 };
    z.generators << 0.5, -0.2,  //
        0.1, 0.4;
    const auto hull = z.intervalHull();
    for ( const V2& xi : cubeSamples( 15 ) ) EXPECT_TRUE( hull.contains( z.denormalize( xi ) ) );
}

TEST( Soundness, PolynomialZonotopeIntervalHullContainsSampledImage )
{
    using PZ = PolynomialZonotope< double, 4, M >;
    PZ pz = PZ::fromBox( Box< double, M >{ { 0.2, -0.4 }, { 0.6, 0.3 } } );
    tax::MultiIndex< M > q{};
    q[0] = 1;
    q[1] = 2;
    pz.value[0][tax::IsotropicScheme< 4, M >::flatOf( q )] = 0.15;  // cross curvature
    const auto hull = pz.intervalHull();
    for ( const V2& xi : cubeSamples( 15 ) ) EXPECT_TRUE( hull.contains( pz.denormalize( xi ) ) );
}

// ---------------------------------------------------------------------------
// Reorientation preserves the MAP, deliberately not the SET.
// ---------------------------------------------------------------------------

TEST( Soundness, ReorientPreservesTheMapNotTheSet )
{
    Zonotope< double, M > z;
    z.center = V2{ 0.0, 0.0 };
    z.generators << 0.5, 0.0,  //
        0.0, 0.1;              // thin axis-aligned parallelotope
    const double th = 0.6;
    Eigen::Matrix2d R;
    R << std::cos( th ), -std::sin( th ), std::sin( th ), std::cos( th );
    const auto zr = tax::domain::reorientZonotope( z, R );

    // Map preserved: zr in the new frame equals z at the rotated coordinates.
    for ( const V2& eta : cubeSamples( 9 ) )
        EXPECT_TRUE( zr.denormalize( eta ).isApprox( z.denormalize( R * eta ), 1e-13 ) );

    // Set NOT preserved (the documented caveat): a cube corner mapped through
    // the rotated frame leaves the original thin parallelotope. Reorientation
    // alone is therefore NOT an enclosure operation — pairing it with the
    // payload (reorientState) or an over-approximating rewrap is mandatory.
    bool escaped = false;
    for ( const V2& eta : cubeSamples( 9 ) )
        escaped = escaped || !z.contains( zr.denormalize( eta ), 1e-12 );
    EXPECT_TRUE( escaped );
}
