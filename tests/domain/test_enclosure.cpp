// tests/domain/test_enclosure.cpp
//
// Enclosures of the image of the factor cube under a DA state: interval
// hulls, the even-exponent zonotope enclosure, the exact degree-1 frame, and
// the ellipsoid covering helpers. Containment is checked against dense
// deterministic sampling of the image (the Monte-Carlo criterion of
// docs/reviews/ads-domain-interface): every sampled image point must satisfy
// the enclosure's membership/support test.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <tax/core/multi_index.hpp>
#include <tax/domain/box.hpp>
#include <tax/domain/ellipsoid.hpp>
#include <tax/domain/enclosure.hpp>
#include <tax/domain/zonotope.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>

using tax::domain::Box;
using tax::domain::Zonotope;

namespace
{
constexpr int P = 4;
constexpr int M = 2;
constexpr int D = 2;

using TE = tax::TE< P, M >;
using DAState = tax::la::VecNT< D, TE >;
using V2 = tax::la::VecNT< M, double >;
using Mat2 = tax::la::MatNT< M, double >;

// Deterministic dense polynomial state with all monomials populated.
DAState randomState( unsigned seed, double scale = 0.3 )
{
    std::mt19937 gen{ seed };
    std::uniform_real_distribution< double > u{ -scale, scale };
    DAState x;
    constexpr std::size_t Nc = tax::numMonomials( P, M );
    for ( int i = 0; i < D; ++i )
        for ( std::size_t k = 0; k < Nc; ++k ) x( i )[k] = u( gen );
    return x;
}

V2 evalState( const DAState& x, const V2& xi )
{
    return V2{ x( 0 ).eval( xi ), x( 1 ).eval( xi ) };
}

// Deterministic grid + pseudo-random cube samples.
std::vector< V2 > cubeSamples( unsigned seed, int n )
{
    std::mt19937 gen{ seed };
    std::uniform_real_distribution< double > u{ -1.0, 1.0 };
    std::vector< V2 > out;
    for ( double a : { -1.0, -0.5, 0.0, 0.5, 1.0 } )
        for ( double b : { -1.0, -0.5, 0.0, 0.5, 1.0 } ) out.push_back( V2{ a, b } );
    for ( int i = 0; i < n; ++i ) out.push_back( V2{ u( gen ), u( gen ) } );
    return out;
}
}  // namespace

// ---------------------------------------------------------------------------
// intervalHull of a DA image
// ---------------------------------------------------------------------------

TEST( Enclosure, IntervalHullContainsSampledImage )
{
    const DAState x = randomState( 42 );
    const auto hull = tax::domain::intervalHull( x );
    for ( const V2& xi : cubeSamples( 7, 500 ) ) EXPECT_TRUE( hull.contains( evalState( x, xi ) ) );
}

TEST( Enclosure, IntervalHullRadiusIsCoefficientSum )
{
    DAState x;
    x( 0 )[0] = 2.0;
    tax::MultiIndex< M > e{};
    e[0] = 1;
    x( 0 )[tax::flatIndex< M >( e )] = 0.5;
    e[0] = 0;
    e[1] = 2;
    x( 0 )[tax::flatIndex< M >( e )] = -0.25;
    const auto hull = tax::domain::intervalHull( x );
    EXPECT_DOUBLE_EQ( hull.center( 0 ), 2.0 );
    EXPECT_DOUBLE_EQ( hull.halfWidth( 0 ), 0.75 );
}

// ---------------------------------------------------------------------------
// zonotopeEnclosure (even-exponent shift, Kochdumper & Althoff 2020)
// ---------------------------------------------------------------------------

TEST( Enclosure, ZonotopeEnclosureSupportsSampledImage )
{
    const DAState x = randomState( 43 );
    const auto z = tax::domain::zonotopeEnclosure( x, std::array< int, 2 >{ 0, 1 } );

    // Membership via support dominance: for every direction d and image
    // point p, d·p <= ρ(d). (The zonotope is convex, so dominance over a
    // dense direction set certifies containment of the sampled cloud.)
    std::mt19937 gen{ 11 };
    std::uniform_real_distribution< double > u{ -1.0, 1.0 };
    std::vector< V2 > dirs;
    for ( int k = 0; k < 64; ++k )
    {
        const double th = 2.0 * M_PI * k / 64.0;
        dirs.push_back( V2{ std::cos( th ), std::sin( th ) } );
    }
    const auto pts = cubeSamples( 19, 500 );
    for ( const V2& d : dirs )
    {
        const double rho = z.support( d );
        for ( const V2& xi : pts )
        {
            const V2 p = evalState( x, xi );
            EXPECT_LE( d.dot( p ), rho + 1e-12 );
        }
    }
}

TEST( Enclosure, ZonotopeEnclosureTighterThanIntervalHull )
{
    const DAState x = randomState( 44 );
    const auto z = tax::domain::zonotopeEnclosure( x, std::array< int, 2 >{ 0, 1 } );
    const auto hz = z.intervalHull();
    const auto hx = tax::domain::intervalHull( x );
    for ( int i = 0; i < D; ++i ) EXPECT_LE( hz.halfWidth( i ), hx.halfWidth( i ) + 1e-15 );
}

TEST( Enclosure, EvenExponentTermsShiftTheCenter )
{
    // x0(ξ) = c + g·ξ0², ξ0² ∈ [0,1] ⇒ range [c, c+g]: center c+g/2, radius g/2.
    DAState x;
    x( 0 )[0] = 1.0;
    tax::MultiIndex< M > e{};
    e[0] = 2;
    x( 0 )[tax::flatIndex< M >( e )] = 0.4;
    x( 1 )[0] = 0.0;
    const auto z = tax::domain::zonotopeEnclosure( x, std::array< int, 2 >{ 0, 1 } );
    EXPECT_DOUBLE_EQ( z.center( 0 ), 1.2 );
    const auto hull = z.intervalHull();
    EXPECT_DOUBLE_EQ( hull.halfWidth( 0 ), 0.2 );
}

// ---------------------------------------------------------------------------
// zonotopeFrame (exact degree-1 part)
// ---------------------------------------------------------------------------

TEST( Enclosure, ZonotopeFrameExtractsLinearPart )
{
    DAState x;
    x( 0 )[0] = 1.0;
    x( 1 )[0] = -2.0;
    Mat2 G;
    G << 0.5, -0.1, 0.2, 0.7;
    for ( int i = 0; i < D; ++i )
        for ( int j = 0; j < M; ++j )
        {
            tax::MultiIndex< M > e{};
            e[static_cast< std::size_t >( j )] = 1;
            x( i )[tax::flatIndex< M >( e )] = G( i, j );
        }
    const auto z = tax::domain::zonotopeFrame( x );
    EXPECT_DOUBLE_EQ( z.center( 0 ), 1.0 );
    EXPECT_DOUBLE_EQ( z.center( 1 ), -2.0 );
    for ( int i = 0; i < D; ++i )
        for ( int j = 0; j < M; ++j ) EXPECT_DOUBLE_EQ( z.generators( i, j ), G( i, j ) );
}

// ---------------------------------------------------------------------------
// Zonotope::intervalHull and the ellipsoid covering helpers
// ---------------------------------------------------------------------------

TEST( Enclosure, ZonotopeIntervalHullIsL1RowNorm )
{
    Zonotope< double, M > z;
    z.center = V2{ 1.0, -1.0 };
    z.generators << 0.3, -0.2, 0.1, 0.4;
    const auto hull = z.intervalHull();
    EXPECT_DOUBLE_EQ( hull.halfWidth( 0 ), 0.5 );
    EXPECT_DOUBLE_EQ( hull.halfWidth( 1 ), 0.5 );
    // The hull must contain every sampled parallelotope point.
    for ( const V2& xi : cubeSamples( 5, 200 ) )
        EXPECT_TRUE( hull.contains( z.denormalize( xi ) ) );
}

TEST( Enclosure, EllipsoidCoverContainsEllipsoid )
{
    Mat2 L;
    L << 0.5, 0.0, 0.3, 0.2;  // Cholesky-like factor
    const V2 c{ 1.0, 2.0 };
    // Any orthogonal orientation R still covers the ellipsoid.
    const double th = 0.7;
    Mat2 R;
    R << std::cos( th ), -std::sin( th ), std::sin( th ), std::cos( th );
    const auto z0 = tax::domain::ellipsoidCover( c, L );
    const auto zR = tax::domain::ellipsoidCover( c, L, R );
    std::mt19937 gen{ 3 };
    std::uniform_real_distribution< double > u{ 0.0, 2.0 * M_PI };
    std::uniform_real_distribution< double > r{ 0.0, 1.0 };
    for ( int k = 0; k < 500; ++k )
    {
        const double a = u( gen );
        const double s = std::sqrt( r( gen ) );  // uniform on the disk
        const V2 p = c + L * V2{ s * std::cos( a ), s * std::sin( a ) };
        EXPECT_TRUE( z0.contains( p, 1e-9 ) );
        EXPECT_TRUE( zR.contains( p, 1e-9 ) );
    }
}

TEST( Enclosure, EllipsoidIntervalHullIsExactSupport )
{
    Mat2 L;
    L << 0.5, 0.0, 0.3, 0.2;
    const V2 c{ 0.0, 0.0 };
    const auto hull = tax::domain::ellipsoidIntervalHull( c, L );
    // Per axis the hull extent equals the L2 row norm = sup over the ball.
    EXPECT_DOUBLE_EQ( hull.halfWidth( 0 ), L.row( 0 ).norm() );
    EXPECT_DOUBLE_EQ( hull.halfWidth( 1 ), L.row( 1 ).norm() );
    // Attained: the maximizing ball point u = rowᵀ/‖row‖ lands on the face.
    for ( int i = 0; i < M; ++i )
    {
        const V2 u = L.row( i ).transpose() / L.row( i ).norm();
        const V2 p = L * u;
        EXPECT_NEAR( std::abs( p( i ) ), hull.halfWidth( i ), 1e-14 );
        EXPECT_TRUE( hull.contains( p ) );
    }
}
