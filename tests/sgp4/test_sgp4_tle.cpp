// tests/sgp4/test_sgp4_tle.cpp
//
// TLE parsing: element values, unit conversions and epoch -> Julian date,
// for a near-earth and a deep-space TLE.

#include <gtest/gtest.h>

#include <tax/sgp4.hpp>

using namespace tax::sgp4;

namespace
{
// Canonical Vallado verification sat 00005 (near-earth).
constexpr const char* k5L1 =
    "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753";
constexpr const char* k5L2 =
    "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667";

// Deep-space 12-hour resonance sat 09880.
constexpr const char* k9880L1 =
    "1 09880U 77021A   06176.56157475  .00000421  00000-0  10000-3 0  9814";
constexpr const char* k9880L2 =
    "2 09880  64.5968 349.3786 7069051 270.0229  16.3320  2.00813614112380";
}  // namespace

TEST( Sgp4Tle, NearEarthElements )
{
    const Tle tle = Tle::parse( k5L1, k5L2 );

    EXPECT_EQ( tle.objectId, "00005" );
    EXPECT_EQ( tle.classification, 'U' );
    EXPECT_EQ( tle.epochyr, 0 );
    EXPECT_NEAR( tle.epochdays, 179.78495062, 1e-8 );

    // Angles convert degrees -> radians.
    EXPECT_NEAR( tle.inclo, 34.2682 * deg2rad, 1e-12 );
    EXPECT_NEAR( tle.nodeo, 348.7242 * deg2rad, 1e-12 );
    EXPECT_NEAR( tle.argpo, 331.7664 * deg2rad, 1e-12 );
    EXPECT_NEAR( tle.mo, 19.3264 * deg2rad, 1e-12 );
    EXPECT_NEAR( tle.ecco, 0.1859667, 1e-12 );

    // Mean motion converts rev/day -> rad/min (divide by 1440/2pi).
    const double xpdotp = 1440.0 / ( 2.0 * pi );
    EXPECT_NEAR( tle.no_kozai, 10.82419157 / xpdotp, 1e-12 );

    // bstar "28098-4" == 0.28098e-4.
    EXPECT_NEAR( tle.bstar, 0.28098e-4, 1e-12 );
}

TEST( Sgp4Tle, EpochToJulianDate )
{
    const Tle tle = Tle::parse( k5L1, k5L2 );
    // 2000-06-27 18:50:19.733 UTC -> JD 2451722.5 + 0.78495062.
    EXPECT_NEAR( tle.jdsatepoch, 2451722.5, 1e-9 );
    EXPECT_NEAR( tle.jdsatepochF, 0.78495062, 1e-8 );
}

TEST( Sgp4Tle, DeepSpaceEpochAndElements )
{
    const Tle tle = Tle::parse( k9880L1, k9880L2 );
    EXPECT_EQ( tle.objectId, "09880" );
    EXPECT_EQ( tle.epochyr, 6 );
    EXPECT_NEAR( tle.epochdays, 176.56157475, 1e-8 );
    EXPECT_NEAR( tle.ecco, 0.7069051, 1e-12 );
    EXPECT_NEAR( tle.inclo, 64.5968 * deg2rad, 1e-12 );
}
