// tests/sgp4/test_sgp4_vallado.cpp
//
// Regression of the double propagator against Vallado's reference output
// (tcppver.out) for a representative subset of the SGP4-VER.TLE suite:
//   - sat 00005  near-earth SGP4
//   - sat 09880  deep-space, half-day geopotential resonance (irez = 2)
//   - sat 11801  deep-space, no resonance (irez = 0)
//   - sat 04632  deep-space, Lyddane long-period fix (low inclination)
// Expected position/velocity are copied verbatim from tcppver.out and matched
// to well below a metre.  opsmode 'a' (afspc) matches the reference output.

#include <gtest/gtest.h>

#include <cmath>
#include <tax/sgp4.hpp>

using namespace tax::sgp4;

namespace
{
struct Expect
{
    double t;     // minutes since epoch
    double r[3];  // km
    double v[3];  // km/s
};

void checkCase( const char* l1, const char* l2, const std::vector< Expect >& rows )
{
    const Tle tle = Tle::parse( l1, l2 );
    Satellite< double > sat( tle, GravModel::Wgs72, seedsFrom< double >( tle ), 'a' );
    for ( const auto& e : rows )
    {
        const auto s = sat.propagate( e.t );
        ASSERT_EQ( sat.error(), 0 ) << "propagation error at t=" << e.t;
        for ( int i = 0; i < 3; ++i )
        {
            EXPECT_NEAR( s.r( i ), e.r[i], 1e-4 ) << "r[" << i << "] at t=" << e.t;
            EXPECT_NEAR( s.v( i ), e.v[i], 1e-7 ) << "v[" << i << "] at t=" << e.t;
        }
    }
}
}  // namespace

TEST( Sgp4Vallado, NearEarth_Sat00005 )
{
    checkCase( "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
               "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667",
               { { 0.0,
                   { 7022.46529266, -1400.08296755, 0.03995155 },
                   { 1.893841015, 6.405893759, 4.534807250 } },
                 { 360.0,
                   { -7154.03120202, -3783.17682504, -3536.19412294 },
                   { 4.741887409, -4.151817765, -2.093935425 } } } );
}

TEST( Sgp4Vallado, DeepSpaceResonance_Sat09880 )
{
    checkCase( "1 09880U 77021A   06176.56157475  .00000421  00000-0  10000-3 0  9814",
               "2 09880  64.5968 349.3786 7069051 270.0229  16.3320  2.00813614112380",
               { { 0.0,
                   { 13020.06750784, -2449.07193500, 1.15896030 },
                   { 4.247363935, 1.597178501, 4.956708611 } },
                 { 120.0,
                   { 19190.32482476, 9249.01266902, 26596.71345328 },
                   { -0.624960193, 1.324550562, 2.495697637 } } } );
}

TEST( Sgp4Vallado, DeepSpace_Sat11801 )
{
    checkCase( "1 11801U          80230.29629788  .01431103  00000-0  14311-1      13",
               "2 11801  46.7916 230.4354 7318036  47.4722  10.4117  2.28537848    13",
               { { 0.0,
                   { 7473.37102491, 428.94748312, 5828.74846783 },
                   { 5.107155391, 6.444680305, -0.186133297 } },
                 { 360.0,
                   { -3305.22148694, 32410.84323331, -24697.16974954 },
                   { -1.301137319, -1.151315600, -0.283335823 } } } );
}

TEST( Sgp4Vallado, DeepSpaceLyddane_Sat04632 )
{
    checkCase( "1 04632U 70093B   04031.91070959 -.00000084  00000-0  10000-3 0  9955",
               "2 04632  11.4628 273.1101 1450506 207.6000 143.9350  1.20231981 44145",
               { { 0.0,
                   { 2334.11450085, -41920.44035349, -0.03867437 },
                   { 2.826321032, -0.065091664, 0.570936053 } },
                 { -5184.0,
                   { -29020.02587128, 13819.84419063, -5713.33679183 },
                   { -1.768068390, -3.235371192, -0.395206135 } } } );
}
