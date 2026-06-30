// tests/sgp4/test_sgp4_taylor.cpp
//
// Taylor-expansion compatibility of the propagator.  Seeding the six mean
// orbital elements as identity expansions yields a polynomial map of the TEME
// state w.r.t. those elements.  We check:
//   (a) the constant part of the expanded propagation equals the plain double
//       propagation, and
//   (b) the first-order coefficients (the Jacobian d state / d element) match
//       central finite differences of the double propagator,
// for both a near-earth and a deep-space TLE.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <tax/sgp4.hpp>
#include <tax/tax.hpp>

using namespace tax::sgp4;

namespace
{
// Second-order expansion over the 6 mean elements (order 2 is enough to expose
// the linear term; the value/Jacobian checks only use orders 0 and 1).
using TE = tax::TaylorExpansion< double, tax::IsotropicScheme< 2, 6 > >;

// Variable order: 0 inclo, 1 nodeo, 2 ecco, 3 argpo, 4 mo, 5 no_kozai.
std::array< double, 6 > nominal( const Tle& tle )
{
    return { tle.inclo, tle.nodeo, tle.ecco, tle.argpo, tle.mo, tle.no_kozai };
}

Seeds< double > seedsWith( const Tle& tle, int idx, double value )
{
    auto s = seedsFrom< double >( tle );
    switch ( idx )
    {
        case 0:
            s.inclo = value;
            break;
        case 1:
            s.nodeo = value;
            break;
        case 2:
            s.ecco = value;
            break;
        case 3:
            s.argpo = value;
            break;
        case 4:
            s.mo = value;
            break;
        case 5:
            s.no_kozai = value;
            break;
    }
    return s;
}

// Plain-double r(component) with element `idx` set to `value`.
double propR( const Tle& tle, int idx, double value, double t, int comp )
{
    Satellite< double > sat( tle, GravModel::Wgs72, seedsWith( tle, idx, value ), 'a' );
    return sat.propagate( t ).r( comp );
}

void checkTaylor( const char* l1, const char* l2, double t )
{
    const Tle tle = Tle::parse( l1, l2 );
    const auto x0 = nominal( tle );

    // Seed each element as an identity expansion x_i = nominal_i + dx_i.
    Seeds< TE > seeds{
        TE( tle.bstar ),          TE( tle.ndot ),           TE( tle.nddot ),
        TE::variable( x0[0], 0 ), TE::variable( x0[1], 1 ), TE::variable( x0[2], 2 ),
        TE::variable( x0[3], 3 ), TE::variable( x0[4], 4 ), TE::variable( x0[5], 5 ) };
    Satellite< TE > satTE( tle, GravModel::Wgs72, seeds, 'a' );
    const auto sTE = satTE.propagate( t );

    Satellite< double > satD( tle, GravModel::Wgs72, seedsFrom< double >( tle ), 'a' );
    const auto sD = satD.propagate( t );
    ASSERT_EQ( satD.error(), 0 );

    // (a) constant part == double propagation.
    for ( int c = 0; c < 3; ++c )
    {
        EXPECT_NEAR( sTE.r( c ).value(), sD.r( c ), 1e-7 ) << "r[" << c << "] value";
        EXPECT_NEAR( sTE.v( c ).value(), sD.v( c ), 1e-10 ) << "v[" << c << "] value";
    }

    // (b) Jacobian d r[c] / d element[i] vs central finite difference.
    for ( int c = 0; c < 3; ++c )
    {
        const auto grad = sTE.r( c ).gradient();
        for ( int i = 0; i < 6; ++i )
        {
            const double h = 1e-7 * ( std::abs( x0[i] ) + 1.0 );
            const double fd =
                ( propR( tle, i, x0[i] + h, t, c ) - propR( tle, i, x0[i] - h, t, c ) ) / ( 2 * h );
            const double da = grad( i );
            const double scale = std::max( 1.0, std::abs( fd ) );
            EXPECT_NEAR( da, fd, 1e-4 * scale ) << "d r[" << c << "]/d x[" << i << "]";
        }
    }
}
}  // namespace

TEST( Sgp4Taylor, NearEarth_Sat00005 )
{
    checkTaylor( "1 00005U 58002B   00179.78495062  .00000023  00000-0  28098-4 0  4753",
                 "2 00005  34.2682 348.7242 1859667 331.7664  19.3264 10.82419157413667", 75.0 );
}

TEST( Sgp4Taylor, DeepSpace_Sat11801 )
{
    checkTaylor( "1 11801U          80230.29629788  .01431103  00000-0  14311-1      13",
                 "2 11801  46.7916 230.4354 7318036  47.4722  10.4117  2.28537848    13", 90.0 );
}
