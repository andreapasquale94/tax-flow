// include/tax/sgp4/tle.hpp
//
// Two-Line Element parsing.  A TLE is always parsed in double precision (the
// numbers come from fixed-column ASCII); the resulting nominal mean elements
// are then handed to a Satellite, which seeds them into an ElsetRec<T>.
//
// Field columns and unit conversions follow Vallado / aholinch's reference.

#pragma once

#include <cmath>
#include <cstdlib>
#include <string>
#include <string_view>
#include <tax/sgp4/detail/scalar.hpp>
#include <tax/sgp4/detail/time.hpp>

namespace tax::sgp4
{

/// Parsed TLE: nominal mean elements (SI-ish: radians, rad/min) plus epoch.
struct Tle
{
    std::string line1;
    std::string line2;
    std::string intlid;
    std::string objectId;
    char classification = 'U';

    int epochyr = 0;
    double epochdays = 0.0;
    double jdsatepoch = 0.0;   ///< Julian date of epoch (integer-ish part)
    double jdsatepochF = 0.0;  ///< Julian date fractional part
    long epochMillisSince1970 = 0;

    // ---- mean elements in the propagator's units ----
    double bstar = 0.0;     ///< drag term [1/earth radii]
    double ndot = 0.0;      ///< mean-motion 1st derivative [rad/min^2]
    double nddot = 0.0;     ///< mean-motion 2nd derivative [rad/min^3]
    double inclo = 0.0;     ///< inclination [rad]
    double nodeo = 0.0;     ///< RAAN [rad]
    double ecco = 0.0;      ///< eccentricity
    double argpo = 0.0;     ///< argument of perigee [rad]
    double mo = 0.0;        ///< mean anomaly [rad]
    double no_kozai = 0.0;  ///< Kozai mean motion [rad/min]

    long elnum = 0;
    long revnum = 0;

    [[nodiscard]] static Tle parse( std::string_view line1, std::string_view line2 );
};

namespace detail
{

/// Parse a plain double from the column range [i1, i2) of `s`.
[[nodiscard]] inline double gd( std::string_view s, int i1, int i2 )
{
    return std::stod( std::string( s.substr( i1, i2 - i1 ) ) );
}

/// Parse a double with an implied leading decimal ("0." + digits) from [i1, i2).
[[nodiscard]] inline double gdi( std::string_view s, int i1, int i2 )
{
    return std::stod( "0." + std::string( s.substr( i1, i2 - i1 ) ) );
}

[[nodiscard]] inline bool isLeap( int year )
{
    if ( year % 4 != 0 ) return false;
    if ( year % 100 == 0 ) return year % 400 == 0;
    return true;
}

/// Parse the epoch field (year + day-of-year fraction) into Julian date parts.
inline void parseEpoch( Tle& tle, std::string_view epochField )
{
    int year = std::stoi( std::string( epochField.substr( 0, 2 ) ) );
    tle.epochyr = year;
    year += ( year > 56 ) ? 1900 : 2000;

    int doy = std::stoi( std::string( epochField.substr( 2, 3 ) ) );
    // epochField[5] is the decimal point, so prepend just "0": "0" + ".7849..".
    double dfrac = std::stod( "0" + std::string( epochField.substr( 5, 9 ) ) );
    tle.epochdays = doy + dfrac;

    dfrac *= 24.0;
    int hr = static_cast< int >( dfrac );
    dfrac = 60.0 * ( dfrac - hr );
    int mn = static_cast< int >( dfrac );
    dfrac = 60.0 * ( dfrac - mn );
    int sc = static_cast< int >( dfrac );
    dfrac = 1000.0 * ( dfrac - sc );
    double sec = static_cast< double >( sc ) + dfrac / 1000.0;

    int days[12] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
    if ( isLeap( year ) ) days[1] = 29;
    int ind = 0;
    while ( ind < 12 && doy > days[ind] )
    {
        doy -= days[ind];
        ++ind;
    }
    int mon = ind + 1;
    int day = doy;
    jday( year, mon, day, hr, mn, sec, tle.jdsatepoch, tle.jdsatepochF );

    double diff = ( tle.jdsatepoch - 2440587.5 ) * 86400000.0;
    double diff2 = 86400000.0 * tle.jdsatepochF;
    tle.epochMillisSince1970 = static_cast< long >( diff2 ) + static_cast< long >( diff );
}

}  // namespace detail

inline Tle Tle::parse( std::string_view l1, std::string_view l2 )
{
    using namespace detail;
    Tle tle;
    tle.line1 = std::string( l1.substr( 0, 69 ) );
    tle.line2 = std::string( l2.substr( 0, 69 ) );
    std::string_view line1 = tle.line1;
    std::string_view line2 = tle.line2;

    tle.intlid = std::string( line1.substr( 9, 8 ) );
    tle.classification = line1[7];
    tle.objectId = std::string( line1.substr( 2, 5 ) );

    double ndot = gdi( line1, 35, 44 );
    if ( line1[33] == '-' ) ndot = -ndot;

    double nddot = gdi( line1, 45, 50 );
    if ( line1[44] == '-' ) nddot = -nddot;
    nddot *= std::pow( 10.0, gd( line1, 50, 52 ) );

    double bstar = gdi( line1, 54, 59 );
    if ( line1[53] == '-' ) bstar = -bstar;
    bstar *= std::pow( 10.0, gd( line1, 59, 61 ) );

    tle.elnum = static_cast< long >( gd( line1, 64, 68 ) );

    double incDeg = gd( line2, 8, 16 );
    double raanDeg = gd( line2, 17, 25 );
    double ecc = gdi( line2, 26, 33 );
    double argpDeg = gd( line2, 34, 42 );
    double maDeg = gd( line2, 43, 51 );
    double n = gd( line2, 52, 63 );
    tle.revnum = static_cast< long >( gd( line2, 63, 68 ) );

    parseEpoch( tle, line1.substr( 18 ) );

    // ---- convert to the propagator's units (Vallado's setValsToRec) ----
    const double xpdotp = 1440.0 / ( 2.0 * pi );  // rev/day -> rad/min divisor
    tle.bstar = bstar;
    tle.inclo = incDeg * deg2rad;
    tle.nodeo = raanDeg * deg2rad;
    tle.argpo = argpDeg * deg2rad;
    tle.mo = maDeg * deg2rad;
    tle.ecco = ecc;
    tle.no_kozai = n / xpdotp;
    tle.ndot = ndot / ( xpdotp * 1440.0 );
    tle.nddot = nddot / ( xpdotp * 1440.0 * 1440.0 );
    return tle;
}

}  // namespace tax::sgp4
