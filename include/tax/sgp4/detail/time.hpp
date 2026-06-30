// include/tax/sgp4/detail/time.hpp
//
// Epoch / sidereal-time utilities (Vallado's jday and gstime).  These depend
// only on the epoch, never on the orbital elements, so they are plain double.

#pragma once

#include <cmath>
#include <tax/sgp4/detail/scalar.hpp>

namespace tax::sgp4::detail
{

/// Julian date from a calendar UTC date/time, split into integer-ish day and
/// fractional parts (Vallado alg 14).  Outputs through `jd` and `jdfrac`.
inline void jday( int year, int mon, int day, int hr, int minute, double sec, double& jd,
                  double& jdfrac ) noexcept
{
    jd = 367.0 * year - std::floor( ( 7 * ( year + std::floor( ( mon + 9 ) / 12.0 ) ) ) * 0.25 ) +
         std::floor( 275 * mon / 9.0 ) + day + 1721013.5;
    jdfrac = ( sec + minute * 60.0 + hr * 3600.0 ) / 86400.0;

    if ( std::abs( jdfrac ) > 1.0 )
    {
        const double dtt = std::floor( jdfrac );
        jd += dtt;
        jdfrac -= dtt;
    }
}

/// Greenwich sidereal time [rad], 0..2pi, from a UT1 Julian date
/// (Vallado 2013, eq 3-45).
[[nodiscard]] inline double gstime( double jdut1 ) noexcept
{
    const double tut1 = ( jdut1 - 2451545.0 ) / 36525.0;
    double temp = -6.2e-6 * tut1 * tut1 * tut1 + 0.093104 * tut1 * tut1 +
                  ( 876600.0 * 3600 + 8640184.812866 ) * tut1 + 67310.54841;  // sec
    temp = mod( temp * deg2rad / 240.0, twopi );  // 1/240 = 360/86400, deg, to rad
    if ( temp < 0.0 ) temp += twopi;
    return temp;
}

}  // namespace tax::sgp4::detail
