// include/tax/sgp4/gravconst.hpp
//
// Gravity-model constants for SGP4 (WGS-72 low precision, WGS-72, WGS-84).
// These are pure `double` physical constants — independent of the satellite
// elements — so they stay scalar even when the propagator runs on a
// TaylorExpansion scalar.  Replaces Vallado's `getgravconst`.

#pragma once

#include <cmath>

namespace tax::sgp4
{

/// Which set of Earth gravity constants to use.  `Wgs72` is the common usage
/// and the convention the published TLEs are fit against.
enum class GravModel
{
    Wgs72old,  ///< WGS-72 low-precision SpaceTrack #3 constants
    Wgs72,     ///< WGS-72 constants (default)
    Wgs84      ///< WGS-84 constants
};

/// Earth/gravity constants consumed by the propagator.
struct GravConstants
{
    double tumin;          ///< minutes in one time unit
    double mu;             ///< Earth gravitational parameter [km^3/s^2]
    double radiusearthkm;  ///< Earth radius [km]
    double xke;            ///< reciprocal of tumin
    double j2;             ///< un-normalized 2nd zonal harmonic
    double j3;             ///< un-normalized 3rd zonal harmonic
    double j4;             ///< un-normalized 4th zonal harmonic
    double j3oj2;          ///< j3 / j2
};

/// Return the constants for `model` (mirrors Vallado's getgravconst).
[[nodiscard]] inline GravConstants gravConstants( GravModel model ) noexcept
{
    GravConstants c{};
    switch ( model )
    {
        case GravModel::Wgs72old:
            c.mu = 398600.79964;
            c.radiusearthkm = 6378.135;
            c.xke = 0.0743669161;
            c.tumin = 1.0 / c.xke;
            c.j2 = 0.001082616;
            c.j3 = -0.00000253881;
            c.j4 = -0.00000165597;
            break;
        case GravModel::Wgs84:
            c.mu = 398600.5;
            c.radiusearthkm = 6378.137;
            c.xke = 60.0 / std::sqrt( c.radiusearthkm * c.radiusearthkm * c.radiusearthkm / c.mu );
            c.tumin = 1.0 / c.xke;
            c.j2 = 0.00108262998905;
            c.j3 = -0.00000253215306;
            c.j4 = -0.00000161098761;
            break;
        case GravModel::Wgs72:
        default:
            c.mu = 398600.8;
            c.radiusearthkm = 6378.135;
            c.xke = 60.0 / std::sqrt( c.radiusearthkm * c.radiusearthkm * c.radiusearthkm / c.mu );
            c.tumin = 1.0 / c.xke;
            c.j2 = 0.001082616;
            c.j3 = -0.00000253881;
            c.j4 = -0.00000165597;
            break;
    }
    c.j3oj2 = c.j3 / c.j2;
    return c;
}

}  // namespace tax::sgp4
