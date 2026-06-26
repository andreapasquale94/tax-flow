// =============================================================================
// examples/two_body/common.hpp
//
// Shared problem definition for the two-body examples (taylor, ads, loads,
// validation). I/O scaffolding lives in examples/common/output.hpp.
//
// The model is the planar Kepler problem in canonical units (GM = 1,
// semi-major axis a = 1):
//
//    d/dt (x, y)   = (vx, vy)
//    d/dt (vx, vy) = -(x, y) / r^3 ,    r = sqrt(x^2 + y^2)
//
// The reference IC is at periapsis of an eccentricity-0.5 ellipse:
//
//    x  = a(1 - e) = 0.5
//    vy = sqrt((1 + e)/(1 - e)) = sqrt(3)
//
// One full orbital period is T = 2*pi.
// =============================================================================

#pragma once

#include <array>
#include <cmath>
#include <tax/ads/domains/box.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>
#include <utility>

#include "../common/output.hpp"

namespace example::two_body
{

// Re-export the shared I/O helpers into this namespace.
using example::printBanner;
using example::unitSquareBoundary;
using example::writeJsonArray;

// ---- Orbit constants -------------------------------------------------------
inline constexpr double kEcc = 0.5;
inline constexpr double kPeriapsis = 1.0 - kEcc;  // x(0)
inline const double kVPeriapsis = std::sqrt( ( 1.0 + kEcc ) / ( 1.0 - kEcc ) );
inline const double kPeriod = 2.0 * M_PI;

// ---- Right-hand side -------------------------------------------------------
//
// Generic over the state type so the same lambda accepts:
//   * tax::la::VecNT<4, double>            (scalar reference path)
//   * tax::la::VecNT<4, tax::TE<P, M>>     (DA-valued state).
// ADL picks up tax::sqrt for TE; <cmath> provides ::sqrt for double.
inline auto rhs()
{
    return []( const auto& s, const auto& /*t*/ ) {
        using S = std::decay_t< decltype( s ) >;
        const auto x = s( 0 );
        const auto y = s( 1 );
        const auto r2 = x * x + y * y;
        const auto r3 = r2 * sqrt( r2 );  // r^3 = r^2 * r

        S out;
        out( 0 ) = s( 2 );   // dx/dt  = vx
        out( 1 ) = s( 3 );   // dy/dt  = vy
        out( 2 ) = -x / r3;  // dvx/dt = -x/r^3
        out( 3 ) = -y / r3;  // dvy/dt = -y/r^3
        return out;
    };
}

inline tax::la::VecNT< 4, double > icCenter()
{
    return tax::la::VecNT< 4, double >{ kPeriapsis, 0.0, 0.0, kVPeriapsis };
}

// ---- IC box configuration --------------------------------------------------
//
// Edit kIcBoxHalfWidth to change the size of the initial-condition box used
// by all examples. Zero-half-width components pin the corresponding state
// axis to its centerpoint value.
//
// The defaults vary only the y position and the y-velocity — enough to
// produce visible distortion in one orbit at e = 0.5 without triggering
// excessive ADS subdivisions.
inline const tax::la::VecNT< 4, double > kIcBoxHalfWidth{ 0.0, 8e-3, 0.0, 2e-2 };

inline tax::ads::Box< double, 4 > icBox()
{
    return tax::ads::Box< double, 4 >{ icCenter(), kIcBoxHalfWidth };
}

// ---- Boundary coordinates -> normalised 4D displacement ---------------------
//
// The box varies along axes 1 (y) and 3 (vy); the two boundary coordinates
// map there and the pinned axes get 0.
inline std::array< double, 4 > boundaryToBox( double a, double b ) { return { 0.0, a, 0.0, b }; }

}  // namespace example::two_body
