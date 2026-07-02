// =============================================================================
// examples/wsb/common.hpp
//
// Shared Sun-Earth CR3BP fixture for the WSB box-propagation examples
// (taylor / ads / loads).
//
// IC centre = the tangent WSB-arrival reference trajectory found by
// wsb_search:
//   r_a   = 1 279 400 km   (Earth-orbit Kepler apoapsis)
//   omega = 329.97 deg     (argument of perigee, synodic frame)
//   perigee altitude       = 250 km
//
// The box halfwidth around this IC is 1 km on each position axis and
// 1 m/s on each velocity axis. The two examples that drive the figure
// vary the box on the (x, vy) face (boundaryToBox sets the y and vx
// indices to 0); change kBoundaryAxes if you'd like a different face.
// =============================================================================

#pragma once

#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <tax/domain/box.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>
#include <utility>
#include <vector>

#include "../common/output.hpp"

namespace example::wsb
{

// Re-export the shared I/O helpers into this namespace.
using example::printBanner;
using example::unitSquareBoundary;
using example::writeJsonArray;

// ---- Physical constants ----------------------------------------------------
inline constexpr double kSunEarthMu = 3.00348959632e-6;
inline constexpr double kAU_km = 149597870.7;
inline constexpr double kVelU_kms = 29.78469183;
inline constexpr double kTimeU_days = 58.13235;
inline constexpr double kGM_E_km3s2 = 398600.4418;
inline constexpr double kEarthR_km = 6378.0;
inline constexpr double kAlt_km = 250.0;
inline constexpr double kRp_km = kEarthR_km + kAlt_km;
inline constexpr double kRp = kRp_km / kAU_km;
inline constexpr double kMoonOrbitKm = 384400.0;
inline constexpr double kEarthX = 1.0 - kSunEarthMu;

inline double earthHillR() { return std::cbrt( kSunEarthMu / 3.0 ); }
inline double moonOrbitR() { return kMoonOrbitKm / kAU_km; }

// ---- Best WSB IC found by wsb_search ---------------------------------------
inline constexpr double kRaBestKm = 1279400.0;
inline constexpr double kOmegaBest = 329.97 * std::numbers::pi / 180.0;
inline constexpr double kTArrivalDays = 76.0;  // Moon-orbit interception

// ---- Right-hand side -------------------------------------------------------
inline auto rhs( double mu = kSunEarthMu )
{
    return [mu]( const auto& s, const auto& /*t*/ ) {
        using S = std::decay_t< decltype( s ) >;
        using V = typename S::Scalar;
        S out;
        const V x = s( 0 ), y = s( 1 ), vx = s( 2 ), vy = s( 3 );
        const V x1 = x + V( mu );
        const V x2 = x - V( 1.0 - mu );
        const V r1_2 = x1 * x1 + y * y;
        const V r2_2 = x2 * x2 + y * y;
        const V r1_3 = r1_2 * sqrt( r1_2 );
        const V r2_3 = r2_2 * sqrt( r2_2 );
        out( 0 ) = vx;
        out( 1 ) = vy;
        out( 2 ) = V( 2.0 ) * vy + x - V( 1.0 - mu ) * x1 / r1_3 - V( mu ) * x2 / r2_3;
        out( 3 ) = -V( 2.0 ) * vx + y - V( 1.0 - mu ) * y / r1_3 - V( mu ) * y / r2_3;
        return out;
    };
}

// ---- Best WSB centerpoint --------------------------------------------------
inline tax::la::VecNT< 4, double > icCenter()
{
    const double a_km = 0.5 * ( kRp_km + kRaBestKm );
    const double e = ( kRaBestKm - kRp_km ) / ( kRaBestKm + kRp_km );
    const double v_p_kms = std::sqrt( kGM_E_km3s2 * ( 1.0 + e ) / kRp_km );
    const double v_p = v_p_kms / kVelU_kms;
    const double cw = std::cos( kOmegaBest );
    const double sw = std::sin( kOmegaBest );

    const double rx_in = kRp * cw;
    const double ry_in = kRp * sw;
    const double vx_in = -v_p * sw;
    const double vy_in = v_p * cw;

    // synodic = inertial - omega_rot x r,  omega_rot = +1 ẑ
    const double vx_syn = vx_in + ry_in;
    const double vy_syn = vy_in - rx_in;

    (void)a_km;
    return tax::la::VecNT< 4, double >{ kEarthX + rx_in, ry_in, vx_syn, vy_syn };
}

// ---- Configurable IC box halfwidth (canonical units) -----------------------
//
//   1 km   in AU       = 1 / 149,597,870.7  ≈ 6.685e-9
//   1 m/s in AU/T      = 0.001 / 29.78469   ≈ 3.358e-5
inline const tax::la::VecNT< 4, double > kIcBoxHalfWidth{
    10.0 / kAU_km,      // x   : 1 km
    10.0 / kAU_km,      // y   : 1 km
    0.005 / kVelU_kms,  // vx  : 1 m/s
    0.005 / kVelU_kms   // vy  : 1 m/s
};

inline tax::domain::Box< double, 4 > icBox()
{
    return tax::domain::Box< double, 4 >{ icCenter(), kIcBoxHalfWidth };
}

// ---- Boundary helpers ------------------------------------------------------
// Vary on (x, vy); pin (y, vx) to 0. Change the index pattern below if
// you reconfigure kIcBoxHalfWidth to spread on different axes.
inline std::array< double, 4 > boundaryToBox( double a, double b ) { return { a, 0.0, 0.0, b }; }

}  // namespace example::wsb
