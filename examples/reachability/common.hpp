// =============================================================================
// examples/reachability/common.hpp
//
// Shared problem definition for the low-thrust reachability example.
// I/O scaffolding lives in examples/common/output.hpp.
//
// Planar two-body in canonical units (mu = 1) on a CIRCULAR heliocentric
// orbit at the Earth semi-major axis: r0 = 1, v0 = 1, period T = 2*pi.
//
// A spacecraft applies a CONSTANT low-thrust acceleration over one orbit:
//
//     a = m * R(theta) * vhat,
//
// where m is the magnitude, vhat the unit velocity, R(theta) a planar
// rotation, and theta = 0 means prograde (thrust along the velocity).
//
// The two controls (m, theta) are the reachability parameters. They are
// carried as zero-dynamics state components so ADS's Box<double,2> axes
// seed them as DA variables, while the physical position lives in state
// components 2, 3:
//
//     s = [ m, theta, x, y, vx, vy ]   (D = 6),   M = 2 control axes.
//
// This is an illustrative reachable set (constant thrust over a
// revolution), not an optimal-control reachable set.
// =============================================================================

#pragma once

#include <array>
#include <cmath>
#include <tax/ads/domains/box.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>

#include "../common/output.hpp"

namespace example::reachability
{

using example::printBanner;
using example::unitSquareBoundary;

// ---- Orbit + control constants ---------------------------------------------
inline constexpr double kR0 = 1.0;         // circular radius (Earth sma, normalized)
inline constexpr double kV0 = 1.0;         // circular speed sqrt(mu/r0)
inline const double kPeriod = 2.0 * M_PI;  // one orbital period
// Acceleration unit: mu_sun / (1 AU)^2 = 5.9301e-3 m/s^2.
inline constexpr double kAccelUnit = 5.9301e-3;  // m/s^2 at 1 AU (= mu_sun / LU^2)
inline constexpr double kDaysPerYear = 365.25;
inline const double kTUperDay = 2.0 * M_PI / kDaysPerYear;  // 1 day in TU
inline constexpr int kSnapStepDays = 10;                    // snapshot cadence

// ---- Spacecraft presets ----------------------------------------------------
// A spacecraft preset: max thrust (N), mass (kg), a human label, and the
// output JSON filename. a_max (normalized) = (thrust/mass) / kAccelUnit.
struct Preset
{
    const char* name;
    double thrustN;
    double massKg;
    const char* outfile;
};

// 1000 kg spacecraft, 100 mN  -> a_max ~ 0.0169.
inline constexpr Preset kSpacecraft{ "1000 kg, 100 mN", 0.100, 1000.0, "reachability.json" };
// 24U CubeSat: 40 kg, 2 mN  -> a_max ~ 0.0084.
inline constexpr Preset kCubeSat{ "40 kg, 2 mN", 0.002, 40.0, "reachability_cubesat.json" };

[[nodiscard]] inline constexpr double aMax( const Preset& p )
{
    return ( p.thrustN / p.massKg ) / kAccelUnit;
}

// ---- Right-hand side --------------------------------------------------------
//
// Generic over the state type so the same lambda accepts:
//   * tax::la::VecNT<6, double>           (scalar reference path)
//   * tax::la::VecNT<6, tax::TE<P, 2>>    (DA-valued ADS state).
// ADL picks up tax::sqrt/cos/sin for TE; <cmath> provides the global
// ::sqrt/::cos/::sin for double.
inline auto rhs()
{
    return []( const auto& s, const auto& /*t*/ ) {
        using S = std::decay_t< decltype( s ) >;
        const auto m = s( 0 );   // thrust magnitude  (constant control)
        const auto th = s( 1 );  // thrust angle from velocity (constant control)
        const auto x = s( 2 );
        const auto y = s( 3 );
        const auto vx = s( 4 );
        const auto vy = s( 5 );

        const auto r2 = x * x + y * y;
        const auto r3 = r2 * sqrt( r2 );

        // Thrust direction = velocity direction rotated by theta.
        const auto vmag = sqrt( vx * vx + vy * vy );  // > 0 on a Kepler orbit
        const auto ux = vx / vmag;
        const auto uy = vy / vmag;
        const auto c = cos( th );
        const auto sn = sin( th );
        const auto dx = c * ux - sn * uy;
        const auto dy = sn * ux + c * uy;

        const auto zero = m - m;  // typed zero (works for double and TE)

        S out;
        out( 0 ) = zero;              // d(m)/dt     = 0
        out( 1 ) = zero;              // d(theta)/dt = 0
        out( 2 ) = vx;                // dx/dt  = vx
        out( 3 ) = vy;                // dy/dt  = vy
        out( 4 ) = -x / r3 + m * dx;  // dvx/dt = gravity + thrust_x
        out( 5 ) = -y / r3 + m * dy;  // dvy/dt = gravity + thrust_y
        return out;
    };
}

// ---- State IC (D = 6) -------------------------------------------------------
// Control center: mid-magnitude, theta = pi (so theta spans [0, 2*pi]).
// Physical center: the circular orbit periapsis-equivalent (x=1, vy=1).
// The first M = 2 entries must equal controlBox(a_max).center.
inline tax::la::VecNT< 6, double > icCenter( double a_max )
{
    return tax::la::VecNT< 6, double >{ a_max / 2.0, M_PI, kR0, 0.0, 0.0, kV0 };
}

// Ballistic (no-thrust) IC for the reference underlay: m = 0.
inline tax::la::VecNT< 6, double > ballisticCenter()
{
    return tax::la::VecNT< 6, double >{ 0.0, 0.0, kR0, 0.0, 0.0, kV0 };
}

// ---- Control box (M = 2) ----------------------------------------------------
// axis 0: m     in [0, a_max]   (center a_max/2, half-width a_max/2)
// axis 1: theta in [0, 2*pi]    (center pi,      half-width pi)
inline tax::ads::Box< double, 2 > controlBox( double a_max )
{
    return tax::ads::Box< double, 2 >{ { a_max / 2.0, M_PI }, { a_max / 2.0, M_PI } };
}

// ---- Boundary coordinates -> normalized M=2 control point -------------------
// The [-1,1]^2 boundary sample (a, b) is exactly the (xi_m, xi_theta) point
// fed to TE::eval; both control axes are active.
inline std::array< double, 2 > boundaryToBox( double a, double b ) { return { a, b }; }

}  // namespace example::reachability
