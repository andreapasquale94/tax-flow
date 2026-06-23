// =============================================================================
// examples/flyby/common.hpp
//
// Shared problem definition for the planetary gravity-assist reachable-set
// example. I/O scaffolding lives in examples/common/output.hpp.
//
// Planar Sun-Jupiter CIRCULAR RESTRICTED THREE-BODY PROBLEM (CR3BP) in the
// synodic rotating frame, barycentre at the origin:
//   Sun     (primary)   at (-mu,     0)
//   Jupiter (secondary) at (1 - mu,  0)
//   State = (x, y, vx, vy),  mu = m_Jupiter / (m_Sun + m_Jupiter).
//
//   d/dt x  =  vx
//   d/dt y  =  vy
//   d/dt vx =  2 vy + x - (1-mu)(x+mu)/r1^3 - mu (x-1+mu)/r2^3
//   d/dt vy = -2 vx + y - (1-mu) y    /r1^3 - mu  y      /r2^3
//
// A spacecraft approaches Jupiter and performs a GRAVITY ASSIST. The mission
// gets to choose its AIM POINT: the incoming position is a free 2-D design
// variable inside a small square upstream of Jupiter (the velocity is fixed,
// aimed at the planet). As the aim sweeps the square the spacecraft passes
// Jupiter at different impact parameters, so the flyby deflects it by wildly
// different amounts — the turn angle is violently nonlinear near a grazing
// pass. That is the regime ADS is built for: a single Taylor map cannot cover
// the encounter, so Automatic Domain Splitting bisects the aim box, piling its
// splits onto the close-approach corner.
//
// The two aim coordinates are exactly the first two state components, so
// ads::create seeds the identity box on them with no extra machinery:
//
//     s = [ x, y, vx, vy ]   (D = 4),   aim box on (x, y)   (M = 2).
//
// One ADS propagation maps the whole aim square through the flyby; sampling the
// leaf flow maps yields the REACHABLE SET OF OUTCOMES — the achievable
// post-flyby heliocentric energies and outgoing directions the gravity assist
// can buy (see flyby.cpp).
// =============================================================================

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <tax/ads/box.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>
#include <utility>

#include "../common/output.hpp"

namespace example::flyby
{

using example::printBanner;

// ---- CR3BP system: Sun-Jupiter ---------------------------------------------
inline constexpr double kMu = 9.5388e-4;          // Jupiter / (Sun + Jupiter)
inline const double kJupiterX = 1.0 - kMu;        // secondary position on the x-axis
inline constexpr double kSunX = -9.5388e-4;       // = -kMu (Sun on the x-axis)
// Hill radius of the secondary: r_H = (mu/3)^(1/3) ~ 0.0683.
inline const double kHill = std::cbrt( kMu / 3.0 );

// ---- Encounter geometry -----------------------------------------------------
// The spacecraft starts a distance kRho0 up-RIGHT of Jupiter and flies in along
// (-1,-1)/sqrt2 toward it; the 45-degree obliquity keeps both aim axes active
// (neither is a pure along-track time shift).
inline const double kRho0 = 3.2 * kHill;                  // start distance from Jupiter
inline const double kInvSqrt2 = 0.70710678118654752440;  // 1/sqrt(2)

// ---- Aim-box design range ---------------------------------------------------
// Half-width of the square aim box (in CR3BP length units) about the nominal
// aim point. Chosen so the swept impact parameter ranges from a near-grazing
// pass to a distant one (~ one Hill radius).
inline const double kAimHalf = 0.55 * kHill;

// ---- Flyby-speed scenario ---------------------------------------------------
// The approach speed relative to Jupiter sets the assist strength: a slow pass
// turns much more than a fast one. Three presets bracket it.
struct Scenario
{
    const char* name;
    double vApproach;  // incoming speed relative to Jupiter (CR3BP units)
    const char* outfile;
};

inline const Scenario kSlow{ "slow", 0.26, "flyby_slow.json" };
inline const Scenario kMedium{ "medium", 0.36, "flyby_medium.json" };
inline const Scenario kFast{ "fast", 0.50, "flyby_fast.json" };

// ---- Right-hand side (planar CR3BP) ----------------------------------------
// Generic over the state type: same lambda for the scalar reference path and
// the DA-valued ADS state.
inline auto rhs( double mu = kMu )
{
    return [mu]( const auto& s, const auto& /*t*/ ) {
        using S = std::decay_t< decltype( s ) >;
        const auto x = s( 0 );
        const auto y = s( 1 );
        const auto vx = s( 2 );
        const auto vy = s( 3 );

        const auto x1 = x + mu;          // distance vector to the Sun  (x-comp)
        const auto x2 = x - ( 1.0 - mu );  // distance vector to Jupiter (x-comp)
        const auto r1_2 = x1 * x1 + y * y;
        const auto r2_2 = x2 * x2 + y * y;
        const auto r1_3 = r1_2 * sqrt( r1_2 );
        const auto r2_3 = r2_2 * sqrt( r2_2 );

        S out;
        out( 0 ) = vx;
        out( 1 ) = vy;
        out( 2 ) = 2.0 * vy + x - ( 1.0 - mu ) * x1 / r1_3 - mu * x2 / r2_3;
        out( 3 ) = -2.0 * vx + y - ( 1.0 - mu ) * y / r1_3 - mu * y / r2_3;
        return out;
    };
}

// ---- Nominal incoming aim point + velocity ---------------------------------
// Start up-right of Jupiter, heading in along (-1,-1)/sqrt2 at the scenario
// approach speed. The nominal aim is the box centre.
[[nodiscard]] inline tax::la::VecNT< 4, double > aimCenter( const Scenario& sc )
{
    const double px = kJupiterX + kRho0 * kInvSqrt2;
    const double py = 0.0 + kRho0 * kInvSqrt2;
    const double vx = -sc.vApproach * kInvSqrt2;
    const double vy = -sc.vApproach * kInvSqrt2;
    return tax::la::VecNT< 4, double >{ px, py, vx, vy };
}

// Aim box (M = 2): a square on the incoming position (x, y); velocity fixed.
[[nodiscard]] inline tax::ads::Box< double, 2 > aimBox()
{
    return tax::ads::Box< double, 2 >{ { 0.0, 0.0 }, { kAimHalf, kAimHalf } };
}

// Build a scalar incoming state from normalised aim coordinates xi in [-1,1]^2
// (for the trajectory-bundle sweep and validation).
[[nodiscard]] inline tax::la::VecNT< 4, double > aimState( const Scenario& sc, double xix,
                                                           double xiy )
{
    auto s = aimCenter( sc );
    s( 0 ) += kAimHalf * xix;
    s( 1 ) += kAimHalf * xiy;
    return s;
}

// ---- Gravity-assist outcome from a (downstream) state ----------------------
// Heliocentric (Sun-relative, inertial) specific orbital energy and the
// outgoing velocity direction in the rotating frame. Far from Jupiter the
// heliocentric energy is conserved, so its post-flyby value measures the
// assist. Rotating -> inertial: v_I = v_rot + omega x r, omega = +1 about z.
struct Outcome
{
    double energy;    // heliocentric specific energy (Sun two-body)
    double angleDeg;  // outgoing velocity direction in the rotating frame (deg)
};

[[nodiscard]] inline Outcome outcome( const tax::la::VecNT< 4, double >& s )
{
    const double x = s( 0 ), y = s( 1 ), vx = s( 2 ), vy = s( 3 );
    // Spacecraft and Sun inertial velocities (omega x r), then relative motion.
    const double vIx = vx - y, vIy = vy + x;          // spacecraft inertial velocity
    const double vSunx = 0.0, vSuny = -kMu;            // Sun inertial velocity (at (-mu,0))
    const double dvx = vIx - vSunx, dvy = vIy - vSuny;  // velocity relative to the Sun
    const double rsun = std::hypot( x - kSunX, y );     // distance to the Sun
    const double e = 0.5 * ( dvx * dvx + dvy * dvy ) - ( 1.0 - kMu ) / rsun;
    const double ang = std::atan2( vy, vx ) * 180.0 / M_PI;
    return { e, ang };
}

// Distance from the spacecraft to Jupiter (for closest-approach tracking).
[[nodiscard]] inline double distToJupiter( const tax::la::VecNT< 4, double >& s )
{
    return std::hypot( s( 0 ) - kJupiterX, s( 1 ) );
}

// ---- xorshift RNG (header-only, reproducible) -------------------------------
class Rng
{
   public:
    explicit Rng( std::uint64_t seed = 0x9E3779B97F4A7C15ULL ) : s_( seed ? seed : 1 ) {}
    [[nodiscard]] double uniform()
    {
        s_ ^= s_ << 13;
        s_ ^= s_ >> 7;
        s_ ^= s_ << 17;
        return ( s_ >> 11 ) * ( 1.0 / 9007199254740992.0 );
    }
    [[nodiscard]] double symmetric() { return 2.0 * uniform() - 1.0; }

   private:
    std::uint64_t s_;
};

}  // namespace example::flyby
