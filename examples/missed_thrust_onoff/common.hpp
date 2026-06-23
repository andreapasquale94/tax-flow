// =============================================================================
// examples/missed_thrust_onoff/common.hpp
//
// Shared problem definition for the ON/OFF missed-thrust dispersion example.
// I/O scaffolding lives in examples/common/output.hpp.
//
// A spacecraft on a CIRCULAR heliocentric orbit (canonical units, mu = 1,
// r0 = 1, v0 = 1, period T = 2*pi = 1 year) follows a NOMINAL plan of
// CONTINUOUS full thrust: a constant magnitude m_nom along a nominal direction
// theta_nom (measured from the instantaneous velocity, theta = 0 = prograde).
//
// The thruster is BANG-BANG: at any instant it is either fully ON or fully
// OFF. Switching is only allowed on a 4-DAY grid, so the thruster always holds
// its state for at least 4 days (minimum dwell). The realised plan is thus a
// binary on/off schedule over 4-day arcs; "missed thrust" is the OFF arcs.
//
// Whether an arc is ON or OFF follows a 2-STATE Markov chain that transitions
// once per 4-day arc:
//
//     ON  -> OFF  with probability p_fail     (the thruster trips off)
//     OFF -> ON   with probability p_recover  (the thruster recovers)
//
// On top of the binary schedule, each manoeuvre carries small EXECUTION ERRORS
// while the thruster is ON:
//
//   1. Magnitude execution error  delta_m  in [-sigma_m, +sigma_m]   (~2%)
//   2. Pointing  execution error  delta_th in [-sigma_th,+sigma_th]  (~5 deg)
//
// The realised acceleration on arc k is therefore
//
//     a = f_k * m_nom * (1 + delta_m) * R(theta_nom + delta_th) * vhat,
//
// with f_k in {0, 1} the ON/OFF state on arc k.
//
// The two execution errors are carried as zero-dynamics state components so the
// DA flow map expands in them, and the ON/OFF state multiplies the commanded
// magnitude (a per-arc constant). The state is
//
//     s = [ delta_m, delta_th, x, y, vx, vy ]   (D = 6),
//
// and the DA expansion variables are JUST the two execution errors (M = 2):
// the box is the small (delta_m, delta_th) rectangle. A Monte-Carlo sample is
// one on/off schedule; its DA state is propagated arc-by-arc, carried forward
// across the 4-day boundaries, so the integrator composes the per-arc flow maps
// for us. After each arc the physical components are Taylor polynomials in
// (delta_m, delta_th), and the execution-error "blob" at that snapshot is read
// off by evaluating those polynomials over the box — the polynomial surrogate
// that makes the per-schedule Monte Carlo cheap (see missed_thrust_onoff.cpp).
// =============================================================================

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <tax/ads/box.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>

#include "../common/output.hpp"

namespace example::missed_thrust_onoff
{

using example::printBanner;

// ---- Orbit + control constants ---------------------------------------------
inline constexpr double kR0 = 1.0;               // circular radius (Earth sma, normalized)
inline constexpr double kV0 = 1.0;               // circular speed sqrt(mu/r0)
inline const double kPeriod = 2.0 * M_PI;        // one orbital period (= 1 year)
inline constexpr double kAccelUnit = 5.9301e-3;  // m/s^2 at 1 AU (= mu_sun / LU^2)
inline constexpr double kDaysPerYear = 365.25;
inline const double kTUperDay = 2.0 * M_PI / kDaysPerYear;  // 1 day in TU

// ---- Decision grid: switch the thruster only every 4 days -------------------
inline constexpr double kArcDays = 4.0;           // 4-day minimum dwell / decision arc
inline const double kArc = kArcDays * kTUperDay;  // one 4-day arc, in TU
inline constexpr int kNArcs = 91;                 // 91 arcs of 4 days ~ one revolution
inline const double kHorizon = kNArcs * kArc;     // total propagated time (~364 days)

// ---- Execution-error magnitudes (continuous DA / surrogate variables) ------
inline constexpr double kSigmaM = 0.02;                    // +-2% magnitude error
inline constexpr double kSigmaTheta = 5.0 * M_PI / 180.0;  // +-5 deg pointing error

// ---- Spacecraft preset ------------------------------------------------------
// 1000 kg spacecraft, 100 mN -> a_max ~ 0.0169 (the nominal commanded accel).
struct Preset
{
    const char* name;
    double thrustN;
    double massKg;
    double thetaNomDeg;  // nominal thrust direction from velocity (deg)
};

inline constexpr Preset kSpacecraft{ "1000 kg, 100 mN, prograde", 0.100, 1000.0, 0.0 };

[[nodiscard]] inline constexpr double aMax( const Preset& p )
{
    return ( p.thrustN / p.massKg ) / kAccelUnit;
}

// ---- ON/OFF thruster Markov chain ------------------------------------------
// Two states: 0 = OFF (missed thrust), 1 = ON (nominal full thrust). The chain
// transitions once per 4-day arc, so the thruster always dwells at least 4 days
// in whichever state it is in. Starting ON, the stationary ON fraction is
// p_recover / (p_fail + p_recover).
struct ThrusterModel
{
    static constexpr int kNStates = 2;  // 0 = OFF, 1 = ON
    double pFail = 0.08;                // ON  -> OFF (thruster trips off)
    double pRecover = 0.30;             // OFF -> ON  (thruster recovers)
    int initState = 1;                  // start ON (nominal)

    [[nodiscard]] static constexpr double levelFrac( int s ) { return s == 1 ? 1.0 : 0.0; }

    // Sample the next state given the current one, with a uniform draw u in [0,1).
    [[nodiscard]] int next( int s, double u ) const
    {
        if ( s == 1 ) return u < pFail ? 0 : 1;  // ON: trip off with p_fail
        return u < pRecover ? 1 : 0;             // OFF: recover with p_recover
    }
};

// A named thruster-reliability scenario: a chain plus an output file. The three
// presets bracket the mission risk from a reliable thruster (rare, brief
// outages) to an unreliable one with frequent, slow-to-recover outages.
struct Scenario
{
    const char* name;
    ThrusterModel thruster;
    const char* outfile;
};

//   reliable     : rare trips, fast recovery     (~97% ON).
//   intermittent : occasional trips, moderate     (~79% ON).
//   unreliable   : frequent trips, slow recovery  (~43% ON).
inline const Scenario kReliable{ "reliable", ThrusterModel{ 0.02, 0.60, 1 },
                                 "missed_thrust_onoff_reliable.json" };
inline const Scenario kIntermittent{ "intermittent", ThrusterModel{ 0.08, 0.30, 1 },
                                     "missed_thrust_onoff_intermittent.json" };
inline const Scenario kUnreliable{ "unreliable", ThrusterModel{ 0.20, 0.15, 1 },
                                   "missed_thrust_onoff_unreliable.json" };

// ---- Right-hand side --------------------------------------------------------
//
// One generic lambda for both double (reference / validation) and TE (DA flow
// map) states. `magBase` = f_state * m_nom is the commanded magnitude on this
// arc (m_nom when ON, 0 when OFF); `thetaNom` the nominal direction. The
// execution errors are read from the state: delta_m = s(0), delta_th = s(1).
inline auto rhs( double magBase, double thetaNom )
{
    return [magBase, thetaNom]( const auto& s, const auto& /*t*/ ) {
        using S = std::decay_t< decltype( s ) >;
        const auto dm = s( 0 );   // magnitude execution error (fraction)
        const auto dth = s( 1 );  // pointing execution error (rad)
        const auto x = s( 2 );
        const auto y = s( 3 );
        const auto vx = s( 4 );
        const auto vy = s( 5 );

        const auto r2 = x * x + y * y;
        const auto r3 = r2 * sqrt( r2 );

        // Thrust direction = velocity direction rotated by (theta_nom + delta_th).
        const auto vmag = sqrt( vx * vx + vy * vy );  // > 0 on a Kepler orbit
        const auto ux = vx / vmag;
        const auto uy = vy / vmag;
        const auto th = dth + thetaNom;
        const auto c = cos( th );
        const auto sn = sin( th );
        const auto dx = c * ux - sn * uy;
        const auto dy = sn * ux + c * uy;

        const auto mag = ( dm + 1.0 ) * magBase;  // f_state * m_nom * (1 + delta_m)

        const auto zero = dm - dm;  // typed zero (double or TE)

        S out;
        out( 0 ) = zero;                // d(delta_m)/dt  = 0
        out( 1 ) = zero;                // d(delta_th)/dt = 0
        out( 2 ) = vx;                  // dx/dt  = vx
        out( 3 ) = vy;                  // dy/dt  = vy
        out( 4 ) = -x / r3 + mag * dx;  // dvx/dt = gravity + thrust_x
        out( 5 ) = -y / r3 + mag * dy;  // dvy/dt = gravity + thrust_y
        return out;
    };
}

// ---- State IC (D = 6) -------------------------------------------------------
// Physical IC = the circular orbit (x = 1, vy = 1); execution errors centered
// at zero. A Monte-Carlo sample overrides components 0, 1 with its own fixed
// (delta_m, delta_th) bias.
inline tax::la::VecNT< 6, double > stateIC( double dm = 0.0, double dth = 0.0 )
{
    return tax::la::VecNT< 6, double >{ dm, dth, kR0, 0.0, 0.0, kV0 };
}

// ---- xorshift RNG (header-only, reproducible, no <random> dependency) -------
class Rng
{
   public:
    explicit Rng( std::uint64_t seed = 0x9E3779B97F4A7C15ULL ) : s_( seed ? seed : 1 ) {}
    [[nodiscard]] double uniform()  // in [0, 1)
    {
        s_ ^= s_ << 13;
        s_ ^= s_ >> 7;
        s_ ^= s_ << 17;
        return ( s_ >> 11 ) * ( 1.0 / 9007199254740992.0 );  // 53-bit mantissa
    }
    // Uniform in [-h, +h].
    [[nodiscard]] double symmetric( double h ) { return ( 2.0 * uniform() - 1.0 ) * h; }

   private:
    std::uint64_t s_;
};

}  // namespace example::missed_thrust_onoff
