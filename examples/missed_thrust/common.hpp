// =============================================================================
// examples/missed_thrust/common.hpp
//
// Shared problem definition for the missed-thrust reachability example.
// I/O scaffolding lives in examples/common/output.hpp.
//
// A spacecraft on a CIRCULAR heliocentric orbit (canonical units, mu = 1,
// r0 = 1, v0 = 1, period T = 2*pi) follows a NOMINAL low-thrust plan: a
// constant magnitude m_nom applied along a nominal direction theta_nom
// (measured from the instantaneous velocity, theta = 0 = prograde).
//
// The realised thrust departs from that plan in three ways:
//
//   1. Magnitude execution error  delta_m  in [-sigma_m, +sigma_m]   (~2%)
//   2. Pointing  execution error  delta_th in [-sigma_th,+sigma_th]  (~5 deg)
//   3. MISSED THRUST: the delivered fraction of the commanded thrust is
//      governed by a 5-state Markov chain over {0, 25, 50, 75, 100}% that
//      transitions once per 10-degree arc (a sticky birth-death chain:
//      mostly nominal, occasional degradation that recovers over a few arcs).
//
// The realised acceleration on arc k is therefore
//
//     a = f_k * m_nom * (1 + delta_m) * R(theta_nom + delta_th) * vhat,
//
// with f_k in {0, .25, .5, .75, 1} the Markov thrust level on arc k.
//
// The two execution errors are carried as zero-dynamics state components so
// the DA flow map expands in them, and the Markov level multiplies the
// commanded magnitude (a per-arc constant). The state is
//
//     s = [ delta_m, delta_th, x, y, vx, vy ]   (D = 6),
//
// and the DA expansion variables are JUST the two execution errors (M = 2):
// the box is the small (delta_m, delta_th) rectangle. A Monte-Carlo sample is
// one missed-thrust sequence; its DA state is propagated arc-by-arc, carried
// forward across the 10-degree boundaries, so the integrator composes the
// per-arc flow maps for us. After each arc the physical components are Taylor
// polynomials in (delta_m, delta_th), and the execution-error "blob" at that
// snapshot is read off by evaluating those polynomials over the box — the
// polynomial surrogate that makes the per-sequence Monte Carlo cheap
// (see missed_thrust.cpp).
// =============================================================================

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <tax/domain/box.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>

#include "../common/output.hpp"

namespace example::missed_thrust
{

using example::printBanner;

// ---- Orbit + control constants ---------------------------------------------
inline constexpr double kR0 = 1.0;               // circular radius (Earth sma, normalized)
inline constexpr double kV0 = 1.0;               // circular speed sqrt(mu/r0)
inline const double kPeriod = 2.0 * M_PI;        // one orbital period
inline constexpr double kAccelUnit = 5.9301e-3;  // m/s^2 at 1 AU (= mu_sun / LU^2)

inline constexpr int kNArcs = 36;                     // 36 arcs of 10 deg each
inline const double kArc = kPeriod / kNArcs;          // one 10-degree arc, in TU
inline constexpr double kDegPerArc = 360.0 / kNArcs;  // 10 degrees per arc

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
    const char* outfile;
};

inline constexpr Preset kSpacecraft{ "1000 kg, 100 mN, prograde", 0.100, 1000.0, 0.0,
                                     "missed_thrust.json" };

[[nodiscard]] inline constexpr double aMax( const Preset& p )
{
    return ( p.thrustN / p.massKg ) / kAccelUnit;
}

// ---- Missed-thrust Markov chain --------------------------------------------
// Five thrust levels indexed 0..4; level i delivers fraction i/4 of the
// commanded thrust (0, 25, 50, 75, 100 %). State 4 (100%) is nominal.
//
// Sticky birth-death transition each arc:
//   * stay    with the remaining probability (high -> "sticky"),
//   * step DOWN one level with pDown   (degradation / outage onset),
//   * step UP   one level with pUp     (recovery toward nominal).
struct MarkovModel
{
    static constexpr int kNStates = 5;
    double pDown = 0.08;  // onset of degradation
    double pUp = 0.30;    // recovery toward nominal
    int initState = 4;    // start at nominal (100%)

    [[nodiscard]] static constexpr double levelFrac( int i ) { return 0.25 * i; }

    // Sample the next level given the current one, using a uniform draw u in
    // [0, 1). Boundary states cannot step past the {0, 4} ends.
    [[nodiscard]] int next( int i, double u ) const
    {
        const double down = ( i > 0 ) ? pDown : 0.0;
        const double up = ( i < kNStates - 1 ) ? pUp : 0.0;
        if ( u < down ) return i - 1;
        if ( u < down + up ) return i + 1;
        return i;
    }
};

// A named missed-thrust severity scenario: a Markov model plus an output file.
// The three presets bracket the mission risk from a reliable thruster
// (optimistic) to a frequently-degrading one with slow recovery (pessimistic).
struct Scenario
{
    const char* name;
    MarkovModel markov;
    const char* outfile;
};

//   optimistic   : rare degradation, fast recovery toward nominal.
//   intermediate : occasional degradation, moderate recovery.
//   pessimistic  : frequent degradation, slow recovery (clustered outages).
inline const Scenario kOptimistic{ "optimistic", MarkovModel{ 0.03, 0.50, 4 },
                                   "missed_thrust_optimistic.json" };
inline const Scenario kIntermediate{ "intermediate", MarkovModel{ 0.08, 0.30, 4 },
                                     "missed_thrust_intermediate.json" };
inline const Scenario kPessimistic{ "pessimistic", MarkovModel{ 0.20, 0.12, 4 },
                                    "missed_thrust_pessimistic.json" };

// ---- Right-hand side --------------------------------------------------------
//
// One generic lambda for both double (reference / validation) and TE (DA flow
// map) states. `magBase` = f_level * m_nom is the commanded magnitude on this
// arc (a constant); `thetaNom` the nominal direction. The execution errors are
// read from the state: delta_m = s(0), delta_th = s(1).
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

        const auto mag = ( dm + 1.0 ) * magBase;  // f_level * m_nom * (1 + delta_m)

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
// A small deterministic generator so the example is self-contained and its
// Monte-Carlo output is bit-reproducible across runs/platforms.
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

}  // namespace example::missed_thrust
