// =============================================================================
// examples/missed_thrust_reachable/common.hpp
//
// Shared problem definition for the SET-VALUED missed-thrust reachability
// example. I/O scaffolding lives in examples/common/output.hpp.
//
// This is the deterministic, set-valued counterpart of the probabilistic
// missed-thrust dispersion example (examples/missed_thrust/). Instead of
// Monte-Carlo sampling discrete Markov outage sequences, we ask the
// reachability question directly:
//
//     Over ONE heliocentric revolution, what region of the orbital plane
//     can the spacecraft occupy under ANY admissible thrust outage?
//
// An outage is parameterized by THREE continuous descriptors — the expansion
// variables of the DA flow map (M = 3):
//
//     (tau, w, d)
//
//   * tau  outage onset time                    in [0, T]
//   * w    outage duration                      in [0, w_max]
//   * d    outage depth (lost thrust fraction)  in [0, d_max]
//
// (The reachable set is outage-dominated: the small ±2% magnitude / ±5°
// pointing execution errors of the dispersion example only thicken its
// boundary marginally, so they are dropped here to concentrate the ADS split
// budget on the strongly nonlinear onset sweep. They could be re-added as two
// more axes at higher cost.)
//
// The delivered thrust fraction is a SMOOTH top-hat notch in time,
//
//     g(t) = 1/2 [ tanh((t - tau)/eps) - tanh((t - (tau + w))/eps) ] in [0, 1]
//     f(t) = 1 - d * g(t)                                            in [1-d, 1]
//
// so f is analytic in (tau, w, d) and the DA flow map expands in all three
// axes. The realised acceleration on the revolution is
//
//     a(t) = f(t) * m_nom * R(theta_nom) * vhat.
//
// The three axes are carried as zero-dynamics state components so ADS's
// Box<double,3> seeds them as DA variables, while the physical state lives in
// components 3..6:
//
//     s = [ tau, w, d, x, y, vx, vy ]   (D = 7), M = 3.
//
// A SINGLE ADS propagation (per snapshot time) covers the whole outage box;
// Automatic Domain Splitting subdivides it where the onset sweeping across the
// snapshot makes the flow map nonlinear. The reachable set is the union of the
// leaf images: sampling each leaf's polynomial over its sub-box and pooling the
// (x, y) points fills the reachable region, whose outer boundary is the
// reachable-set envelope (see missed_thrust_reachable.cpp).
//
// Three severity scenarios bound the outage budget (max duration / depth):
//   mild     : short, shallow outages
//   moderate : up to a half-revolution full outage
//   severe   : up to a three-quarter-revolution full outage
// =============================================================================

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <tax/ads/box.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>

#include "../common/output.hpp"

namespace example::missed_thrust_reachable
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

// Edge smoothing of the outage notch (in TU). About one 10-degree arc: the
// window edges have a finite ramp (a realistic thruster spin-down/up) and stay
// resolvable by a bounded number of onset splits.
inline const double kEdge = 1.0 * kArc;

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

// ---- Outage-severity scenario ----------------------------------------------
// Each scenario bounds the outage budget: the largest admissible duration
// (as a fraction of the orbital period) and depth (lost thrust fraction).
struct Scenario
{
    const char* name;
    double wMaxFrac;  // max outage duration as a fraction of the period
    double dMax;      // max outage depth (1 = total loss of thrust)
    const char* outfile;
};

//   mild     : short, shallow outages (quarter-rev, half depth).
//   moderate : up to a half-revolution full outage.
//   severe   : up to a three-quarter-revolution full outage.
inline const Scenario kMild{ "mild", 0.25, 0.5, "missed_thrust_reachable_mild.json" };
inline const Scenario kModerate{ "moderate", 0.50, 1.0, "missed_thrust_reachable_moderate.json" };
inline const Scenario kSevere{ "severe", 0.75, 1.0, "missed_thrust_reachable_severe.json" };

// ---- Right-hand side --------------------------------------------------------
//
// Generic over the state type so the same lambda accepts both the scalar
// reference path (VecNT<7,double>) and the DA-valued ADS state
// (VecNT<7, TE<P,3>>). `magNom` = m_nom is the nominal commanded magnitude;
// `thetaNom` the nominal direction. The three outage axes are read from the
// state; `t` (absolute time) drives the outage notch.
inline auto rhs( double magNom, double thetaNom )
{
    return [magNom, thetaNom]( const auto& s, const auto& t ) {
        using S = std::decay_t< decltype( s ) >;
        const auto tau = s( 0 );  // outage onset time
        const auto w = s( 1 );    // outage duration
        const auto d = s( 2 );    // outage depth (lost thrust fraction)
        const auto x = s( 3 );
        const auto y = s( 4 );
        const auto vx = s( 5 );
        const auto vy = s( 6 );

        const auto r2 = x * x + y * y;
        const auto r3 = r2 * sqrt( r2 );

        // Thrust direction = velocity direction rotated by theta_nom.
        const auto vmag = sqrt( vx * vx + vy * vy );  // > 0 on a Kepler orbit
        const auto ux = vx / vmag;
        const auto uy = vy / vmag;
        const auto c = cos( thetaNom );
        const auto sn = sin( thetaNom );
        const auto dx = c * ux - sn * uy;
        const auto dy = sn * ux + c * uy;

        // Smooth top-hat outage notch g(t) in [0,1]; delivered fraction f in [1-d,1].
        const auto g = 0.5 * ( tanh( ( t - tau ) / kEdge ) - tanh( ( t - ( tau + w ) ) / kEdge ) );
        const auto f = 1.0 - d * g;
        const auto mag = f * magNom;  // f * m_nom

        const auto zero = tau - tau;  // typed zero (double or TE)

        S out;
        out( 0 ) = zero;                // d(tau)/dt = 0
        out( 1 ) = zero;                // d(w)/dt   = 0
        out( 2 ) = zero;                // d(d)/dt   = 0
        out( 3 ) = vx;                  // dx/dt  = vx
        out( 4 ) = vy;                  // dy/dt  = vy
        out( 5 ) = -x / r3 + mag * dx;  // dvx/dt = gravity + thrust_x
        out( 6 ) = -y / r3 + mag * dy;  // dvy/dt = gravity + thrust_y
        return out;
    };
}

// ---- DA box (M = 3) and matching state IC (D = 7) ---------------------------
// axis 0: tau in [0, T]
// axis 1: w   in [0, w_max]
// axis 2: d   in [0, d_max]
[[nodiscard]] inline tax::ads::Box< double, 3 > outageBox( const Scenario& sc )
{
    const double wMax = sc.wMaxFrac * kPeriod;
    return tax::ads::Box< double, 3 >{ { kPeriod / 2.0, wMax / 2.0, sc.dMax / 2.0 },
                                       { kPeriod / 2.0, wMax / 2.0, sc.dMax / 2.0 } };
}

// State IC: the first M = 3 entries MUST equal outageBox(sc).center; the
// physical IC is the circular orbit (x = 1, vy = 1).
[[nodiscard]] inline tax::la::VecNT< 7, double > stateIC( const Scenario& sc )
{
    const double wMax = sc.wMaxFrac * kPeriod;
    return tax::la::VecNT< 7, double >{ kPeriod / 2.0, wMax / 2.0, sc.dMax / 2.0, kR0, 0.0,
                                        0.0,           kV0 };
}

// ---- Constant-fraction reference orbit (4-D, no outage machinery) ----------
// Integrate one revolution at a fixed delivered fraction (delta = 0):
// magBase = m_nom for the nominal (full-thrust) orbit, 0 for the ballistic
// (all-missed) orbit. Returns the (x, y) along the accepted-step grid.
[[nodiscard]] inline std::pair< std::vector< double >, std::vector< double > > referenceOrbit(
    double magBase, double thetaNom, const tax::ode::IntegratorConfig< double >& cfg )
{
    auto f = [magBase, thetaNom]( const auto& s, const auto& /*t*/ ) {
        using S = std::decay_t< decltype( s ) >;
        const auto x = s( 0 );
        const auto y = s( 1 );
        const auto vx = s( 2 );
        const auto vy = s( 3 );
        const auto r2 = x * x + y * y;
        const auto r3 = r2 * sqrt( r2 );
        const auto vmag = sqrt( vx * vx + vy * vy );
        const auto ux = vx / vmag;
        const auto uy = vy / vmag;
        const auto c = cos( thetaNom );
        const auto sn = sin( thetaNom );
        const auto dx = c * ux - sn * uy;
        const auto dy = sn * ux + c * uy;
        S out;
        out( 0 ) = vx;
        out( 1 ) = vy;
        out( 2 ) = -x / r3 + magBase * dx;
        out( 3 ) = -y / r3 + magBase * dy;
        return out;
    };
    tax::la::VecNT< 4, double > x0{ kR0, 0.0, 0.0, kV0 };
    auto sol = tax::ode::propagate( tax::ode::methods::Verner89{}, f, x0, 0.0, kPeriod, cfg );
    std::vector< double > xs, ys;
    xs.reserve( sol.x.size() );
    ys.reserve( sol.x.size() );
    for ( const auto& s : sol.x )
    {
        xs.push_back( s( 0 ) );
        ys.push_back( s( 1 ) );
    }
    return { std::move( xs ), std::move( ys ) };
}

// ---- xorshift RNG (header-only, reproducible) -------------------------------
class Rng
{
   public:
    explicit Rng( std::uint64_t seed = 0x9E3779B97F4A7C15ULL ) : s_( seed ? seed : 1 ) {}
    [[nodiscard]] double uniform()  // in [0, 1)
    {
        s_ ^= s_ << 13;
        s_ ^= s_ >> 7;
        s_ ^= s_ << 17;
        return ( s_ >> 11 ) * ( 1.0 / 9007199254740992.0 );
    }
    [[nodiscard]] double symmetric()  // in [-1, 1)
    {
        return 2.0 * uniform() - 1.0;
    }

   private:
    std::uint64_t s_;
};

}  // namespace example::missed_thrust_reachable
