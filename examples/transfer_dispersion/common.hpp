// =============================================================================
// examples/transfer_dispersion/common.hpp
//
// Dispersion analysis of a low-thrust Earth -> NEA interplanetary transfer.
// I/O scaffolding lives in examples/common/output.hpp.
//
// Heliocentric planar two-body in canonical units (mu = 1, 1 DU = 1 AU, Earth
// circular speed 1, period 2*pi = 1 year). The spacecraft departs co-moving
// with Earth and flies a fixed THRUST - COAST - THRUST plan (the nominal
// transfer designed from a porkchop + finite-burn fit): a burn of duration
// tau1 along a constant inertial direction phi1, a ballistic coast, then a
// burn of duration tau2 along phi2, arriving on the NEA orbit.
//
// We propagate three uncertainty boxes through this nominal plan and read off
// the DISPERSION SET (the image of the box) along the trajectory:
//
//   1. initial : +-1000 km position, +-1 m/s velocity at departure
//   2. thrust  : +-2% magnitude, +-5 deg direction execution error (no IC disp.)
//   3. both    : initial + thrust together
//
// A single 6-D DA state carries every uncertainty as an expansion variable:
//
//     s = [ delta_m, delta_th, x, y, vx, vy ]   (D = 6, M = 6),
//
// and each case is just a choice of box half-widths (a zero half-width axis
// simply does not disperse). The flow map is carried arc-by-arc; sampling its
// polynomial over the box at each snapshot gives the dispersion-set cloud whose
// convex hull is plotted (see transfer_dispersion.cpp).
// =============================================================================

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <tax/ads/box.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>

#include "../common/output.hpp"

namespace example::transfer_dispersion
{

using example::printBanner;

// ---- Canonical units --------------------------------------------------------
inline constexpr double kAUkm = 1.495978707e8;  // 1 AU in km
inline constexpr double kVUkms = 29.7847;        // canonical speed unit (km/s)
inline const double kYear = 2.0 * M_PI;          // one year in TU

// ---- Departure epoch + Earth state (spacecraft co-moves with Earth) ---------
inline constexpr double kTd = 32.205271423734565;  // departure epoch (TU)

// State = [delta_m, delta_th, x, y, vx, vy]; physical IC = Earth at departure.
inline tax::la::VecNT< 6, double > stateIC()
{
    return tax::la::VecNT< 6, double >{ 0.0,           0.0,           0.7043105257,
                                        0.7098920224, -0.7098920224, 0.7043105257 };
}

// ---- Target NEA (Earth-crossing Amor, for the plotted reference orbit) ------
inline constexpr double kNeaA = 1.05, kNeaE = 0.0952, kNeaW = 0.872665, kNeaM0 = 1.047198;

// ---- Dispersion magnitudes --------------------------------------------------
inline const double kPos = 1000.0 / kAUkm;          // +-1000 km  (position)
inline const double kVel = 1.0 / ( kVUkms * 1000 );  // +-1 m/s    (velocity)
inline constexpr double kSigM = 0.02;               // +-2%       (thrust magnitude)
inline const double kSigTh = 5.0 * M_PI / 180.0;    // +-5 deg    (thrust direction)

// ---- Nominal thrust-coast-thrust per thrust level ---------------------------
// (a_lt, tau1, phi1, coast, tau2, phi2, T) from the porkchop + finite-burn fit.
struct Preset
{
    const char* name;
    double aLt, tau1, phi1, coast, tau2, phi2, T;
    const char* outfile;
};

inline const Preset kLow{ "0.10 N",  0.016863, 4.035285, 3.667623, 8.551102,
                          3.900709, 1.392032, 16.487096, "transfer_dispersion_low.json" };
inline const Preset kMed{ "0.20 N",  0.033726, 2.285716, 2.356362, 11.135560,
                          0.254067, -0.375321, 13.675343, "transfer_dispersion_med.json" };
inline const Preset kHigh{ "0.30 N", 0.050589, 0.689053, 2.161519, 15.228689,
                           0.569354, 2.790645, 16.487096, "transfer_dispersion_high.json" };

// ---- Dispersion case = a choice of box half-widths --------------------------
struct DispCase
{
    const char* name;
    double hdm, hdth, hpos, hvel;
};
inline const DispCase kInit{ "initial", 0.0, 0.0, kPos, kVel };
inline const DispCase kThrust{ "thrust", kSigM, kSigTh, 0.0, 0.0 };
inline const DispCase kBoth{ "both", kSigM, kSigTh, kPos, kVel };

[[nodiscard]] inline tax::ads::Box< double, 6 > dispBox( const DispCase& c )
{
    return tax::ads::Box< double, 6 >{ { 0, 0, 0, 0, 0, 0 },
                                       { c.hdm, c.hdth, c.hpos, c.hpos, c.hvel, c.hvel } };
}

// ---- Right-hand side --------------------------------------------------------
// Two-body gravity + constant-direction thrust on this arc. `magArc` is the
// commanded magnitude (a_lt during a burn, 0 during a coast); `phiArc` the
// inertial thrust direction. The execution errors perturb it:
//     a = (1 + delta_m) * magArc * (cos(phiArc + delta_th), sin(phiArc + delta_th)).
inline auto rhs( double magArc, double phiArc )
{
    return [magArc, phiArc]( const auto& s, const auto& /*t*/ ) {
        using S = std::decay_t< decltype( s ) >;
        const auto dm = s( 0 );
        const auto dth = s( 1 );
        const auto x = s( 2 );
        const auto y = s( 3 );
        const auto vx = s( 4 );
        const auto vy = s( 5 );

        const auto r2 = x * x + y * y;
        const auto r3 = r2 * sqrt( r2 );
        const auto mag = ( dm + 1.0 ) * magArc;
        const auto th = dth + phiArc;

        const auto zero = dm - dm;
        S out;
        out( 0 ) = zero;                       // d(delta_m)/dt  = 0
        out( 1 ) = zero;                       // d(delta_th)/dt = 0
        out( 2 ) = vx;
        out( 3 ) = vy;
        out( 4 ) = -x / r3 + mag * cos( th );  // gravity + thrust_x
        out( 5 ) = -y / r3 + mag * sin( th );  // gravity + thrust_y
        return out;
    };
}

// ---- xorshift RNG (reproducible box sampling) -------------------------------
class Rng
{
   public:
    explicit Rng( std::uint64_t seed = 0x9E3779B97F4A7C15ULL ) : s_( seed ? seed : 1 ) {}
    [[nodiscard]] double symmetric()  // in [-1, 1)
    {
        s_ ^= s_ << 13;
        s_ ^= s_ >> 7;
        s_ ^= s_ << 17;
        return 2.0 * ( ( s_ >> 11 ) * ( 1.0 / 9007199254740992.0 ) ) - 1.0;
    }

   private:
    std::uint64_t s_;
};

}  // namespace example::transfer_dispersion
