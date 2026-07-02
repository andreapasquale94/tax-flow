// =============================================================================
// examples/hyperbola/common.hpp
//
// Shared problem definition for the hyperbolic-flyby tutorial (bplane_taylor,
// bplane_ads, snapshots). I/O scaffolding lives in examples/common/output.hpp.
//
// The model is the *spatial* (3-D) two-body problem in canonical units
// (GM = 1):
//
//    d/dt r = v
//    d/dt v = -r / |r|^3
//
// with a 6-D state s = (x, y, z, vx, vy, vz). Unlike the planar two_body
// examples this keeps all three spatial dimensions so the flyby has a genuine
// B-plane (the aiming plane perpendicular to the incoming asymptote).
//
// The reference orbit is an incoming hyperbola with eccentricity kEcc > 1 and
// periapsis radius kRp, tilted out of the x-y plane by (kInc, kArgP, kRAAN).
// The initial condition is placed at a true anomaly kNu0 < 0 (incoming leg);
// the integration runs symmetrically through periapsis (true anomaly 0) to the
// outgoing leg. Closest approach is detected as the root of r . v = 0.
//
// The B-plane machinery (incoming asymptote S-hat, impact-parameter vector B,
// and the (B.T, B.R) coordinates) is written generic over the scalar type, so
// the SAME code evaluates the nominal frame on a double state and the
// Taylor-expanded B-vector on a DA (TaylorExpansion) state.
// =============================================================================

#pragma once

#include <array>
#include <cmath>
#include <tax/ads/domains/box.hpp>
#include <tax/ads/domains/polynomial_zonotope.hpp>
#include <tax/ads/domains/zonotope.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>

#include "../common/output.hpp"

namespace example::hyperbola
{

// Re-export the shared I/O helpers into this namespace.
using example::printBanner;
using example::unitSquareBoundary;
using example::writeJsonArray;

// ---- Orbit constants -------------------------------------------------------
inline constexpr double kEcc = 1.5;   // hyperbolic eccentricity (> 1)
inline constexpr double kRp = 1.0;    // periapsis radius
inline const double kNu0 = -120.0 * M_PI / 180.0;  // incoming true anomaly

// Orientation of the orbit plane (perifocal -> inertial): Rz(RAAN) Rx(inc) Rz(argP).
inline const double kInc = 45.0 * M_PI / 180.0;
inline const double kArgP = 30.0 * M_PI / 180.0;
inline const double kRAAN = 15.0 * M_PI / 180.0;

// Semi-latus rectum p = rp (1 + e) and semi-major axis a = rp / (1 - e) (< 0).
inline double semiLatus() { return kRp * ( 1.0 + kEcc ); }
inline double semiMajor() { return kRp / ( 1.0 - kEcc ); }

// ---- Right-hand side (6-D, GM = 1) -----------------------------------------
//
// Generic over the state type so the same lambda accepts:
//   * tax::la::VecNT<6, double>          (scalar reference path)
//   * tax::la::VecNT<6, tax::TE<P, M>>   (DA-valued state).
// ADL picks up tax::sqrt for TE; <cmath> provides ::sqrt for double.
inline auto rhs()
{
    return []( const auto& s, const auto& /*t*/ ) {
        using S = std::decay_t< decltype( s ) >;
        using V = typename S::Scalar;
        const V x = s( 0 ), y = s( 1 ), z = s( 2 );
        const V r2 = x * x + y * y + z * z;
        const V r3 = r2 * sqrt( r2 );  // r^3 = r^2 * r

        S out;
        out( 0 ) = s( 3 );   // dx/dt = vx
        out( 1 ) = s( 4 );   // dy/dt = vy
        out( 2 ) = s( 5 );   // dz/dt = vz
        out( 3 ) = -x / r3;  // dvx/dt = -x/r^3
        out( 4 ) = -y / r3;  // dvy/dt = -y/r^3
        out( 5 ) = -z / r3;  // dvz/dt = -z/r^3
        return out;
    };
}

// ---- Orbital-element -> Cartesian ------------------------------------------
//
// State on the reference hyperbola at true anomaly nu, rotated into the
// inertial frame. Used to seed the incoming initial condition (nu = kNu0).
inline tax::la::VecNT< 6, double > stateAtTrueAnomaly( double nu )
{
    using M3 = tax::la::MatNT< 3, double >;
    using V3 = tax::la::VecNT< 3, double >;

    const double e = kEcc;
    const double p = semiLatus();
    const double r = p / ( 1.0 + e * std::cos( nu ) );
    const double vs = 1.0 / std::sqrt( p );  // sqrt(GM / p), GM = 1

    // Perifocal position / velocity (periapsis along +x_pf).
    const V3 r_pf{ r * std::cos( nu ), r * std::sin( nu ), 0.0 };
    const V3 v_pf{ -vs * std::sin( nu ), vs * ( e + std::cos( nu ) ), 0.0 };

    auto Rz = []( double a ) {
        M3 m;
        m << std::cos( a ), -std::sin( a ), 0.0, std::sin( a ), std::cos( a ), 0.0, 0.0, 0.0, 1.0;
        return m;
    };
    auto Rx = []( double a ) {
        M3 m;
        m << 1.0, 0.0, 0.0, 0.0, std::cos( a ), -std::sin( a ), 0.0, std::sin( a ), std::cos( a );
        return m;
    };
    const M3 R = Rz( kRAAN ) * Rx( kInc ) * Rz( kArgP );
    const V3 r_in = R * r_pf;
    const V3 v_in = R * v_pf;

    tax::la::VecNT< 6, double > s;
    s << r_in( 0 ), r_in( 1 ), r_in( 2 ), v_in( 0 ), v_in( 1 ), v_in( 2 );
    return s;
}

// Signed time from periapsis to true anomaly nu (hyperbolic Kepler equation).
// nu < 0 (incoming) returns a negative time.
inline double timeFromPeriapsis( double nu )
{
    const double e = kEcc;
    const double H = 2.0 * std::atanh( std::sqrt( ( e - 1.0 ) / ( e + 1.0 ) ) * std::tan( nu / 2.0 ) );
    const double Mh = e * std::sinh( H ) - H;             // hyperbolic mean anomaly
    const double n = std::sqrt( 1.0 / std::pow( -semiMajor(), 3.0 ) );  // mean motion
    return Mh / n;
}

inline tax::la::VecNT< 6, double > icCenter() { return stateAtTrueAnomaly( kNu0 ); }

// Time (from the incoming IC at t = 0) to reach periapsis, and the symmetric
// final time on the outgoing leg (true anomaly -kNu0).
inline double tPeri() { return -timeFromPeriapsis( kNu0 ); }
inline double tFinal() { return 2.0 * tPeri(); }

// ---- Initial-condition uncertainty (2-D transverse position dispersion) ----
//
// The uncertainty is a delivery/navigation position error in the inertial
// (x, y) plane at the incoming epoch: an oriented rectangle (Zonotope) whose
// principal axes are tilted by kIcTilt, its axis-aligned bounding Box, and a
// gently curved PolynomialZonotope variant. Only state components 0 (x) and
// 1 (y) carry uncertainty (M = 2 factors); the rest stay pinned to the center.
inline constexpr int kM = 2;               // number of DA / uncertainty factors
inline constexpr double kIcHalfA = 0.28;   // half-width along principal axis 1
inline constexpr double kIcHalfB = 0.15;   // half-width along principal axis 2
inline const double kIcTilt = M_PI / 6.0;  // orientation of the (x, y) block

inline tax::ads::Zonotope< double, kM > icZonotope()
{
    tax::ads::Zonotope< double, kM > z;
    z.center = tax::la::VecNT< kM, double >{ icCenter()( 0 ), icCenter()( 1 ) };
    const double c = std::cos( kIcTilt );
    const double s = std::sin( kIcTilt );
    z.generators( 0, 0 ) = c * kIcHalfA;
    z.generators( 0, 1 ) = -s * kIcHalfB;
    z.generators( 1, 0 ) = s * kIcHalfA;
    z.generators( 1, 1 ) = c * kIcHalfB;
    return z;
}

// Axis-aligned Box tightly bounding icZonotope(): each half-width is the L1
// norm of the corresponding generator row.
inline tax::ads::Box< double, kM > icBox()
{
    const auto z = icZonotope();
    tax::la::VecNT< kM, double > hw;
    for ( int i = 0; i < kM; ++i ) hw( i ) = z.generators.row( i ).cwiseAbs().sum();
    return tax::ads::Box< double, kM >{ z.center, hw };
}

// Curved (polynomial) IC set: the bounding box plus a small quadratic xi_0^2
// term on the x component, so its x-boundary bends into a parabola. P must
// match the DA truncation order used to propagate it.
template < int P >
inline tax::ads::PolynomialZonotope< double, P, kM > icPolyZono()
{
    auto pz = tax::ads::PolynomialZonotope< double, P, kM >::fromBox( icBox() );
    tax::MultiIndex< kM > a{};
    a[0] = 2;  // total degree 2 on the xi_0 axis
    pz.value[0][tax::flatIndex< kM >( a )] = 0.12 * icBox().halfWidth( 0 );
    return pz;
}

// Boundary coordinates (a, b) in [-1, 1]^2 -> normalised M-D factor point.
inline std::array< double, kM > boundaryToBox( double a, double b ) { return { a, b }; }

// ---- Closest-approach event predicate --------------------------------------
//
// g(x, t) = r . v of the *center* trajectory (the DA constant terms). Its
// increasing zero crossing is periapsis (radial velocity turning from negative
// to positive). Returns a plain double, so RootFindingEvent<DAState, double, G>
// bisects on the center orbit even when the state is DA-valued.
inline auto radialVelocityOfCenter()
{
    return []( const auto& x, double /*t*/ ) -> double {
        return x( 0 )[0] * x( 3 )[0] + x( 1 )[0] * x( 4 )[0] + x( 2 )[0] * x( 5 )[0];
    };
}

// ---- B-plane geometry (generic over scalar type) ---------------------------
//
// Everything below is a Kepler orbit invariant except rmag / rv, which are
// evaluated wherever the state is sampled. For a DA state each field is a
// TaylorExpansion in the IC factors xi.
template < class S >
struct BData
{
    S Sx, Sy, Sz;       // incoming asymptote unit vector (S-hat)
    S Bx, By, Bz;       // impact-parameter (B) vector
    S ecc, vinf, bmag;  // eccentricity, hyperbolic excess speed, |B|
    S rmag, rv;         // radius and radial velocity r . v (event diagnostics)
};

template < class State >
[[nodiscard]] BData< typename State::Scalar > bData( const State& s )
{
    using S = typename State::Scalar;
    const S rx = s( 0 ), ry = s( 1 ), rz = s( 2 );
    const S vx = s( 3 ), vy = s( 4 ), vz = s( 5 );

    const S r2 = rx * rx + ry * ry + rz * rz;
    const S r = sqrt( r2 );
    const S rinv = S( 1.0 ) / r;
    const S v2 = vx * vx + vy * vy + vz * vz;
    const S rv = rx * vx + ry * vy + rz * vz;

    // Angular momentum h = r x v.
    const S hx = ry * vz - rz * vy;
    const S hy = rz * vx - rx * vz;
    const S hz = rx * vy - ry * vx;
    const S h = sqrt( hx * hx + hy * hy + hz * hz );
    const S hinv = S( 1.0 ) / h;

    // Eccentricity vector e = (v^2 - 1/r) r - (r.v) v   (GM = 1).
    const S coef = v2 - rinv;
    const S ex = coef * rx - rv * vx;
    const S ey = coef * ry - rv * vy;
    const S ez = coef * rz - rv * vz;
    const S e2 = ex * ex + ey * ey + ez * ez;
    const S ecc = sqrt( e2 );
    const S einv = S( 1.0 ) / ecc;

    // Unit vectors: e-hat (to periapsis), h-hat (orbit normal), p-hat = h x e.
    const S ehx = ex * einv, ehy = ey * einv, ehz = ez * einv;
    const S hhx = hx * hinv, hhy = hy * hinv, hhz = hz * hinv;
    const S phx = hhy * ehz - hhz * ehy;
    const S phy = hhz * ehx - hhx * ehz;
    const S phz = hhx * ehy - hhy * ehx;

    // Incoming asymptote velocity unit vector S-hat = (e-hat + sqrt(e^2-1) p-hat)/e.
    const S sq = sqrt( e2 - S( 1.0 ) );
    const S Sx = ( ehx + sq * phx ) * einv;
    const S Sy = ( ehy + sq * phy ) * einv;
    const S Sz = ( ehz + sq * phz ) * einv;

    // Hyperbolic excess speed and impact parameter b = h / vinf, direction S x h-hat.
    const S vinf = sqrt( v2 - S( 2.0 ) * rinv );
    const S bmag = h / vinf;
    const S bxx = Sy * hhz - Sz * hhy;
    const S bxy = Sz * hhx - Sx * hhz;
    const S bxz = Sx * hhy - Sy * hhx;

    BData< S > out;
    out.Sx = Sx;   out.Sy = Sy;   out.Sz = Sz;
    out.Bx = bmag * bxx;   out.By = bmag * bxy;   out.Bz = bmag * bxz;
    out.ecc = ecc; out.vinf = vinf; out.bmag = bmag;
    out.rmag = r;  out.rv = rv;
    return out;
}

// ---- Fixed nominal B-plane frame -------------------------------------------
//
// (S-hat, T-hat, R-hat) built from the center orbit: T-hat = (S x zhat)^, then
// R-hat = S x T. The DA B-vector is projected onto this fixed frame so the
// enclosure lives in constant (B.T, B.R) axes.
struct Frame
{
    std::array< double, 3 > S, T, R;
};

inline Frame nominalFrame( const tax::la::VecNT< 6, double >& center )
{
    const auto d = bData( center );
    const std::array< double, 3 > S{ d.Sx, d.Sy, d.Sz };
    // S x zhat = (Sy, -Sx, 0); normalise.
    std::array< double, 3 > T{ S[1], -S[0], 0.0 };
    const double tn = std::sqrt( T[0] * T[0] + T[1] * T[1] + T[2] * T[2] );
    for ( auto& c : T ) c /= tn;
    const std::array< double, 3 > R{ S[1] * T[2] - S[2] * T[1], S[2] * T[0] - S[0] * T[2],
                                     S[0] * T[1] - S[1] * T[0] };
    return { S, T, R };
}

// Project a (DA or scalar) B-vector onto a fixed frame axis.
template < class S >
[[nodiscard]] S project( const S& bx, const S& by, const S& bz, const std::array< double, 3 >& u )
{
    return bx * u[0] + by * u[1] + bz * u[2];
}

}  // namespace example::hyperbola
