// =============================================================================
// examples/wsb/wsb_search.cpp
//
// Two-parameter shooting search for a Belbruno-style weak-stability
// boundary lunar transfer in the planar Sun-Earth CR3BP.
//
// Reference Keplerian orbit at Earth:
//   r_perigee = 6378 + 250 km                  (250 km altitude)
//   r_apogee  ∈ [1.2e6, 1.5e6] km              (WSB region near Hill sphere)
//
// Free parameters:
//   r_a       — Earth-orbit apoapsis (km)
//   omega     — argument of perigee in synodic frame (rad)
//
// Initial state in Sun-Earth synodic, prograde tangential burn at
// perigee:
//   r_inertial   = r_p * (cos w, sin w)            (relative to Earth)
//   v_inertial   = v_p * (-sin w, cos w)            (prograde tangential)
//   v_synodic    = v_inertial - omega_rot x r       (omega_rot = +1 ẑ)
//
// For each (r_a, omega) we propagate 150 days in the Sun-Earth CR3BP,
// find the first apogee (r > 0.4 * Hill), then look for the first
// inbound crossing of the Moon's mean orbital radius (384 400 km).
// The "tangency score" at the crossing is |v_r| / |v|, where v_r is
// the inertial radial velocity relative to Earth. Score = 0 means the
// arrival is perfectly tangent to the Moon's orbit, so a small
// matching impulse could capture into Moon orbit.
//
// We additionally require the arrival to be prograde (CCW around
// Earth in inertial), matching the Moon's orbital direction; trials
// with retrograde arrival are penalised.
//
// Run:    ./wsb_search
// Writes: wsb_search.json
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <string>
#include <string_view>
#include <vector>

#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <tax/ode/io.hpp>

namespace
{

using Vec4 = tax::la::VecNT< 4, double >;

inline constexpr double kSunEarthMu = 3.00348959632E-6;
inline constexpr double kAU_km      = 149597870.7;
inline constexpr double kVelU_kms   = 29.78469183;
inline constexpr double kTimeU_days = 58.13235;

inline constexpr double kEarthX     = 1.0 - kSunEarthMu;
inline constexpr double kEarthR_km  = 6378.0;
inline constexpr double kAlt_km     = 250.0;
inline constexpr double kRp_km      = kEarthR_km + kAlt_km;
inline constexpr double kRp         = kRp_km / kAU_km;
inline constexpr double kGM_E       = 398600.4418;
inline constexpr double kMoonOrbitKm = 384400.0;

inline double earthHillR() { return std::cbrt( kSunEarthMu / 3.0 ); }
inline double moonOrbitR() { return kMoonOrbitKm / kAU_km; }

inline auto rhs( double mu = kSunEarthMu )
{
    return [ mu ]( const auto& s, const auto& /*t*/ )
    {
        using S = std::decay_t< decltype( s ) >;
        using V = typename S::Scalar;
        S out;
        const V x = s( 0 ), y = s( 1 ), vx = s( 2 ), vy = s( 3 );
        const V x1   = x + V( mu );
        const V x2   = x - V( 1.0 - mu );
        const V r1_2 = x1 * x1 + y * y;
        const V r2_2 = x2 * x2 + y * y;
        const V r1_3 = r1_2 * sqrt( r1_2 );
        const V r2_3 = r2_2 * sqrt( r2_2 );
        out( 0 ) = vx;
        out( 1 ) = vy;
        out( 2 ) =  V( 2.0 ) * vy + x
                   - V( 1.0 - mu ) * x1 / r1_3 - V( mu ) * x2 / r2_3;
        out( 3 ) = -V( 2.0 ) * vx + y
                   - V( 1.0 - mu ) * y  / r1_3 - V( mu ) * y  / r2_3;
        return out;
    };
}

struct IcInfo
{
    Vec4    state;
    double  v_p_kms;
    double  C3_kms2;
    double  a_km, e;
    double  T_days;
};

inline IcInfo makeIc( double r_a_km, double omega )
{
    const double a_km    = 0.5 * ( kRp_km + r_a_km );
    const double e       = ( r_a_km - kRp_km ) / ( r_a_km + kRp_km );
    const double v_p_kms = std::sqrt( kGM_E * ( 1.0 + e ) / kRp_km );
    const double C3_kms2 = v_p_kms * v_p_kms - 2.0 * kGM_E / kRp_km;
    const double T_s     = 2.0 * std::numbers::pi
                         * std::sqrt( std::pow( a_km, 3.0 ) / kGM_E );

    const double v_p = v_p_kms / kVelU_kms;
    const double cw  = std::cos( omega );
    const double sw  = std::sin( omega );

    const double rx_in = kRp * cw;
    const double ry_in = kRp * sw;
    const double vx_in = -v_p * sw;
    const double vy_in =  v_p * cw;

    // omega_rot x r = (-r_y, +r_x)
    const double vx_syn = vx_in + ry_in;     // vx_in - (-r_y)
    const double vy_syn = vy_in - rx_in;     // vy_in - (+r_x)

    IcInfo info{};
    info.state << kEarthX + rx_in, ry_in, vx_syn, vy_syn;
    info.v_p_kms = v_p_kms;
    info.C3_kms2 = C3_kms2;
    info.a_km    = a_km;
    info.e       = e;
    info.T_days  = T_s / 86400.0;
    return info;
}

struct TrialResult
{
    double r_a_km;
    double omega_rad;
    bool   reached_apogee   = false;
    bool   reached_moon     = false;
    bool   prograde_arrival = false;
    double r_apogee_km      = 0.0;
    double t_apogee_days    = 0.0;
    double t_arrival_days   = 0.0;
    double vr_arrival_kms   = 0.0;
    double vt_arrival_kms   = 0.0;
    double v_arrival_kms    = 0.0;
    double score            = 1e9;
    Vec4   ic{};
    IcInfo ic_info{};
};

TrialResult scanOne( double r_a_km, double omega )
{
    TrialResult res;
    res.r_a_km    = r_a_km;
    res.omega_rad = omega;
    res.ic_info   = makeIc( r_a_km, omega );
    res.ic        = res.ic_info.state;

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol               = cfg.reltol = 1e-10;
    cfg.max_steps            = 50000000;
    cfg.max_rejects_per_step = 5000;
    cfg.initial_step         = 1.0e-9;       // very small for LEO-perigee start
    cfg.min_step             = 1.0e-14;

    using namespace tax::ode::methods;
    const double tFinal_days = 150.0;
    tax::ode::Solution< tax::ode::Feagin12Stepper< Vec4 >, Vec4, /*Dense=*/true > sol;
    try
    {
        sol = tax::ode::propagate< /*Dense=*/true >(
            Feagin12{}, rhs(), res.ic, 0.0, tFinal_days / kTimeU_days, cfg );
    }
    catch ( const std::exception& )
    {
        return res;     // integration failed; trial unusable
    }

    const double hill        = earthHillR();
    const double moon_orbit  = moonOrbitR();
    const double apogee_gate = 0.4 * hill;

    constexpr int N_sample = 4000;
    const auto    times    = tax::ode::linspace(
        0.0, tFinal_days / kTimeU_days, N_sample );

    bool   found_apogee   = false;
    double r_apogee_canon = 0.0;
    double t_apogee_canon = 0.0;
    double r_prev_canon   = 0.0;

    for ( int i = 0; i < N_sample; ++i )
    {
        const double t  = times[ static_cast< std::size_t >( i ) ];
        const Vec4   s  = sol( t );
        const double rx = s( 0 ) - kEarthX;
        const double ry = s( 1 );
        const double r  = std::hypot( rx, ry );

        if ( !found_apogee )
        {
            if ( r > r_apogee_canon )
            {
                r_apogee_canon = r;
                t_apogee_canon = t;
            }
            if ( i > 0 && r_apogee_canon > apogee_gate && r < r_prev_canon )
            {
                found_apogee = true;
                res.reached_apogee = true;
                res.r_apogee_km    = r_apogee_canon * kAU_km;
                res.t_apogee_days  = t_apogee_canon * kTimeU_days;
            }
        }
        else if ( !res.reached_moon )
        {
            if ( r_prev_canon > moon_orbit && r < moon_orbit )
            {
                const double t_prev = times[ static_cast< std::size_t >( i - 1 ) ];
                const double frac   = ( r_prev_canon - moon_orbit )
                                    / ( r_prev_canon - r );
                const double t_x    = t_prev + frac * ( t - t_prev );
                const Vec4   s_x    = sol( t_x );
                const double rxx    = s_x( 0 ) - kEarthX;
                const double ryy    = s_x( 1 );
                const double r_x    = std::hypot( rxx, ryy );

                // Inertial = synodic + omega_rot x r = (vx - r_y, vy + r_x)
                const double vx_in  = s_x( 2 ) - ryy;
                const double vy_in  = s_x( 3 ) + rxx;
                const double vr     = ( rxx * vx_in + ryy * vy_in ) / r_x;
                const double vt     = ( rxx * vy_in - ryy * vx_in ) / r_x;
                const double vmag   = std::hypot( vx_in, vy_in );

                res.reached_moon      = true;
                res.t_arrival_days    = t_x * kTimeU_days;
                res.vr_arrival_kms    = vr * kVelU_kms;
                res.vt_arrival_kms    = vt * kVelU_kms;
                res.v_arrival_kms     = vmag * kVelU_kms;
                res.prograde_arrival  = vt > 0.0;
                res.score             = std::abs( vr ) / vmag;
                if ( !res.prograde_arrival ) res.score += 10.0;
                break;
            }
        }
        r_prev_canon = r;
    }
    return res;
}

inline void writeArray( std::ostream& o, const std::vector< double >& v )
{
    o << '[';
    for ( std::size_t i = 0; i < v.size(); ++i )
    {
        if ( i ) o << ", ";
        o << v[ i ];
    }
    o << ']';
}

}  // namespace


namespace
{

inline double normaliseDeg( double w_deg )
{
    double w = std::fmod( w_deg, 360.0 );
    if ( w < 0.0 ) w += 360.0;
    return w;
}

struct Sweep
{
    std::vector< double >       r_as_km;
    std::vector< double >       omegas_rad;     // in [0, 2*pi)
    std::vector< TrialResult >  results;
    double                      wall_ms;
};

inline Sweep runSweep( std::vector< double > r_as_km,
                       std::vector< double > omegas_rad,
                       std::string_view      label )
{
    Sweep sw;
    sw.r_as_km    = std::move( r_as_km );
    sw.omegas_rad = std::move( omegas_rad );
    sw.results.reserve( sw.r_as_km.size() * sw.omegas_rad.size() );

    std::cout << "[wsb_search] " << label << " grid "
              << sw.r_as_km.size() << " x " << sw.omegas_rad.size() << " = "
              << ( sw.r_as_km.size() * sw.omegas_rad.size() ) << " trials..."
              << std::flush;

    const auto t0 = std::chrono::high_resolution_clock::now();
    for ( double r_a : sw.r_as_km )
        for ( double w : sw.omegas_rad )
            sw.results.push_back( scanOne( r_a, w ) );
    const auto t1 = std::chrono::high_resolution_clock::now();
    sw.wall_ms = std::chrono::duration< double, std::milli >( t1 - t0 ).count();

    int n_apogee = 0, n_moon = 0, n_pro = 0, n_success = 0;
    for ( const auto& r : sw.results )
    {
        if ( r.reached_apogee )   ++n_apogee;
        if ( r.reached_moon )     ++n_moon;
        if ( r.prograde_arrival ) ++n_pro;
        if ( r.score < 1e8 )      ++n_success;
    }
    std::cout << " done in " << ( sw.wall_ms / 1e3 ) << " s"
              << "  (apogee=" << n_apogee << ", Moon=" << n_moon
              << ", pro=" << n_pro << ", valid=" << n_success
              << " / " << sw.results.size() << ")\n";
    return sw;
}

inline void writeSweepArray( std::ostream& out, const std::vector< TrialResult >& v )
{
    out << "[\n";
    for ( std::size_t i = 0; i < v.size(); ++i )
    {
        const auto& r = v[ i ];
        out << "    { "
            << "\"r_a_km\": "    << r.r_a_km    << ", "
            << "\"omega_deg\": " << ( r.omega_rad * 180.0 / std::numbers::pi ) << ", "
            << "\"reached_apogee\": " << ( r.reached_apogee ? "true" : "false" ) << ", "
            << "\"reached_moon\": "   << ( r.reached_moon   ? "true" : "false" ) << ", "
            << "\"prograde\": "       << ( r.prograde_arrival ? "true" : "false" ) << ", "
            << "\"score\": "          << r.score          << ", "
            << "\"r_apogee_km\": "    << r.r_apogee_km    << ", "
            << "\"t_apogee_days\": "  << r.t_apogee_days  << ", "
            << "\"t_arrival_days\": " << r.t_arrival_days << ", "
            << "\"vr_arrival_kms\": " << r.vr_arrival_kms << ", "
            << "\"vt_arrival_kms\": " << r.vt_arrival_kms << ", "
            << "\"v_arrival_kms\": "  << r.v_arrival_kms
            << " }" << ( i + 1 < v.size() ? "," : "" ) << "\n";
    }
    out << "  ]";
}

}  // namespace


int main()
{
    // ---- Pass 1 — coarse sweep ----------------------------------------------
    std::vector< double > r_as_coarse;
    for ( double r = 1.20e6; r <= 1.501e6; r += 0.05e6 ) r_as_coarse.push_back( r );
    std::vector< double > omegas_coarse;
    constexpr int N_omega_coarse = 72;     // every 5 degrees
    for ( int i = 0; i < N_omega_coarse; ++i )
        omegas_coarse.push_back( 2.0 * std::numbers::pi
                               * static_cast< double >( i ) / N_omega_coarse );

    const Sweep sw_coarse = runSweep( r_as_coarse, omegas_coarse, "coarse" );

    auto coarse_best_it = std::min_element(
        sw_coarse.results.begin(), sw_coarse.results.end(),
        []( const TrialResult& a, const TrialResult& b )
        { return a.score < b.score; } );

    // ---- Pass 2 — fine sweep around the coarse minimum ----------------------
    const double coarse_best_r_a    = coarse_best_it->r_a_km;
    const double coarse_best_w_deg  = coarse_best_it->omega_rad * 180.0 / std::numbers::pi;

    std::vector< double > r_as_fine;
    {
        const double rmin = std::max( 1.05e6, coarse_best_r_a - 0.075e6 );
        const double rmax = std::min( 1.55e6, coarse_best_r_a + 0.075e6 );
        for ( double r = rmin; r <= rmax + 1.0; r += 0.0025e6 )
            r_as_fine.push_back( r );
    }
    std::vector< double > omegas_fine;
    {
        const double wmin = coarse_best_w_deg - 7.5;
        const double wmax = coarse_best_w_deg + 7.5;
        for ( double w = wmin; w <= wmax + 1e-9; w += 0.25 )
            omegas_fine.push_back( normaliseDeg( w ) * std::numbers::pi / 180.0 );
    }

    const Sweep sw_fine = runSweep( r_as_fine, omegas_fine, "fine  " );

    // ---- Pass 3 — ultra-fine sweep around the fine minimum ------------------
    auto fine_best_it = std::min_element(
        sw_fine.results.begin(), sw_fine.results.end(),
        []( const TrialResult& a, const TrialResult& b )
        { return a.score < b.score; } );

    const double fine_best_r_a    = fine_best_it->r_a_km;
    const double fine_best_w_deg  = fine_best_it->omega_rad * 180.0 / std::numbers::pi;

    std::vector< double > r_as_ultra;
    {
        const double rmin = std::max( 1.05e6, fine_best_r_a - 0.005e6 );
        const double rmax = std::min( 1.55e6, fine_best_r_a + 0.005e6 );
        for ( double r = rmin; r <= rmax + 1.0; r += 1.0e2 )    // 100 km
            r_as_ultra.push_back( r );
    }
    std::vector< double > omegas_ultra;
    {
        const double wmin = fine_best_w_deg - 0.5;
        const double wmax = fine_best_w_deg + 0.5;
        for ( double w = wmin; w <= wmax + 1e-9; w += 0.01 )    // 0.01 deg
            omegas_ultra.push_back( normaliseDeg( w ) * std::numbers::pi / 180.0 );
    }

    const Sweep sw_ultra = runSweep( r_as_ultra, omegas_ultra, "ultra " );

    auto ultra_best_it = std::min_element(
        sw_ultra.results.begin(), sw_ultra.results.end(),
        []( const TrialResult& a, const TrialResult& b )
        { return a.score < b.score; } );

    const TrialResult& best =
        ( ultra_best_it->score < fine_best_it->score )
            ? ( ultra_best_it->score < coarse_best_it->score
                    ? *ultra_best_it : *coarse_best_it )
            : ( fine_best_it->score < coarse_best_it->score
                    ? *fine_best_it  : *coarse_best_it );

    using namespace tax::ode::methods;
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol               = cfg.reltol = 1e-11;
    cfg.max_steps            = 50000000;
    cfg.max_rejects_per_step = 5000;
    cfg.initial_step         = 1.0e-9;
    cfg.min_step             = 1.0e-14;

    const double tFinal_days = 150.0;
    const double tFinal      = tFinal_days / kTimeU_days;
    auto sol_best = tax::ode::propagate< /*Dense=*/true >(
        Feagin12{}, rhs(), best.ic, 0.0, tFinal, cfg );

    constexpr int n_dense = 4000;
    const auto    times   = tax::ode::linspace( 0.0, tFinal, n_dense );

    std::vector< double > t, x_syn, y_syn, x_e, y_e;
    t.reserve( n_dense );
    x_syn.reserve( n_dense ); y_syn.reserve( n_dense );
    x_e.reserve( n_dense ); y_e.reserve( n_dense );
    for ( double tt : times )
    {
        const Vec4 s = sol_best( tt );
        t.push_back( tt );
        x_syn.push_back( s( 0 ) );
        y_syn.push_back( s( 1 ) );
        x_e.push_back( s( 0 ) - kEarthX );
        y_e.push_back( s( 1 ) );
    }

    std::ofstream out( "wsb_search.json" );
    out << std::setprecision( 14 );
    out << "{\n";
    out << "  \"problem\": \"sun_earth_cr3bp_wsb_search\",\n";
    out << "  \"config\": {\n";
    out << "    \"mu_SE\":             " << kSunEarthMu  << ",\n";
    out << "    \"earth_x\":           " << kEarthX      << ",\n";
    out << "    \"sun_x\":             " << ( -kSunEarthMu ) << ",\n";
    out << "    \"earth_hill_radius\": " << earthHillR() << ",\n";
    out << "    \"L1_x\":              " << ( kEarthX - earthHillR() ) << ",\n";
    out << "    \"L2_x\":              " << ( kEarthX + earthHillR() ) << ",\n";
    out << "    \"moon_orbit_AU\":     " << moonOrbitR() << ",\n";
    out << "    \"moon_orbit_km\":     " << kMoonOrbitKm << ",\n";
    out << "    \"AU_km\":             " << kAU_km       << ",\n";
    out << "    \"velocity_unit_kms\": " << kVelU_kms    << ",\n";
    out << "    \"time_unit_days\":    " << kTimeU_days  << ",\n";
    out << "    \"r_park_km\":         " << kRp_km       << ",\n";
    out << "    \"alt_km\":            " << kAlt_km      << "\n";
    out << "  },\n";

    out << "  \"sweep_coarse\": "; writeSweepArray( out, sw_coarse.results ); out << ",\n";
    out << "  \"sweep_fine\": ";   writeSweepArray( out, sw_fine.results );   out << ",\n";
    out << "  \"sweep_ultra\": ";  writeSweepArray( out, sw_ultra.results );  out << ",\n";

    out << "  \"best\": {\n";
    out << "    \"r_a_km\":         " << best.r_a_km << ",\n";
    out << "    \"omega_deg\":      " << ( best.omega_rad * 180.0 / std::numbers::pi ) << ",\n";
    out << "    \"C3_kms2\":        " << best.ic_info.C3_kms2 << ",\n";
    out << "    \"v_p_kms\":        " << best.ic_info.v_p_kms << ",\n";
    out << "    \"a_km\":           " << best.ic_info.a_km << ",\n";
    out << "    \"e\":              " << best.ic_info.e << ",\n";
    out << "    \"T_kepler_days\":  " << best.ic_info.T_days << ",\n";
    out << "    \"r_apogee_km\":    " << best.r_apogee_km << ",\n";
    out << "    \"t_apogee_days\":  " << best.t_apogee_days << ",\n";
    out << "    \"t_arrival_days\": " << best.t_arrival_days << ",\n";
    out << "    \"vr_arrival_kms\": " << best.vr_arrival_kms << ",\n";
    out << "    \"vt_arrival_kms\": " << best.vt_arrival_kms << ",\n";
    out << "    \"v_arrival_kms\":  " << best.v_arrival_kms << ",\n";
    out << "    \"score\":          " << best.score << ",\n";
    out << "    \"ic_state\":       [" << best.ic( 0 ) << ", " << best.ic( 1 )
        << ", "                        << best.ic( 2 ) << ", " << best.ic( 3 ) << "]\n";
    out << "  },\n";

    out << "  \"trajectory\": {\n";
    out << "    \"t\":       "; writeArray( out, t );     out << ",\n";
    out << "    \"x_syn\":   "; writeArray( out, x_syn ); out << ",\n";
    out << "    \"y_syn\":   "; writeArray( out, y_syn ); out << ",\n";
    out << "    \"x_earth\": "; writeArray( out, x_e );   out << ",\n";
    out << "    \"y_earth\": "; writeArray( out, y_e );   out << "\n";
    out << "  }\n";
    out << "}\n";

    constexpr std::size_t lw = 26;
    auto row = [ & ]( std::string_view label, std::string_view value )
    {
        std::cout << "  " << std::string( label.size() < lw ? lw - label.size() : 0, ' ' )
                  << label << " : " << value << '\n';
    };
    std::cout << "\n=== Sun-Earth WSB search — best trial ===\n";
    row( "coarse grid",         std::to_string( sw_coarse.r_as_km.size() ) + " x "
                              + std::to_string( sw_coarse.omegas_rad.size() ) + " ("
                              + std::to_string( sw_coarse.wall_ms / 1e3 ) + " s)" );
    row( "fine grid",           std::to_string( sw_fine.r_as_km.size() ) + " x "
                              + std::to_string( sw_fine.omegas_rad.size() ) + " ("
                              + std::to_string( sw_fine.wall_ms / 1e3 ) + " s)" );
    row( "ultra grid",          std::to_string( sw_ultra.r_as_km.size() ) + " x "
                              + std::to_string( sw_ultra.omegas_rad.size() ) + " ("
                              + std::to_string( sw_ultra.wall_ms / 1e3 ) + " s)" );
    row( "r_a (km)",            std::to_string( best.r_a_km ) );
    row( "omega (deg)",         std::to_string( best.omega_rad * 180.0 / std::numbers::pi ) );
    row( "C3 (km^2/s^2)",       std::to_string( best.ic_info.C3_kms2 ) );
    row( "v_p inertial (km/s)", std::to_string( best.ic_info.v_p_kms ) );
    row( "Kepler T (days)",     std::to_string( best.ic_info.T_days ) );
    row( "observed r_a (Mkm)",  std::to_string( best.r_apogee_km / 1e6 ) );
    row( "t apogee (days)",     std::to_string( best.t_apogee_days ) );
    row( "t arrival (days)",    std::to_string( best.t_arrival_days ) );
    row( "v_r arrival (km/s)",  std::to_string( best.vr_arrival_kms ) );
    row( "v_t arrival (km/s)",  std::to_string( best.vt_arrival_kms ) );
    row( "|v| arrival (km/s)",  std::to_string( best.v_arrival_kms ) );
    row( "score |v_r|/|v|",     std::to_string( best.score ) );
    row( "output",              "wsb_search.json" );
    std::cout << '\n';
    return 0;
}
