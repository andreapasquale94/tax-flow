// =============================================================================
// examples/wsb/taylor.cpp
//
// Single multivariate Taylor flow polynomial of the WSB IC box, propagated
// through the Sun-Earth CR3BP from LEO perigee for ~150 days. Polygon
// boundary samples are written for snapshots every 5 days.
//
// Run:    ./wsb_taylor
// Writes: wsb_taylor.json
// =============================================================================

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <tax/ads/box.hpp>
#include <tax/ads/da_state.hpp>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>

#include "common.hpp"

int main()
{
    using namespace example::wsb;
    using namespace tax::ode::methods;

    constexpr int P = 4;
    constexpr int M = 4;
    constexpr int D = 4;

    using TE = tax::TE< P, M >;
    using DAState = tax::la::VecNT< D, TE >;

    constexpr double tFinal_days = kTArrivalDays;  // Moon-orbit interception (76 days)
    constexpr double snap_step = 5.0;
    const int kNSnaps = static_cast< int >( tFinal_days / snap_step ) + 1;
    constexpr int kNPerEdge = 24;

    auto ic_box = icBox();

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-11;
    cfg.max_steps = 50000000;
    cfg.max_rejects_per_step = 5000;
    cfg.initial_step = 1.0e-9;
    cfg.min_step = 1.0e-14;
    cfg.save_steps = true;

    DAState x0_da = tax::ads::create< P, M >( ic_box, icCenter() );
    const auto t0 = std::chrono::high_resolution_clock::now();
    auto sol = tax::ode::propagate( Feagin12{}, rhs(), x0_da, 0.0, tFinal_days / kTimeU_days, cfg );
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration< double, std::milli >( t1 - t0 ).count();

    // Scalar centerpoint reference orbit.
    auto ref_sol =
        tax::ode::propagate( Feagin12{}, rhs(), icCenter(), 0.0, tFinal_days / kTimeU_days, cfg );

    std::vector< double > times_canon( kNSnaps );
    for ( int i = 0; i < kNSnaps; ++i ) times_canon[i] = ( i * snap_step ) / kTimeU_days;

    const auto boundary = unitSquareBoundary( kNPerEdge );

    // Helper: find stored step with t closest to t_query in a solution.
    auto stateAt = [&]( const auto& s, double t_query ) -> decltype( s.x.front() ) {
        auto it = std::lower_bound( s.t.begin(), s.t.end(), t_query );
        if ( it == s.t.end() ) return s.x.back();
        if ( it == s.t.begin() ) return s.x.front();
        auto prev = std::prev( it );
        const std::size_t idx = ( t_query - *prev < *it - t_query )
                                    ? static_cast< std::size_t >( prev - s.t.begin() )
                                    : static_cast< std::size_t >( it - s.t.begin() );
        return s.x[idx];
    };

    std::ofstream out( "wsb_taylor.json" );
    out << std::setprecision( 14 );
    out << "{\n";
    out << "  \"method\": \"taylor\",\n";
    out << "  \"problem\": \"sun_earth_cr3bp_wsb\",\n";
    out << "  \"config\": {\n";
    out << "    \"mu_SE\":   " << kSunEarthMu << ",\n";
    out << "    \"earth_x\": " << kEarthX << ",\n";
    out << "    \"sun_x\":   " << ( -kSunEarthMu ) << ",\n";
    out << "    \"earth_hill_radius\": " << earthHillR() << ",\n";
    out << "    \"L1_x\":   " << ( kEarthX - earthHillR() ) << ",\n";
    out << "    \"L2_x\":   " << ( kEarthX + earthHillR() ) << ",\n";
    out << "    \"moon_orbit_AU\": " << moonOrbitR() << ",\n";
    out << "    \"AU_km\":  " << kAU_km << ",\n";
    out << "    \"velocity_unit_kms\": " << kVelU_kms << ",\n";
    out << "    \"time_unit_days\": " << kTimeU_days << ",\n";
    out << "    \"P\": " << P << ", \"M\": " << M << ", \"D\": " << D << ",\n";
    out << "    \"snap_step_days\": " << snap_step << ",\n";
    out << "    \"t_final_days\": " << tFinal_days << ",\n";
    out << "    \"ic_box\": {\n";
    out << "      \"center\":    ";
    writeJsonArray( out, ic_box.center );
    out << ",\n";
    out << "      \"halfWidth\": ";
    writeJsonArray( out, ic_box.halfWidth );
    out << "\n";
    out << "    }\n";
    out << "  },\n";
    out << "  \"timing\": { \"elapsed_ms\": " << ms << " },\n";

    // Reference orbit — iterate the stored grid.
    out << "  \"reference_orbit\": {\n";
    out << "    \"t\":  ";
    writeJsonArray( out, ref_sol.t );
    out << ",\n";
    std::vector< double > col( ref_sol.t.size() );
    for ( int j = 0; j < D; ++j )
    {
        for ( std::size_t i = 0; i < ref_sol.t.size(); ++i ) col[i] = ref_sol.x[i]( j );
        out << "    \"x" << j << "\": ";
        writeJsonArray( out, col );
        out << ( j + 1 < D ? ",\n" : "\n" );
    }
    out << "  },\n";

    // Polygon per snapshot.
    out << "  \"polygons\": [\n";
    std::vector< double > xs( boundary.size() ), ys( boundary.size() );
    for ( int s = 0; s < kNSnaps; ++s )
    {
        const double t = times_canon[s];
        const auto x_at_t = stateAt( sol, t );
        for ( std::size_t v = 0; v < boundary.size(); ++v )
        {
            const auto d = boundaryToBox( boundary[v][0], boundary[v][1] );
            xs[v] = x_at_t( 0 ).eval( d );
            ys[v] = x_at_t( 1 ).eval( d );
        }
        out << "    { \"t_days\": " << ( s * snap_step ) << ", \"x\": ";
        writeJsonArray( out, xs );
        out << ", \"y\": ";
        writeJsonArray( out, ys );
        out << " }" << ( s + 1 < kNSnaps ? "," : "" ) << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    const std::vector< std::pair< std::string, std::string > > rows{
        { "P, M, D",
          std::to_string( P ) + ", " + std::to_string( M ) + ", " + std::to_string( D ) },
        { "snapshot step", std::to_string( static_cast< int >( snap_step ) ) + " days" },
        { "t_final", std::to_string( static_cast< int >( tFinal_days ) ) + " days" },
        { "n snapshots", std::to_string( kNSnaps ) },
        { "elapsed", std::to_string( ms / 1e3 ) + " s" },
        { "output", "wsb_taylor.json" } };
    printBanner( "WSB taylor (single Taylor flow polynomial)", rows );
    return 0;
}
