// =============================================================================
// examples/wsb/ads.cpp
//
// Per-snapshot ADS run on the WSB IC box. For each of the 5-day snapshot
// times we run a fresh ADS propagation from t = 0 to t_snap and dump
// every done leaf's (x, y) boundary image. Truncation criterion.
//
// Run:    ./wsb_ads
// Writes: wsb_ads.json
// =============================================================================

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <tax/ads.hpp>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <thread>

#include "common.hpp"

int main()
{
    using namespace example::wsb;
    using namespace tax::ode::methods;

    constexpr int P = 6;
    constexpr int M = 4;
    constexpr int D = 4;

    constexpr double tFinal_days = kTArrivalDays;  // Moon-orbit interception (76 days)
    constexpr double snap_step = 5.0;
    const int kNSnaps = static_cast< int >( tFinal_days / snap_step ) + 1;
    constexpr int kNPerEdge = 24;

    auto ic_box = icBox();

    const int kThreads = [] {
        if ( const char* e = std::getenv( "TAX_ADS_THREADS" ) )
        {
            const int n = std::atoi( e );
            if ( n > 0 ) return n;
        }
        const unsigned hc = std::thread::hardware_concurrency();
        return hc > 0 ? static_cast< int >( hc ) : 1;
    }();

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-11;
    cfg.max_steps = 50000000;
    cfg.max_rejects_per_step = 5000;
    cfg.initial_step = 1.0e-9;
    cfg.min_step = 1.0e-14;

    const tax::ads::TruncationCriterion criterion{ /*tol=*/1e-4, /*maxDepth=*/6 };

    cfg.save_steps = true;

    auto ref_sol =
        tax::ode::propagate( Verner89{}, rhs(), icCenter(), 0.0, tFinal_days / kTimeU_days, cfg );

    std::vector< double > times_canon( kNSnaps );
    for ( int i = 0; i < kNSnaps; ++i ) times_canon[i] = ( i * snap_step ) / kTimeU_days;

    const auto boundary = unitSquareBoundary( kNPerEdge );

    std::ofstream out( "wsb_ads.json" );
    out << std::setprecision( 14 );
    out << "{\n";
    out << "  \"method\": \"ads\",\n";
    out << "  \"problem\": \"sun_earth_cr3bp_wsb\",\n";
    out << "  \"criterion\": { \"type\": \"truncation\", \"tol\": " << criterion.tol
        << ", \"maxDepth\": " << criterion.maxDepth << " },\n";
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

    std::cout << "[wsb_ads] running " << kNSnaps << " per-snapshot ADS propagations..."
              << std::flush;

    out << "  \"polygons\": [\n";
    std::vector< double > xs( boundary.size() ), ys( boundary.size() );
    std::vector< int > leaves_per_snap;
    double total_ms = 0.0;
    for ( int s = 0; s < kNSnaps; ++s )
    {
        const double t_snap = times_canon[s];
        out << "    { \"t_days\": " << ( s * snap_step ) << ", \"leaves\": [";

        if ( t_snap <= 0.0 )
        {
            for ( std::size_t v = 0; v < boundary.size(); ++v )
            {
                const tax::la::VecNT< M, double > d{ boundary[v][0], 0.0, 0.0, boundary[v][1] };
                const auto pt = ic_box.denormalize( d );
                xs[v] = pt( 0 );
                ys[v] = pt( 1 );
            }
            out << "\n      { \"id\": 0, \"depth\": 0, \"x\": ";
            writeJsonArray( out, xs );
            out << ", \"y\": ";
            writeJsonArray( out, ys );
            out << " }\n    ";
            leaves_per_snap.push_back( 1 );
        } else
        {
            const auto t_a = std::chrono::high_resolution_clock::now();
            auto tree = tax::ads::propagate< P >( Feagin12{}, criterion, rhs(), ic_box, icCenter(),
                                                  0.0, t_snap, cfg, kThreads ).tree();
            const auto t_b = std::chrono::high_resolution_clock::now();
            total_ms += std::chrono::duration< double, std::milli >( t_b - t_a ).count();

            bool first = true;
            int rank = 0;
            for ( int li : tree.done() )
            {
                const auto& leaf = tree.leaf( li );
                const int id = rank++;
                for ( std::size_t v = 0; v < boundary.size(); ++v )
                {
                    const auto d = boundaryToBox( boundary[v][0], boundary[v][1] );
                    xs[v] = leaf.payload( 0 ).eval( d );
                    ys[v] = leaf.payload( 1 ).eval( d );
                }
                if ( !first ) out << ",";
                first = false;
                out << "\n      { \"id\": " << id << ", \"depth\": " << leaf.depth << ", \"x\": ";
                writeJsonArray( out, xs );
                out << ", \"y\": ";
                writeJsonArray( out, ys );
                out << " }";
            }
            out << "\n    ";
            leaves_per_snap.push_back( static_cast< int >( tree.done().size() ) );
        }

        out << "] }" << ( s + 1 < kNSnaps ? "," : "" ) << "\n";
    }
    out << "  ],\n";
    out << "  \"timing\": { \"elapsed_ms\": " << total_ms << " }\n";
    out << "}\n";

    std::cout << "\r" << std::string( 60, ' ' ) << "\r";

    std::string leaves_str;
    for ( std::size_t i = 0; i < leaves_per_snap.size(); ++i )
    {
        if ( i ) leaves_str += ", ";
        leaves_str += std::to_string( leaves_per_snap[i] );
    }

    const std::vector< std::pair< std::string, std::string > > rows{
        { "P, M, D",
          std::to_string( P ) + ", " + std::to_string( M ) + ", " + std::to_string( D ) },
        { "criterion", "truncation (tol=1e-4, depth<=6)" },
        { "snapshot step", std::to_string( static_cast< int >( snap_step ) ) + " days" },
        { "t_final", std::to_string( static_cast< int >( tFinal_days ) ) + " days" },
        { "n snapshots", std::to_string( kNSnaps ) },
        { "leaves per snap", leaves_str },
        { "elapsed", std::to_string( total_ms / 1e3 ) + " s" },
        { "output", "wsb_ads.json" } };
    printBanner( "WSB ads (piecewise polynomial flow)", rows );
    return 0;
}
