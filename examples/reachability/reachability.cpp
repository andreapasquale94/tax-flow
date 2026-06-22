// =============================================================================
// examples/reachability/reachability.cpp
//
// Low-thrust reachable set of a circular heliocentric orbit, thrusting at
// constant (magnitude, direction-from-velocity) over one revolution.
//
// One ADS propagation per snapshot time (every 10 days) over the 2-D
// control box (m, theta); each leaf is the image of a control sub-box.
// The reachable boundary is drawn as non-filled outlines (plot.py).
//
// Run:    ./reachability
// Writes: reachability.json   (plot with examples/reachability/plot.py)
// =============================================================================

#include <string>
#include <tax/ads.hpp>
#include <tax/ode.hpp>
#include <vector>

#include "common.hpp"

int main()
{
    using namespace example;
    using namespace example::reachability;
    using namespace tax::ode::methods;

    constexpr int P = 6;  // DA truncation order
    constexpr int M = 2;  // control DA variables (m, theta)
    constexpr int D = 6;  // state dimension

    constexpr int kNPerEdge = 24;
    const double t_final = kPeriod;

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    cfg.save_steps = true;

    const tax::ads::TruncationCriterion criterion{ /*tol=*/1e-5, /*maxDepth=*/10 };

    // ---- Snapshot times: every 10 days over one orbit ------------------------
    std::vector< double > snap_times;
    for ( int k = 1; k * kSnapStepDays <= 365; ++k )
        snap_times.push_back( k * kSnapStepDays * kTUperDay );

    // ---- Ballistic reference orbit (m = 0) -----------------------------------
    auto ref_sol = tax::ode::propagate( Verner89{}, rhs(), ballisticCenter(), 0.0, t_final, cfg );
    const auto reference = sampleOrbit( ref_sol, {}, D );

    // ---- One ADS propagation per snapshot time -------------------------------
    const auto boundary = unitSquareBoundary( kNPerEdge );
    std::vector< Snapshot > snapshots;
    std::string leaf_counts;
    Stopwatch clock;
    for ( double t : snap_times )
    {
        auto tree = tax::ads::propagate< P >( Verner89{}, criterion, rhs(), controlBox(),
                                              icCenter(), 0.0, t, cfg, adsThreads() );
        Snapshot snap{ t, {} };
        int id = 0;
        for ( int li : tree.done() )
        {
            const auto& leaf = tree.leaf( li );
            snap.leaves.push_back( evalPolygon( leaf.payload, boundary, boundaryToBox, id++,
                                                leaf.depth, /*ix=*/2, /*iy=*/3 ) );
        }
        leaf_counts += ( leaf_counts.empty() ? "" : ", " ) + std::to_string( snap.leaves.size() );
        snapshots.push_back( std::move( snap ) );
    }
    const double elapsed_ms = clock.ms();

    // ---- Output --------------------------------------------------------------
    writeRunJson( "reachability.json", "ads",
                  { { "P", std::to_string( P ) },
                    { "M", std::to_string( M ) },
                    { "D", std::to_string( D ) },
                    { "t_final", jsonNumber( t_final ) },
                    { "a_max", jsonNumber( kAmax ) },
                    { "snap_step_days", std::to_string( kSnapStepDays ) },
                    { "criterion", "\"truncation\"" },
                    { "tol", jsonNumber( criterion.tol ) },
                    { "max_depth", std::to_string( criterion.maxDepth ) } },
                  reference, snapshots, elapsed_ms );

    printBanner( "reachability — low-thrust reachable set (constant thrust over one orbit)",
                 { { "P, M, D", std::to_string( P ) + ", " + std::to_string( M ) + ", " +
                                    std::to_string( D ) },
                   { "a_max", std::to_string( kAmax ) },
                   { "snapshots", std::to_string( snapshots.size() ) + " (every " +
                                      std::to_string( kSnapStepDays ) + " days)" },
                   { "leaves per snap", leaf_counts },
                   { "elapsed", std::to_string( elapsed_ms ) + " ms" },
                   { "output", "reachability.json" } } );
    return 0;
}
