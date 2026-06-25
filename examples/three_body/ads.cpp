// =============================================================================
// examples/three_body/ads.cpp
//
// Step 2 — Automatic Domain Splitting on the L1 neighbourhood IC box
// (Earth-Moon planar CR3BP).
//
// A single ADS propagation records the partition at each snapshot time.
// The collection grows from a single leaf at small times to many as the
// unstable manifold stretches the IC box into a long thin streak that one
// polynomial can no longer represent.
//
// Run:    ./three_body_ads
// Writes: cr3bp_ads.json   (plot with examples/plot/plot_three_body.py)
// =============================================================================

#include <tax/ads.hpp>
#include <tax/ode.hpp>

#include "common.hpp"

int main()
{
    using namespace example;
    using namespace example::three_body;
    using namespace tax::ode::methods;

    constexpr int P = 4;  // DA truncation order
    constexpr int M = 4;  // number of DA variables
    constexpr int D = 4;  // state dimension

    constexpr int kNSnaps = 13;  // every 0.25 time units
    constexpr int kNPerEdge = 24;
    constexpr double t_final = 3.0;

    const auto ic_box = icBox();

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;
    cfg.max_steps = 100000;

    const tax::ads::TruncationCriterion criterion{ /*tol=*/1e-4, /*maxDepth=*/8 };

    cfg.save_steps = true;

    // ---- Scalar centerpoint orbit (plot underlay) ----------------------------
    auto ref_sol = tax::ode::propagate( Verner89{}, rhs(), icCenter(), 0.0, t_final, cfg );
    const auto reference = sampleOrbit( ref_sol, example::linspace( 0.0, t_final, 200 ), D );

    // ---- Single ADS propagation with a snapshot grid -------------------------
    const auto boundary = unitSquareBoundary( kNPerEdge );
    std::vector< Snapshot > snapshots;
    std::string leaf_counts;
    Stopwatch clock;

    // One propagation; the snapshot grid records the partition at each time.
    const auto grid_times = example::linspace( 0.0, t_final, kNSnaps );
    auto sol = tax::ads::propagate< P >( Verner89{}, criterion, rhs(), ic_box, icCenter(), 0.0,
                                         t_final, grid_times, cfg, adsThreads() );

    for ( const auto& part : sol.snapshots() )
    {
        Snapshot snap{ part.time(), {} };
        for ( const auto& leaf : part )
            snap.leaves.push_back(
                evalPolygon( leaf.flowMap, boundary, boundaryToBox, leaf.id, leaf.depth ) );
        leaf_counts += ( leaf_counts.empty() ? "" : ", " ) + std::to_string( snap.leaves.size() );
        snapshots.push_back( std::move( snap ) );
    }
    const double elapsed_ms = clock.ms();

    // ---- Output ---------------------------------------------------------------
    writeRunJson( "cr3bp_ads.json", "ads",
                  { { "P", std::to_string( P ) },
                    { "M", std::to_string( M ) },
                    { "D", std::to_string( D ) },
                    { "t_final", jsonNumber( t_final ) },
                    { "mu", jsonNumber( kCR3BPMu ) },
                    { "x_L1", jsonNumber( kCR3BPL1 ) },
                    { "lambda_unstable", jsonNumber( linL1().lambda_unstable ) },
                    { "criterion", "\"truncation\"" },
                    { "tol", jsonNumber( criterion.tol ) },
                    { "max_depth", std::to_string( criterion.maxDepth ) },
                    { "ic_center", jsonArray( ic_box.center ) },
                    { "ic_half_width", jsonArray( ic_box.halfWidth ) } },
                  reference, snapshots, elapsed_ms );

    printBanner( "three_body/ads — piecewise polynomial flow (CR3BP)",
                 { { "P, M", std::to_string( P ) + ", " + std::to_string( M ) },
                   { "criterion", "truncation, tol=1e-4, depth<=8" },
                   { "leaves per snap", leaf_counts },
                   { "elapsed", std::to_string( elapsed_ms ) + " ms" },
                   { "output", "cr3bp_ads.json" } } );
    return 0;
}
