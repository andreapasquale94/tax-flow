// =============================================================================
// examples/three_body/ads.cpp
//
// Step 2 — Automatic Domain Splitting on the L1 neighbourhood IC box
// (Earth-Moon planar CR3BP).
//
// For each snapshot time we run a fresh ADS propagation from t = 0 with
// the truncation criterion and dump every done leaf's (x, y) boundary
// image. The collection grows from a single leaf at small times to many
// as the unstable manifold stretches the IC box into a long thin streak
// that one polynomial can no longer represent.
//
// Run:    ./three_body_ads
// Writes: cr3bp_ads.json   (plot with examples/plot/plot_three_body.py)
// =============================================================================

#include <tax/ads.hpp>
#include <tax/ode.hpp>
#include <tax/ode/io.hpp>

#include "common.hpp"

int main()
{
    using namespace example;
    using namespace example::three_body;
    using namespace tax::ode::methods;

    constexpr int P = 4;  // DA truncation order
    constexpr int M = 4;  // number of DA variables
    constexpr int D = 4;  // state dimension

    constexpr int    kNSnaps   = 13;   // every 0.25 time units
    constexpr int    kNPerEdge = 24;
    constexpr double t_final   = 3.0;

    const auto ic_box = icBox();

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;
    cfg.max_steps           = 100000;

    const tax::ads::TruncationCriterion criterion{ /*tol=*/1e-4, /*maxDepth=*/8 };

    // ---- Scalar centerpoint orbit (plot underlay) ----------------------------
    auto ref_sol = tax::ode::propagate< /*Dense=*/true >(
        Verner89{}, rhs(), icCenter(), 0.0, t_final, cfg );
    const auto reference = sampleOrbit( ref_sol, tax::ode::linspace( 0.0, t_final, 200 ), D );

    // ---- One ADS propagation per snapshot time -------------------------------
    const auto boundary = unitSquareBoundary( kNPerEdge );
    std::vector< Snapshot > snapshots;
    std::string leaf_counts;
    Stopwatch   clock;
    for ( double t : tax::ode::linspace( 0.0, t_final, kNSnaps ) )
    {
        Snapshot snap{ t, {} };
        if ( t <= 0.0 )
        {
            snap.leaves.push_back( boxPolygon( ic_box, boundary, boundaryToBox ) );
        }
        else
        {
            auto tree = tax::ads::propagate< P >( Verner89{}, criterion, rhs(), ic_box,
                                                  icCenter(), 0.0, t, cfg, adsThreads() );
            int id = 0;
            for ( int li : tree.done() )
            {
                const auto& leaf = tree.leaf( li );
                snap.leaves.push_back(
                    evalPolygon( leaf.payload, boundary, boundaryToBox, id++, leaf.depth ) );
            }
        }
        leaf_counts += ( leaf_counts.empty() ? "" : ", " )
                       + std::to_string( snap.leaves.size() );
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
