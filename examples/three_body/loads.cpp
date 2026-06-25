// =============================================================================
// examples/three_body/loads.cpp
//
// Step 3 — Low-Order Automatic Domain Splitting (LOADS) on the L1
// neighbourhood IC box. Same structure as ads.cpp but with the
// nonlinearity-index criterion (Losacco/Fossà/Armellin) instead of the
// truncation criterion — note the much lower order P.
//
// Run:    ./three_body_loads
// Writes: cr3bp_loads.json   (plot with examples/plot/plot_three_body.py)
// =============================================================================

#include <tax/ads.hpp>
#include <tax/ode.hpp>

#include "common.hpp"

int main()
{
    using namespace example;
    using namespace example::three_body;
    using namespace tax::ode::methods;

    constexpr int P = 2;  // NLI needs only the Jacobian variation: order 2
    constexpr int M = 4;
    constexpr int D = 4;

    constexpr int kNSnaps = 13;
    constexpr int kNPerEdge = 24;
    constexpr double t_final = 3.0;

    const auto ic_box = icBox();

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;
    cfg.max_steps = 100000;

    const tax::ads::NliCriterion criterion{ /*tol=*/0.3, /*maxDepth=*/12 };

    cfg.save_steps = true;

    // ---- Scalar centerpoint orbit (plot underlay) ----------------------------
    auto ref_sol = tax::ode::propagate( Verner89{}, rhs(), icCenter(), 0.0, t_final, cfg );
    const auto reference = sampleOrbit( ref_sol, example::linspace( 0.0, t_final, 200 ), D );

    // ---- One LOADS propagation per snapshot time ------------------------------
    const auto boundary = unitSquareBoundary( kNPerEdge );
    std::vector< Snapshot > snapshots;
    std::string leaf_counts;
    Stopwatch clock;
    for ( double t : example::linspace( 0.0, t_final, kNSnaps ) )
    {
        Snapshot snap{ t, {} };
        if ( t <= 0.0 )
        {
            snap.leaves.push_back( boxPolygon( ic_box, boundary, boundaryToBox ) );
        } else
        {
            auto tree = tax::ads::propagate< P >( Verner89{}, criterion, rhs(), ic_box, icCenter(),
                                                  0.0, t, cfg, adsThreads() ).tree();
            int id = 0;
            for ( int li : tree.done() )
            {
                const auto& leaf = tree.leaf( li );
                snap.leaves.push_back(
                    evalPolygon( leaf.payload, boundary, boundaryToBox, id++, leaf.depth ) );
            }
        }
        leaf_counts += ( leaf_counts.empty() ? "" : ", " ) + std::to_string( snap.leaves.size() );
        snapshots.push_back( std::move( snap ) );
    }
    const double elapsed_ms = clock.ms();

    // ---- Output ---------------------------------------------------------------
    writeRunJson( "cr3bp_loads.json", "loads",
                  { { "P", std::to_string( P ) },
                    { "M", std::to_string( M ) },
                    { "D", std::to_string( D ) },
                    { "t_final", jsonNumber( t_final ) },
                    { "mu", jsonNumber( kCR3BPMu ) },
                    { "x_L1", jsonNumber( kCR3BPL1 ) },
                    { "lambda_unstable", jsonNumber( linL1().lambda_unstable ) },
                    { "criterion", "\"nli\"" },
                    { "tol", jsonNumber( criterion.tol ) },
                    { "max_depth", std::to_string( criterion.maxDepth ) },
                    { "ic_center", jsonArray( ic_box.center ) },
                    { "ic_half_width", jsonArray( ic_box.halfWidth ) } },
                  reference, snapshots, elapsed_ms );

    printBanner( "three_body/loads — piecewise polynomial flow (NLI criterion)",
                 { { "P, M", std::to_string( P ) + ", " + std::to_string( M ) },
                   { "criterion", "nli, tol=0.3, depth<=12" },
                   { "leaves per snap", leaf_counts },
                   { "elapsed", std::to_string( elapsed_ms ) + " ms" },
                   { "output", "cr3bp_loads.json" } } );
    return 0;
}
