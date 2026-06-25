// =============================================================================
// examples/two_body/loads.cpp
//
// Step 3 — Low-Order Automatic Domain Splitting (LOADS).
//
// Identical structure to ads.cpp; only the splitting criterion changes.
// LOADS (Losacco/Fossà/Armellin 2024) splits on the nonlinearity index —
// the ratio between the mass of the degree->=2 Jacobian-variation bound
// and the mass of the linear Jacobian. NLI is more sensitive to
// swirl-type nonlinearities (periapsis passages) than the truncation
// residual is, which is why a lower order P suffices.
//
// Run:    ./two_body_loads
// Writes: loads.json   (plot with examples/plot/plot_two_body.py)
// =============================================================================

#include <tax/ads.hpp>
#include <tax/ode.hpp>

#include "common.hpp"

int main()
{
    using namespace example;
    using namespace example::two_body;
    using namespace tax::ode::methods;

    constexpr int P = 6;  // LOADS works at lower order than truncation ADS
    constexpr int M = 4;
    constexpr int D = 4;

    constexpr int kNSnaps = 9;
    constexpr int kNPerEdge = 24;
    const double t_final = kPeriod;

    const auto ic_box = icBox();

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const tax::ads::NliCriterion criterion{ /*tol=*/1.0, /*maxDepth=*/6 };

    cfg.save_steps = true;

    // ---- Scalar centerpoint orbit (plot underlay) ----------------------------
    auto ref_sol = tax::ode::propagate( Taylor< 16 >{}, rhs(), icCenter(), 0.0, t_final, cfg );
    const auto reference = sampleOrbit( ref_sol, example::linspace( 0.0, t_final, 200 ), D );

    // ---- Single LOADS propagation with a snapshot grid -----------------------
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
    writeRunJson( "loads.json", "loads",
                  { { "P", std::to_string( P ) },
                    { "M", std::to_string( M ) },
                    { "D", std::to_string( D ) },
                    { "t_final", jsonNumber( t_final ) },
                    { "criterion", "\"nli\"" },
                    { "tol", jsonNumber( criterion.tol ) },
                    { "max_depth", std::to_string( criterion.maxDepth ) },
                    { "ic_center", jsonArray( ic_box.center ) },
                    { "ic_half_width", jsonArray( ic_box.halfWidth ) } },
                  reference, snapshots, elapsed_ms );

    printBanner( "two_body/loads — piecewise polynomial flow (NLI criterion)",
                 { { "P, M", std::to_string( P ) + ", " + std::to_string( M ) },
                   { "criterion", "nli, tol=1, depth<=6" },
                   { "leaves per snap", leaf_counts },
                   { "elapsed", std::to_string( elapsed_ms ) + " ms" },
                   { "output", "loads.json" } } );
    return 0;
}
