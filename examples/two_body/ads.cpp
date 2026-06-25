// =============================================================================
// examples/two_body/ads.cpp
//
// Step 2 — Automatic Domain Splitting on the IC box.
//
// A single Taylor flow polynomial (taylor.cpp) loses accuracy as the IC
// box deforms. ADS (Wittig 2015) subdivides the box whenever the
// polynomial's top-degree mass exceeds a tolerance; each leaf of the
// resulting tree carries its own flow polynomial on a sub-domain.
//
// For the box-evolution figure we run ADS independently to each of 9
// snapshot times. The first snapshot is the IC box itself; later
// snapshots show progressively finer partitions as nonlinearity grows.
//
// Run:    ./two_body_ads
// Writes: ads.json   (plot with examples/plot/plot_two_body.py)
// =============================================================================

#include <tax/ads.hpp>
#include <tax/ode.hpp>

#include "common.hpp"

int main()
{
    using namespace example;
    using namespace example::two_body;
    using namespace tax::ode::methods;

    constexpr int P = 8;  // DA truncation order
    constexpr int M = 4;  // number of DA variables
    constexpr int D = 4;  // state dimension

    constexpr int kNSnaps = 9;
    constexpr int kNPerEdge = 24;
    const double t_final = kPeriod;

    const auto ic_box = icBox();

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    // Wittig's truncation criterion: split when the summed |coefficient|
    // mass at total degree P exceeds tol.
    const tax::ads::TruncationCriterion criterion{ /*tol=*/1e-6, /*maxDepth=*/8 };

    cfg.save_steps = true;

    // ---- Scalar centerpoint orbit (plot underlay) ----------------------------
    auto ref_sol = tax::ode::propagate( Taylor< 16 >{}, rhs(), icCenter(), 0.0, t_final, cfg );
    const auto reference = sampleOrbit( ref_sol, example::linspace( 0.0, t_final, 200 ), D );

    // ---- One ADS propagation per snapshot time -------------------------------
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
    writeRunJson( "ads.json", "ads",
                  { { "P", std::to_string( P ) },
                    { "M", std::to_string( M ) },
                    { "D", std::to_string( D ) },
                    { "t_final", jsonNumber( t_final ) },
                    { "criterion", "\"truncation\"" },
                    { "tol", jsonNumber( criterion.tol ) },
                    { "max_depth", std::to_string( criterion.maxDepth ) },
                    { "ic_center", jsonArray( ic_box.center ) },
                    { "ic_half_width", jsonArray( ic_box.halfWidth ) } },
                  reference, snapshots, elapsed_ms );

    printBanner( "two_body/ads — piecewise polynomial flow (truncation criterion)",
                 { { "P, M", std::to_string( P ) + ", " + std::to_string( M ) },
                   { "criterion", "truncation, tol=1e-6, depth<=8" },
                   { "leaves per snap", leaf_counts },
                   { "elapsed", std::to_string( elapsed_ms ) + " ms" },
                   { "output", "ads.json" } } );
    return 0;
}
