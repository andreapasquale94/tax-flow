// =============================================================================
// examples/three_body/taylor.cpp
//
// Step 1 — Single multivariate Taylor flow polynomial over the L1
// neighbourhood IC box (Earth-Moon planar CR3BP).
//
// One DA-valued state built from icBox() is propagated with Dense=true.
// At evenly spaced snapshot times we evaluate the (x, y) components of
// the flow polynomial along the IC box boundary; the closed polygons
// show the box being sheared along the unstable manifold direction.
//
// Run:    ./three_body_taylor
// Writes: cr3bp_taylor.json   (plot with examples/plot/plot_three_body.py)
// =============================================================================

#include <tax/ads/da_state.hpp>
#include <tax/ode.hpp>
#include <tax/ode/io.hpp>

#include "common.hpp"

int main()
{
    using namespace example;
    using namespace example::three_body;
    using namespace tax::ode::methods;

    constexpr int P = 6;  // DA truncation order
    constexpr int M = 4;  // number of DA variables
    constexpr int D = 4;  // state dimension

    constexpr int    kNSnaps   = 13;   // every 0.25 time units
    constexpr int    kNPerEdge = 24;
    constexpr double t_final   = 3.0;  // ~1.1 Lyapunov times

    const auto ic_box = icBox();

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;
    cfg.max_steps           = 100000;

    // ---- One dense DA propagation over the whole interval -------------------
    auto      x0_da = tax::ads::create< P, M >( ic_box, icCenter() );
    Stopwatch clock;
    auto      sol = tax::ode::propagate< /*Dense=*/true >(
        Verner89{}, rhs(), x0_da, 0.0, t_final, cfg );
    const double elapsed_ms = clock.ms();

    // ---- Scalar centerpoint orbit (plot underlay) ----------------------------
    auto ref_sol = tax::ode::propagate< /*Dense=*/true >(
        Verner89{}, rhs(), icCenter(), 0.0, t_final, cfg );
    const auto reference = sampleOrbit( ref_sol, tax::ode::linspace( 0.0, t_final, 200 ), D );

    // ---- Evaluate the flow polynomial on the box boundary per snapshot ------
    const auto boundary = unitSquareBoundary( kNPerEdge );
    std::vector< Snapshot > snapshots;
    for ( double t : tax::ode::linspace( 0.0, t_final, kNSnaps ) )
        snapshots.push_back( { t, { evalPolygon( sol( t ), boundary, boundaryToBox ) } } );

    // ---- Output ---------------------------------------------------------------
    writeRunJson( "cr3bp_taylor.json", "taylor",
                  { { "P", std::to_string( P ) },
                    { "M", std::to_string( M ) },
                    { "D", std::to_string( D ) },
                    { "t_final", jsonNumber( t_final ) },
                    { "mu", jsonNumber( kCR3BPMu ) },
                    { "x_L1", jsonNumber( kCR3BPL1 ) },
                    { "lambda_unstable", jsonNumber( linL1().lambda_unstable ) },
                    { "v_unstable", jsonArray( linL1().v_unstable ) },
                    { "ic_center", jsonArray( ic_box.center ) },
                    { "ic_half_width", jsonArray( ic_box.halfWidth ) } },
                  reference, snapshots, elapsed_ms );

    printBanner( "three_body/taylor — single Taylor flow polynomial (CR3BP)",
                 { { "P, M", std::to_string( P ) + ", " + std::to_string( M ) },
                   { "t_final", std::to_string( t_final ) },
                   { "lambda_u", std::to_string( linL1().lambda_unstable ) },
                   { "snapshots", std::to_string( kNSnaps ) },
                   { "elapsed", std::to_string( elapsed_ms ) + " ms" },
                   { "output", "cr3bp_taylor.json" } } );
    return 0;
}
