// =============================================================================
// examples/two_body/taylor.cpp
//
// Step 1 — One multivariate Taylor flow polynomial over the IC box.
//
// A DA-valued state (an Eigen vector of TaylorExpansions, seeded as
// ic_center + halfWidth * xi) is integrated once. The result is a
// polynomial flow map: at each snapshot time we retrieve the accepted-step
// state whose time is closest to the snapshot time.
//
// At 9 snapshot times along one orbit we evaluate the (x, y) components
// on the boundary of the IC box's (y, vy)-face. The closed polygons show
// the box deforming — gradually banana-shaped as the orbit wraps around.
//
// Run:    ./two_body_taylor
// Writes: taylor.json   (plot with examples/plot/plot_two_body.py)
// =============================================================================

#include <algorithm>
#include <tax/ads/da_state.hpp>
#include <tax/ode.hpp>

#include "common.hpp"

int main()
{
    using namespace example;
    using namespace example::two_body;
    using namespace tax::ode::methods;

    constexpr int P = 6;  // DA truncation order
    constexpr int M = 4;  // number of DA variables (state dimension here)
    constexpr int D = 4;  // state dimension

    constexpr int kNSnaps = 9;     // 0, pi/4, ..., 2 pi (every 45 deg)
    constexpr int kNPerEdge = 24;  // boundary samples per square edge
    const double t_final = kPeriod;

    const auto ic_box = icBox();

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    cfg.save_steps = true;

    // ---- One DA propagation over the whole interval -------------------------
    auto x0_da = tax::domain::create< P, M >( ic_box, icCenter() );
    Stopwatch clock;
    auto sol = tax::ode::propagate( Verner89{}, rhs(), x0_da, 0.0, t_final, cfg );
    const double elapsed_ms = clock.ms();

    // ---- Scalar centerpoint orbit (plot underlay) ----------------------------
    auto ref_sol = tax::ode::propagate( Taylor< 16 >{}, rhs(), icCenter(), 0.0, t_final, cfg );
    const auto reference = sampleOrbit( ref_sol, example::linspace( 0.0, t_final, 200 ), D );

    // Helper: find stored step with t closest to t_query.
    auto stateAt = [&sol]( double t_query ) -> decltype( sol.x.front() ) {
        auto it = std::lower_bound( sol.t.begin(), sol.t.end(), t_query );
        if ( it == sol.t.end() ) return sol.x.back();
        if ( it == sol.t.begin() ) return sol.x.front();
        auto prev = std::prev( it );
        const std::size_t idx = ( t_query - *prev < *it - t_query )
                                    ? static_cast< std::size_t >( prev - sol.t.begin() )
                                    : static_cast< std::size_t >( it - sol.t.begin() );
        return sol.x[idx];
    };

    // ---- Evaluate the flow polynomial on the box boundary per snapshot ------
    const auto boundary = unitSquareBoundary( kNPerEdge );
    std::vector< Snapshot > snapshots;
    for ( double t : example::linspace( 0.0, t_final, kNSnaps ) )
        snapshots.push_back( { t, { evalPolygon( stateAt( t ), boundary, boundaryToBox ) } } );

    // ---- Output ---------------------------------------------------------------
    writeRunJson( "taylor.json", "taylor",
                  { { "P", std::to_string( P ) },
                    { "M", std::to_string( M ) },
                    { "D", std::to_string( D ) },
                    { "t_final", jsonNumber( t_final ) },
                    { "ic_center", jsonArray( ic_box.center ) },
                    { "ic_half_width", jsonArray( ic_box.halfWidth ) } },
                  reference, snapshots, elapsed_ms );

    printBanner( "two_body/taylor — single Taylor flow polynomial",
                 { { "P, M", std::to_string( P ) + ", " + std::to_string( M ) },
                   { "snapshots", std::to_string( kNSnaps ) },
                   { "elapsed", std::to_string( elapsed_ms ) + " ms" },
                   { "output", "taylor.json" } } );
    return 0;
}
