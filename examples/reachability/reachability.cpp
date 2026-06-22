// =============================================================================
// examples/reachability/reachability.cpp
//
// Low-thrust reachable set of a circular heliocentric orbit, thrusting at
// constant (magnitude, direction-from-velocity) over one revolution.
//
// One ADS propagation per snapshot time (every 10 days) over the 2-D
// control box (m, theta); each snapshot draws ONE envelope (outer boundary
// of the reachable set); the underlying ADS leaf count is reported in the
// banner. The envelope outline is drawn by plot.py.
//
// Run:    ./reachability [spacecraft|cubesat]
// Writes: reachability.json            (spacecraft preset, default)
//         reachability_cubesat.json    (cubesat preset)
// Plot with: python3 examples/reachability/plot.py reachability.json \
//                    reachability_cubesat.json --out reachability_compare.png
// =============================================================================

#include <cmath>
#include <string>
#include <tax/ads.hpp>
#include <tax/ode.hpp>
#include <vector>

#include "common.hpp"

namespace
{

// Outer envelope of the reachable set at one snapshot: trace the full
// control-box perimeter and evaluate each point through the ADS leaf that
// contains it, so accuracy is preserved across ADS splits. Returns one
// closed polygon = the reachable-set boundary (position components x=2, y=3).
template < class Tree >
example::Polygon envelopePolygon( const Tree& tree, const tax::ads::Box< double, 2 >& full_box,
                                  const std::vector< std::array< double, 2 > >& boundary )
{
    example::Polygon p;
    p.x.reserve( boundary.size() );
    p.y.reserve( boundary.size() );
    for ( const auto& ab : boundary )
    {
        const double m = full_box.center( 0 ) + full_box.halfWidth( 0 ) * ab[0];
        const double th = full_box.center( 1 ) + full_box.halfWidth( 1 ) * ab[1];
        for ( int li : tree.done() )
        {
            const auto& leaf = tree.leaf( li );
            const double dm = m - leaf.box.center( 0 );
            const double dt = th - leaf.box.center( 1 );
            if ( std::abs( dm ) <= leaf.box.halfWidth( 0 ) + 1e-9 &&
                 std::abs( dt ) <= leaf.box.halfWidth( 1 ) + 1e-9 )
            {
                const std::array< double, 2 > loc{ dm / leaf.box.halfWidth( 0 ),
                                                   dt / leaf.box.halfWidth( 1 ) };
                p.x.push_back( leaf.payload( 2 ).eval( loc ) );
                p.y.push_back( leaf.payload( 3 ).eval( loc ) );
                break;
            }
        }
    }
    return p;
}

}  // namespace

int main( int argc, char** argv )
{
    using namespace example;
    using namespace example::reachability;
    using namespace tax::ode::methods;

    // ---- Select preset from command line -------------------------------------
    const Preset& preset = [&]() -> const Preset& {
        if ( argc > 1 )
        {
            const std::string arg( argv[1] );
            if ( arg == "cubesat" ) return kCubeSat;
            if ( arg == "spacecraft" ) return kSpacecraft;
        }
        return kSpacecraft;
    }();

    const double a_max = aMax( preset );

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
    const auto full_box = controlBox( a_max );
    const auto boundary = unitSquareBoundary( kNPerEdge );
    std::vector< Snapshot > snapshots;
    std::string leaf_counts;
    Stopwatch clock;
    for ( double t : snap_times )
    {
        auto tree = tax::ads::propagate< P >( Verner89{}, criterion, rhs(), full_box,
                                              icCenter( a_max ), 0.0, t, cfg, adsThreads() );
        int n_leaves = 0;
        for ( int li : tree.done() )
        {
            (void)li;
            ++n_leaves;
        }
        Snapshot snap{ t, {} };
        snap.leaves.push_back( envelopePolygon( tree, full_box, boundary ) );
        leaf_counts += ( leaf_counts.empty() ? "" : ", " ) + std::to_string( n_leaves );
        snapshots.push_back( std::move( snap ) );
    }
    const double elapsed_ms = clock.ms();

    // ---- Output --------------------------------------------------------------
    writeRunJson( preset.outfile, "ads",
                  { { "P", std::to_string( P ) },
                    { "M", std::to_string( M ) },
                    { "D", std::to_string( D ) },
                    { "t_final", jsonNumber( t_final ) },
                    { "case", std::string( "\"" ) + preset.name + "\"" },
                    { "thrust_mN", jsonNumber( preset.thrustN * 1000 ) },
                    { "mass_kg", jsonNumber( preset.massKg ) },
                    { "a_max", jsonNumber( a_max ) },
                    { "snap_step_days", std::to_string( kSnapStepDays ) },
                    { "criterion", "\"truncation\"" },
                    { "tol", jsonNumber( criterion.tol ) },
                    { "max_depth", std::to_string( criterion.maxDepth ) } },
                  reference, snapshots, elapsed_ms );

    printBanner( "reachability — low-thrust reachable set (constant thrust over one orbit)",
                 { { "case", preset.name },
                   { "P, M, D", std::to_string( P ) + ", " + std::to_string( M ) + ", " +
                                    std::to_string( D ) },
                   { "a_max", std::to_string( a_max ) },
                   { "snapshots", std::to_string( snapshots.size() ) + " (every " +
                                      std::to_string( kSnapStepDays ) + " days)" },
                   { "leaves per snap", leaf_counts },
                   { "elapsed", std::to_string( elapsed_ms ) + " ms" },
                   { "output", preset.outfile } } );
    return 0;
}
