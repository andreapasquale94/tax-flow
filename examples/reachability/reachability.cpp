// =============================================================================
// examples/reachability/reachability.cpp
//
// Low-thrust reachable set of a circular heliocentric orbit, thrusting at
// constant (magnitude, direction-from-velocity) over one revolution.
//
// A single ADS propagation over the 2-D control box (m, theta); snapshots()
// records the partition every 10 days and each snapshot draws ONE envelope
// (outer boundary of the reachable set); the underlying ADS leaf count is
// reported in the banner. The envelope outline is drawn by plot.py.
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

// Outer envelope of the reachable set at one snapshot: the MAX-THRUST edge
// (m = a_max) swept over the full direction range theta in [0, 2pi]. Each
// sample is evaluated through the ADS leaf that contains it, so accuracy is
// preserved across splits. Returns one closed polygon = the reachable-set
// boundary (position components x = 2, y = 3). Tracing only this edge avoids
// the radial spoke that a full control-box-perimeter trace would draw (the
// m = 0 edge collapses to a point and the theta seam doubles back).
template < class Partition >
example::Polygon envelopePolygon( const Partition& part,
                                  const tax::domain::Box< double, 2 >& full_box, int n_theta )
{
    example::Polygon p;
    p.x.reserve( static_cast< std::size_t >( n_theta ) + 1 );
    p.y.reserve( static_cast< std::size_t >( n_theta ) + 1 );
    const double m = full_box.center( 0 ) + full_box.halfWidth( 0 );  // xi_m = +1 => m = a_max
    for ( int i = 0; i <= n_theta; ++i )
    {
        const double xitheta =
            -1.0 + 2.0 * static_cast< double >( i ) / static_cast< double >( n_theta );
        const double th = full_box.center( 1 ) + full_box.halfWidth( 1 ) * xitheta;
        // Partition::evaluate locates the owning leaf, recovers its exact
        // factor coordinates and evaluates the flow map in one call.
        const auto state = part.evaluate( tax::la::VecNT< 2, double >{ m, th } );
        if ( !state.has_value() ) continue;
        p.x.push_back( ( *state )( 2 ) );
        p.y.push_back( ( *state )( 3 ) );
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

    constexpr int kNTheta = 360;  // samples along the max-thrust ring (one per degree)
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

    // ---- Single ADS propagation with a snapshot grid -------------------------
    const auto full_box = controlBox( a_max );
    std::vector< Snapshot > snapshots;
    std::string leaf_counts;
    Stopwatch clock;
    // One propagation to the last snapshot time; snapshots() records the partition
    // at each grid time. Skip the t0 bracket: the reachable set is rendered from the
    // first snapshot onward, as the original per-snapshot version did.
    auto sol = tax::ads::propagate< P >( Verner89{}, criterion, rhs(), full_box, icCenter( a_max ),
                                         0.0, snap_times.back(), snap_times, cfg, adsThreads() );
    for ( const auto& part : sol.snapshots() )
    {
        if ( part.time() <= 0.0 ) continue;
        Snapshot snap{ part.time(), {} };
        snap.leaves.push_back( envelopePolygon( part, full_box, kNTheta ) );
        leaf_counts += ( leaf_counts.empty() ? "" : ", " ) + std::to_string( part.size() );
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
