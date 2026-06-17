// =============================================================================
// examples/two_body/refine.cpp
//
// Step 4 — "Propagate-then-assess" ADS refinement (tax::ads::refine).
//
// Where ads.cpp splits a box mid-flight the instant its flow polynomial
// stops converging, refine() instead carries *every* box all the way to the
// final time and only then judges its quality by bisecting it, propagating
// both halves to the end as well, and comparing the result. Because no box
// ever needs another box's partial state, the whole refinement fans out in
// parallel.
//
// This driver runs the refinement at increasing depth caps k = 0, 1, 2, ...
// Iteration 0 is the single box; each iteration adds sub-boxes until the
// partition converges. For every iteration we record the box images at a
// sweep of snapshot times (for the animation) and the RMS error of the
// piecewise-polynomial prediction against a Monte-Carlo reference cloud (to
// show that more boxes ⇒ better matching).
//
// The sweep is run with two quality indices, both at tol = 1e-6, so they can
// be compared: the dimension-free CoefficientMatchCriterion (which re-identifies
// the parent map on each half and checks it reproduces the independently
// propagated child) drives the animation, and the dimension-general geometric
// VolumeRatioCriterion (parent-vs-children image volume) is run alongside.
//
// The IC box is identical to the taylor.cpp / ads.cpp cases — it varies the
// initial y position (±8e-3) and y velocity (±2e-2) — so the figures here can
// be compared directly with those examples.
//
// Run:    ./two_body_refine
// Writes: refine.json   (animate with examples/two_body/plot_refine.py)
// =============================================================================

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <tax/ads.hpp>
#include <tax/ode.hpp>
#include <tax/ode/io.hpp>
#include <vector>

#include "common.hpp"

namespace
{
using namespace example;
using namespace example::two_body;
using namespace tax::ode::methods;

constexpr int P = 6;  // DA truncation order
constexpr int M = 4;  // DA variables (full state; only y and vy are active)
constexpr int D = 4;  // state dimension (x, y, vx, vy)

using TE = tax::TE< P, M >;
using DAState = tax::la::VecNT< D, TE >;
using BoxT = tax::ads::Box< double, M >;

// The box varies axes 1 (y) and 3 (vy); kIcBoxHalfWidth pins x and vx to 0.
constexpr int kAxisY = 1;
constexpr int kAxisVy = 3;

// Reconstruct a leaf's identity (t0) DA state from its sub-box: each axis
// carries the (possibly shifted) center and (possibly halved) half-width.
// Re-propagating this densely recovers the leaf's flow map at every time,
// not just the final one.
DAState leafInit( const BoxT& box )
{
    DAState s;
    for ( int i = 0; i < D; ++i )
    {
        TE c{};
        c[0] = box.center( i );
        tax::MultiIndex< M > alpha{};
        alpha[static_cast< std::size_t >( i )] = 1;
        c[tax::flatIndex< M >( alpha )] = box.halfWidth( i );
        s( i ) = std::move( c );
    }
    return s;
}

// Shoelace area of a closed polygon.
double polygonArea( const Polygon& p )
{
    double twice = 0.0;
    const std::size_t n = p.x.size();
    for ( std::size_t i = 0; i + 1 < n; ++i ) twice += p.x[i] * p.y[i + 1] - p.x[i + 1] * p.y[i];
    return 0.5 * std::abs( twice );
}

struct McSample
{
    tax::la::VecNT< D, double > ic;  // initial state in the box
    std::vector< double > truth_x;   // (x, y) along the snapshot times
    std::vector< double > truth_y;
};

struct Iteration
{
    int max_depth = 0;
    int n_boxes = 0;
    double area = 0.0;  // total covered (x, y) area at the final time
    double rms = 0.0;   // RMS prediction error vs Monte Carlo at the final time
    std::vector< Snapshot > snapshots;
};
}  // namespace

int main( int argc, char** argv )
{
    constexpr int kNSnaps = 13;  // animation frames per iteration
    constexpr int kNPerEdge = 20;
    constexpr int kMaxIter = 6;
    constexpr int kNMonte = 350;
    const double t_final = kPeriod;

    // Optional CLI: a half-width scale factor and an output filename, so the
    // same program produces both the small run (scale 1, comparable to
    // ads.cpp) and a bigger-box run that fragments into many more leaves.
    const double box_scale = argc > 1 ? std::atof( argv[1] ) : 1.0;
    const std::string outfile = argc > 2 ? argv[2] : "refine.json";

    BoxT box = icBox();  // same IC box as taylor.cpp / ads.cpp (scaled below)
    box.halfWidth *= box_scale;
    const double half_y = box.halfWidth( kAxisY );
    const double half_v = box.halfWidth( kAxisVy );

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const auto snap_times = tax::ode::linspace( 0.0, t_final, kNSnaps );
    const auto boundary = unitSquareBoundary( kNPerEdge );

    // ---- Scalar centerpoint orbit (plot underlay) ----------------------------
    auto ref_sol = tax::ode::propagate< /*Dense=*/true >( Taylor< 16 >{}, rhs(), icCenter(), 0.0,
                                                          t_final, cfg );
    const auto reference = sampleOrbit( ref_sol, tax::ode::linspace( 0.0, t_final, 200 ), D );

    // ---- Monte-Carlo reference cloud -----------------------------------------
    // Uniform samples of the IC box, each propagated densely so we can show
    // the true set at every snapshot time and score each iteration against it.
    std::mt19937 rng( 12345u );
    std::uniform_real_distribution< double > unit( -1.0, 1.0 );
    std::vector< McSample > monte;
    monte.reserve( kNMonte );
    for ( int s = 0; s < kNMonte; ++s )
    {
        tax::la::VecNT< D, double > ic = icCenter();
        ic( kAxisY ) += half_y * unit( rng );
        ic( kAxisVy ) += half_v * unit( rng );
        auto sol =
            tax::ode::propagate< /*Dense=*/true >( Verner89{}, rhs(), ic, 0.0, t_final, cfg );

        McSample m;
        m.ic = ic;
        m.truth_x.reserve( snap_times.size() );
        m.truth_y.reserve( snap_times.size() );
        for ( double t : snap_times )
        {
            const auto x = sol( t );
            m.truth_x.push_back( x( 0 ) );
            m.truth_y.push_back( x( 1 ) );
        }
        monte.push_back( std::move( m ) );
    }

    // ---- Refinement sweep: increasing depth cap -----------------------------
    // Run the same sweep with two quality indices (both at tol = 1e-6) so they
    // can be compared. The sweep stops once the partition stops growing (the
    // box count has converged). `collectSnapshots` also re-propagates each leaf
    // densely to record the box images for the animation — only needed for the
    // primary (coefficient-match) run that drives the GIF.
    const std::size_t last = snap_times.size() - 1;
    auto sweep = [&]( auto makeCrit, bool collectSnapshots ) {
        std::vector< Iteration > out;
        int prev_boxes = -1;
        for ( int k = 0; k <= kMaxIter; ++k )
        {
            auto tree = tax::ads::refine< P >( Verner89{}, makeCrit( k ), rhs(), box, icCenter(),
                                               0.0, t_final, cfg, adsThreads() );

            Iteration it;
            it.max_depth = k;
            if ( collectSnapshots )
            {
                it.snapshots.assign( snap_times.size(), Snapshot{} );
                for ( std::size_t si = 0; si < snap_times.size(); ++si )
                    it.snapshots[si].t = snap_times[si];
            }

            int id = 0;
            for ( int li : tree.done() )
            {
                const auto& leaf = tree.leaf( li );
                if ( collectSnapshots )
                {
                    auto sol = tax::ode::propagate< /*Dense=*/true >(
                        Verner89{}, rhs(), leafInit( leaf.box ), 0.0, t_final, cfg );
                    for ( std::size_t si = 0; si < snap_times.size(); ++si )
                    {
                        auto poly = evalPolygon( sol( snap_times[si] ), boundary, boundaryToBox, id,
                                                 leaf.depth );
                        if ( si + 1 == snap_times.size() ) it.area += polygonArea( poly );
                        it.snapshots[si].leaves.push_back( std::move( poly ) );
                    }
                }
                ++id;
            }
            it.n_boxes = id;

            // RMS error of the piecewise-polynomial prediction at the final time.
            double sq = 0.0;
            int counted = 0;
            for ( const auto& m : monte )
            {
                auto idx = tree.leaf( m.ic );
                if ( !idx.has_value() ) continue;
                const auto& leaf = tree.leaf( *idx );
                std::array< double, M > local{};
                for ( int j = 0; j < M; ++j )
                {
                    const double hw = leaf.box.halfWidth( j );
                    local[static_cast< std::size_t >( j )] =
                        hw > 0.0 ? ( m.ic( j ) - leaf.box.center( j ) ) / hw : 0.0;
                }
                const double dx = leaf.payload( 0 ).eval( local ) - m.truth_x[last];
                const double dy = leaf.payload( 1 ).eval( local ) - m.truth_y[last];
                sq += dx * dx + dy * dy;
                ++counted;
            }
            it.rms = counted > 0 ? std::sqrt( sq / counted ) : 0.0;

            const bool converged = ( id == prev_boxes );
            prev_boxes = id;
            out.push_back( std::move( it ) );
            if ( converged ) break;
        }
        return out;
    };

    Stopwatch clock;
    // Primary run (drives the animation): the dimension-free coefficient match.
    const auto iters = sweep(
        []( int k ) { return tax::ads::CoefficientMatchCriterion{ /*tol=*/1e-6, /*maxDepth=*/k }; },
        /*collectSnapshots=*/true );
    // Comparison run: the dimension-general geometric volume ratio over the two
    // active axes (y, vy).
    const auto vol_iters = sweep(
        []( int k ) {
            return tax::ads::VolumeRatioCriterion{ /*tol=*/1e-6, /*maxDepth=*/k,
                                                   /*axes=*/{ kAxisY, kAxisVy }, /*nQuad=*/8 };
        },
        /*collectSnapshots=*/false );
    const double elapsed_ms = clock.ms();

    std::string box_counts;
    for ( const auto& it : iters )
        box_counts += ( box_counts.empty() ? "" : ", " ) + std::to_string( it.n_boxes );

    // ---- Converged partition of each method at the final time ----------------
    // A leaf's payload IS its flow map at t_final, so the final-time region of a
    // partition is just the boundary image of every leaf — no re-propagation.
    auto finalRegion = [&]( auto crit ) {
        auto tree = tax::ads::refine< P >( Verner89{}, crit, rhs(), box, icCenter(), 0.0, t_final,
                                           cfg, adsThreads() );
        std::vector< Polygon > polys;
        int id = 0;
        for ( int li : tree.done() )
        {
            const auto& leaf = tree.leaf( li );
            polys.push_back(
                evalPolygon( leaf.payload, boundary, boundaryToBox, id++, leaf.depth ) );
        }
        return polys;
    };
    const auto region_single = finalRegion( tax::ads::CoefficientMatchCriterion{ 1e-6, 0 } );
    const auto region_coeff = finalRegion( tax::ads::CoefficientMatchCriterion{ 1e-6, kMaxIter } );
    const auto region_volume =
        finalRegion( tax::ads::VolumeRatioCriterion{ 1e-6, kMaxIter, { kAxisY, kAxisVy }, 8 } );

    // ---- Write JSON (custom nested schema: iterations -> snapshots -> leaves) -
    std::ofstream out( outfile );
    out << std::setprecision( 14 );
    out << "{\n  \"method\": \"refine\",\n";
    out << "  \"params\": {\n";
    out << "    \"P\": " << P << ", \"M\": " << M << ", \"D\": " << D << ",\n";
    out << "    \"t_final\": " << jsonNumber( t_final ) << ", \"ecc\": " << jsonNumber( kEcc )
        << ",\n";
    out << "    \"tol\": 1e-6, \"box_scale\": " << jsonNumber( box_scale ) << ",\n";
    out << "    \"ic_center\": " << jsonArray( box.center )
        << ", \"ic_half_width\": " << jsonArray( box.halfWidth ) << ",\n";
    out << "    \"n_monte\": " << kNMonte << "\n  },\n";
    out << "  \"timing\": { \"elapsed_ms\": " << elapsed_ms << " },\n";

    out << "  \"reference_orbit\": {\n    \"t\": ";
    writeJsonArray( out, reference.t );
    out << ",\n    \"x0\": ";
    writeJsonArray( out, reference.cols[0] );
    out << ",\n    \"x1\": ";
    writeJsonArray( out, reference.cols[1] );
    out << "\n  },\n";

    out << "  \"snap_times\": ";
    writeJsonArray( out, snap_times );
    out << ",\n";

    // Monte-Carlo cloud per snapshot time.
    out << "  \"monte_carlo\": [\n";
    for ( std::size_t si = 0; si < snap_times.size(); ++si )
    {
        std::vector< double > xs, ys;
        xs.reserve( monte.size() );
        ys.reserve( monte.size() );
        for ( const auto& m : monte )
        {
            xs.push_back( m.truth_x[si] );
            ys.push_back( m.truth_y[si] );
        }
        out << "    { \"t\": " << snap_times[si] << ", \"x\": ";
        writeJsonArray( out, xs );
        out << ", \"y\": ";
        writeJsonArray( out, ys );
        out << " }" << ( si + 1 < snap_times.size() ? "," : "" ) << "\n";
    }
    out << "  ],\n";

    // Iterations.
    out << "  \"iterations\": [\n";
    for ( std::size_t ki = 0; ki < iters.size(); ++ki )
    {
        const auto& it = iters[ki];
        out << "    { \"iter\": " << it.max_depth << ", \"n_boxes\": " << it.n_boxes
            << ", \"area\": " << jsonNumber( it.area ) << ", \"rms\": " << jsonNumber( it.rms )
            << ", \"snapshots\": [\n";
        for ( std::size_t si = 0; si < it.snapshots.size(); ++si )
        {
            const auto& snap = it.snapshots[si];
            out << "      { \"t\": " << snap.t << ", \"leaves\": [";
            for ( std::size_t l = 0; l < snap.leaves.size(); ++l )
            {
                out << "\n        { \"id\": " << snap.leaves[l].id
                    << ", \"depth\": " << snap.leaves[l].depth << ", \"x\": ";
                writeJsonArray( out, snap.leaves[l].x );
                out << ", \"y\": ";
                writeJsonArray( out, snap.leaves[l].y );
                out << " }" << ( l + 1 < snap.leaves.size() ? "," : "" );
            }
            out << "\n      ] }" << ( si + 1 < it.snapshots.size() ? "," : "" ) << "\n";
        }
        out << "    ] }" << ( ki + 1 < iters.size() ? "," : "" ) << "\n";
    }
    out << "  ],\n";

    // Criterion comparison: box count and RMS-vs-Monte-Carlo per iteration.
    auto writeCurve = [&]( const char* name, const std::vector< Iteration >& curve, bool comma ) {
        out << "    \"" << name << "\": [";
        for ( std::size_t i = 0; i < curve.size(); ++i )
            out << ( i ? ", " : "" ) << "{ \"iter\": " << curve[i].max_depth
                << ", \"n_boxes\": " << curve[i].n_boxes
                << ", \"rms\": " << jsonNumber( curve[i].rms ) << " }";
        out << " ]" << ( comma ? "," : "" ) << "\n";
    };
    out << "  \"comparison\": {\n";
    writeCurve( "coefficient_match", iters, /*comma=*/true );
    writeCurve( "volume_ratio", vol_iters, /*comma=*/false );
    out << "  },\n";

    // Converged regions at the final time, one partition per method.
    auto writeRegion = [&]( const char* name, const std::vector< Polygon >& polys, bool comma ) {
        out << "    \"" << name << "\": [";
        for ( std::size_t l = 0; l < polys.size(); ++l )
        {
            out << ( l ? "," : "" ) << "\n      { \"id\": " << polys[l].id
                << ", \"depth\": " << polys[l].depth << ", \"x\": ";
            writeJsonArray( out, polys[l].x );
            out << ", \"y\": ";
            writeJsonArray( out, polys[l].y );
            out << " }";
        }
        out << "\n    ]" << ( comma ? "," : "" ) << "\n";
    };
    out << "  \"converged\": {\n    \"t\": " << t_final << ",\n";
    out << "    \"n_boxes\": { \"single\": " << region_single.size()
        << ", \"coefficient_match\": " << region_coeff.size()
        << ", \"volume_ratio\": " << region_volume.size() << " },\n";
    writeRegion( "single", region_single, /*comma=*/true );
    writeRegion( "coefficient_match", region_coeff, /*comma=*/true );
    writeRegion( "volume_ratio", region_volume, /*comma=*/false );
    out << "  }\n}\n";

    std::string vol_counts;
    for ( const auto& it : vol_iters )
        vol_counts += ( vol_counts.empty() ? "" : ", " ) + std::to_string( it.n_boxes );

    printBanner( "two_body/refine — propagate-then-assess ADS (criterion comparison, tol=1e-6)",
                 { { "P, M", std::to_string( P ) + ", " + std::to_string( M ) },
                   { "box scale", jsonNumber( box_scale ) },
                   { "coeff-match boxes", box_counts },
                   { "volume boxes", vol_counts },
                   { "coeff-match RMS", jsonNumber( iters.back().rms ) },
                   { "volume RMS", jsonNumber( vol_iters.back().rms ) },
                   { "elapsed", std::to_string( elapsed_ms ) + " ms" },
                   { "output", outfile } } );
    return 0;
}
