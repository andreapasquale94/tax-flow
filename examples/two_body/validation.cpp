// =============================================================================
// examples/two_body/validation.cpp
//
// Validation harness: compare the constructed flow maps (single Taylor
// polynomial, ADS, LOADS) against a Monte-Carlo truth on the same IC
// box.
//
// Two products:
//
//   1. snapshot view  — at each of N_SNAPS times, dump the MC-truth
//      sample states and the envelope polygon obtained from a single
//      multivariate Taylor flow polynomial (P=6, evaluated on the IC
//      box boundary). Used by plot_validation.py to show how the
//      cloud spreads inside the polynomial envelope.
//
//   2. parameter sweep — for each (method, P, tol) cell, evaluate the
//      constructed flow at the MC samples *at the final time* and
//      record max + RMS position error and wall-clock. Sweep
//      P ∈ {4, 6, 8} and tol ∈ {1e-2, 1e-4, 1e-6}; Taylor has no tol
//      (only the order matters). The plot consumes this to render
//      error / timing as a function of P and tol.
//
// Run:    ./two_body_validation
// Writes: validation.json
// =============================================================================

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <tax/ads.hpp>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <vector>

#include "common.hpp"

namespace
{

using Vec4 = tax::la::VecNT< 4, double >;
using BoxT = tax::ads::Box< double, 4 >;
using namespace example::two_body;
using namespace tax::ode::methods;

constexpr int kNOrbits = 1;
constexpr int kNSnaps = 9;  // every 45 degrees including t = 0
constexpr int kNMc = 200;
constexpr int kNPerEdge = 24;

// ---- Map IC sample to a leaf-local point in [-1, 1]^M --------------------
//
// For axes with zero half-width the local coord is fixed at 0 (the box
// is degenerate along that axis and the polynomial does not depend on
// it).
inline std::array< double, 4 > toLocal( const Vec4& ic, const BoxT& box )
{
    std::array< double, 4 > out{};
    for ( int j = 0; j < 4; ++j )
    {
        out[static_cast< std::size_t >( j )] =
            box.halfWidth( j ) > 0.0 ? ( ic( j ) - box.center( j ) ) / box.halfWidth( j ) : 0.0;
    }
    return out;
}

// ---- Compute max + RMS position error of a constructed flow against MC ---
//
// `state_at_t` is the propagated DA state (Vec<4, TE<P, M>>). The leaf
// box is used to convert from MC IC coords to leaf-local coords.
template < class DAState >
std::pair< double, double > posErrors( const DAState& state_at_t, const BoxT& leaf_box,
                                       const std::vector< Vec4 >& mc_ic,
                                       const std::vector< Vec4 >& mc_truth_at_t,
                                       const std::vector< int >& mc_leaf_id, int this_leaf_id )
{
    double sum_sq = 0.0;
    double max_e = 0.0;
    int n = 0;
    for ( std::size_t i = 0; i < mc_ic.size(); ++i )
    {
        if ( mc_leaf_id[i] != this_leaf_id ) continue;
        const auto d = toLocal( mc_ic[i], leaf_box );
        const double px = state_at_t( 0 ).eval( d );
        const double py = state_at_t( 1 ).eval( d );
        const double ex = px - mc_truth_at_t[i]( 0 );
        const double ey = py - mc_truth_at_t[i]( 1 );
        const double e = std::sqrt( ex * ex + ey * ey );
        sum_sq += e * e;
        max_e = std::max( max_e, e );
        ++n;
    }
    if ( n == 0 ) return { 0.0, 0.0 };
    return { max_e, std::sqrt( sum_sq / static_cast< double >( n ) ) };
}

// ---- A single sweep cell -------------------------------------------------
struct Cell
{
    std::string method;
    int P;
    double tol;  // NaN when N/A (Taylor)
    int n_leaves;
    double max_pos_err;
    double rms_pos_err;
    double elapsed_ms;
};

// ---- One Taylor cell (single polynomial) --------------------------------
template < int P >
Cell runTaylor( const BoxT& ic_box, const std::vector< Vec4 >& mc_ic,
                const std::vector< Vec4 >& mc_truth_final,
                const tax::ode::IntegratorConfig< double >& cfg, double tFinal )
{
    using TE = tax::TE< P, 4 >;
    using DAState = tax::la::VecNT< 4, TE >;

    DAState x0 = tax::ads::create< P, 4 >( ic_box, icCenter() );
    const auto t0 = std::chrono::high_resolution_clock::now();
    auto sol = tax::ode::propagate( Verner89{}, rhs(), x0, 0.0, tFinal, cfg );
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration< double, std::milli >( t1 - t0 ).count();

    const auto& xT = sol.x.back();
    double sum_sq = 0.0, max_e = 0.0;
    for ( std::size_t i = 0; i < mc_ic.size(); ++i )
    {
        const auto d = toLocal( mc_ic[i], ic_box );
        const double px = xT( 0 ).eval( d );
        const double py = xT( 1 ).eval( d );
        const double ex = px - mc_truth_final[i]( 0 );
        const double ey = py - mc_truth_final[i]( 1 );
        const double e = std::sqrt( ex * ex + ey * ey );
        sum_sq += e * e;
        max_e = std::max( max_e, e );
    }
    const double rms = std::sqrt( sum_sq / static_cast< double >( mc_ic.size() ) );
    return { "taylor", P, std::nan( "" ), 1, max_e, rms, ms };
}

// ---- One ADS/LOADS cell ------------------------------------------------
template < int P, class Criterion >
Cell runAdsLike( std::string name, Criterion crit, const BoxT& ic_box,
                 const std::vector< Vec4 >& mc_ic, const std::vector< Vec4 >& mc_truth_final,
                 const tax::ode::IntegratorConfig< double >& cfg, double tFinal )
{
    const double tol = crit.tol;

    const auto t0 = std::chrono::high_resolution_clock::now();
    auto tree =
        tax::ads::propagate< P >( Verner89{}, crit, rhs(), ic_box, icCenter(), 0.0, tFinal, cfg )
            .tree();
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration< double, std::milli >( t1 - t0 ).count();

    // Assign each MC IC to its containing leaf.
    std::vector< int > mc_leaf( mc_ic.size(), -1 );
    for ( std::size_t i = 0; i < mc_ic.size(); ++i )
    {
        const auto idx = tree.leaf( mc_ic[i] );
        if ( idx.has_value() ) mc_leaf[i] = *idx;
    }

    double sum_sq = 0.0, max_e = 0.0;
    int n_used = 0;
    for ( int li : tree.done() )
    {
        const auto& leaf = tree.leaf( li );
        const auto [leaf_max, leaf_rms] =
            posErrors( leaf.payload, leaf.box, mc_ic, mc_truth_final, mc_leaf, li );
        // Recount sum_sq / max_e by re-walking
        for ( std::size_t i = 0; i < mc_ic.size(); ++i )
        {
            if ( mc_leaf[i] != li ) continue;
            const auto d = toLocal( mc_ic[i], leaf.box );
            const double px = leaf.payload( 0 ).eval( d );
            const double py = leaf.payload( 1 ).eval( d );
            const double ex = px - mc_truth_final[i]( 0 );
            const double ey = py - mc_truth_final[i]( 1 );
            const double e = std::sqrt( ex * ex + ey * ey );
            sum_sq += e * e;
            max_e = std::max( max_e, e );
            ++n_used;
        }
        (void)leaf_max;
        (void)leaf_rms;
    }
    const double rms = n_used > 0 ? std::sqrt( sum_sq / static_cast< double >( n_used ) ) : 0.0;
    return { name, P, tol, static_cast< int >( tree.done().size() ), max_e, rms, ms };
}

// ---- Sweep for a given P -------------------------------------------------
template < int P >
void sweepP( const BoxT& ic_box, const std::vector< Vec4 >& mc_ic,
             const std::vector< Vec4 >& mc_truth_final,
             const tax::ode::IntegratorConfig< double >& cfg, double tFinal,
             const std::vector< double >& tols, std::vector< Cell >& cells )
{
    cells.push_back( runTaylor< P >( ic_box, mc_ic, mc_truth_final, cfg, tFinal ) );
    for ( double tol : tols )
    {
        cells.push_back( runAdsLike< P >( "ads",
                                          tax::ads::TruncationCriterion{ tol, /*maxDepth=*/8 },
                                          ic_box, mc_ic, mc_truth_final, cfg, tFinal ) );
        cells.push_back( runAdsLike< P >( "loads", tax::ads::NliCriterion{ tol, /*maxDepth=*/5 },
                                          ic_box, mc_ic, mc_truth_final, cfg, tFinal ) );
    }
}

}  // namespace

int main()
{
    const double tFinal = kNOrbits * kPeriod;

    // ---- Configs ------------------------------------------------------------
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    tax::ode::IntegratorConfig< double > ref_cfg;
    ref_cfg.abstol = ref_cfg.reltol = 1e-13;

    // ---- IC box -------------------------------------------------------------
    auto ic_box = icBox();

    // ---- Snapshot times -----------------------------------------------------
    const auto times = example::linspace( 0.0, tFinal, kNSnaps );

    // ---- Monte-Carlo samples + truth ---------------------------------------
    std::cout << "[validation] propagating " << kNMc << " Monte-Carlo samples..." << std::flush;
    std::mt19937 rng( 42 );
    std::uniform_real_distribution< double > uni( -1.0, 1.0 );

    std::vector< Vec4 > mc_ic;
    std::vector< std::vector< Vec4 > > mc_truth( kNSnaps );
    mc_ic.reserve( kNMc );
    for ( auto& snap : mc_truth ) snap.reserve( kNMc );

    // Helper: find stored step with t closest to t_query.
    auto stateAt = []( const auto& sol, double t_query ) -> decltype( sol.x.front() ) {
        auto it = std::lower_bound( sol.t.begin(), sol.t.end(), t_query );
        if ( it == sol.t.end() ) return sol.x.back();
        if ( it == sol.t.begin() ) return sol.x.front();
        auto prev = std::prev( it );
        const std::size_t idx = ( t_query - *prev < *it - t_query )
                                    ? static_cast< std::size_t >( prev - sol.t.begin() )
                                    : static_cast< std::size_t >( it - sol.t.begin() );
        return sol.x[idx];
    };

    ref_cfg.save_steps = true;

    for ( int i = 0; i < kNMc; ++i )
    {
        // Sample in normalized (xi_y, xi_vy); pinned axes at 0.
        tax::la::VecNT< 4, double > xi{ 0.0, uni( rng ), 0.0, uni( rng ) };
        const Vec4 ic = ic_box.denormalize( xi );
        mc_ic.push_back( ic );

        auto sol = tax::ode::propagate( Taylor< 16 >{}, rhs(), ic, 0.0, tFinal, ref_cfg );
        for ( int s = 0; s < kNSnaps; ++s )
            mc_truth[static_cast< std::size_t >( s )].push_back(
                stateAt( sol, times[static_cast< std::size_t >( s )] ) );
    }
    const auto& mc_truth_final = mc_truth[static_cast< std::size_t >( kNSnaps - 1 )];

    // ---- Reference orbit ----------------------------------------------------
    auto ref_sol = tax::ode::propagate( Taylor< 16 >{}, rhs(), icCenter(), 0.0, tFinal, ref_cfg );

    // ---- ADS envelope polygons per snapshot --------------------------------
    //
    // For each snapshot time t_snap we run an independent ADS propagation
    // (truncation criterion, P=6, tol=1e-4) and dump the (x, y) boundary
    // image of every done leaf. The collection of polygons is what the
    // envelope figure renders against the MC samples.
    constexpr int kEnvP = 6;
    constexpr double kEnvTol = 1e-4;

    const auto boundary = unitSquareBoundary( kNPerEdge );

    struct LeafPolygon
    {
        std::vector< double > xs, ys;
    };
    std::vector< std::vector< LeafPolygon > > ads_leaves_per_snap( kNSnaps );

    std::cout << "[validation] computing per-snapshot ADS envelopes..." << std::flush;
    for ( int s = 0; s < kNSnaps; ++s )
    {
        const double t_snap = times[static_cast< std::size_t >( s )];

        if ( t_snap <= 0.0 )
        {
            // Trivial: one polygon equal to the IC box image (in state space
            // this is the box itself).
            LeafPolygon poly{};
            poly.xs.resize( boundary.size() );
            poly.ys.resize( boundary.size() );
            for ( std::size_t v = 0; v < boundary.size(); ++v )
            {
                const tax::la::VecNT< 4, double > d{ 0.0, boundary[v][0], 0.0, boundary[v][1] };
                const auto pt = ic_box.denormalize( d );
                poly.xs[v] = pt( 0 );
                poly.ys[v] = pt( 1 );
            }
            ads_leaves_per_snap[static_cast< std::size_t >( s )].push_back( std::move( poly ) );
            continue;
        }

        auto tree = tax::ads::propagate< kEnvP >(
                        Verner89{}, tax::ads::TruncationCriterion{ kEnvTol, /*maxDepth=*/8 }, rhs(),
                        ic_box, icCenter(), 0.0, t_snap, cfg )
                        .tree();

        for ( int li : tree.done() )
        {
            const auto& leaf = tree.leaf( li );
            LeafPolygon poly{};
            poly.xs.resize( boundary.size() );
            poly.ys.resize( boundary.size() );
            for ( std::size_t v = 0; v < boundary.size(); ++v )
            {
                const std::array< double, 4 > d{ 0.0, boundary[v][0], 0.0, boundary[v][1] };
                poly.xs[v] = leaf.payload( 0 ).eval( d );
                poly.ys[v] = leaf.payload( 1 ).eval( d );
            }
            ads_leaves_per_snap[static_cast< std::size_t >( s )].push_back( std::move( poly ) );
        }
    }
    std::cout << " done.\n";

    // ---- Parameter sweep ----------------------------------------------------
    const std::vector< double > tols{ 1e-2, 1e-4, 1e-6 };

    std::cout << "[validation] running parameter sweep (P, tol)..." << std::flush;
    std::vector< Cell > cells;
    sweepP< 4 >( ic_box, mc_ic, mc_truth_final, cfg, tFinal, tols, cells );
    sweepP< 6 >( ic_box, mc_ic, mc_truth_final, cfg, tFinal, tols, cells );
    sweepP< 8 >( ic_box, mc_ic, mc_truth_final, cfg, tFinal, tols, cells );
    std::cout << " done.\n";

    // ---- JSON output --------------------------------------------------------
    std::ofstream out( "validation.json" );
    out << std::setprecision( 12 );
    out << "{\n";
    out << "  \"config\": {\n";
    out << "    \"n_mc\": " << kNMc << ",\n";
    out << "    \"n_snaps\": " << kNSnaps << ",\n";
    out << "    \"n_orbits\": " << kNOrbits << ",\n";
    out << "    \"t_final\": " << tFinal << ",\n";
    out << "    \"envelope_P\":   " << kEnvP << ",\n";
    out << "    \"envelope_tol\": " << kEnvTol << ",\n";
    out << "    \"P_list\": [4, 6, 8],\n";
    out << "    \"tol_list\": ";
    writeJsonArray( out, tols );
    out << ",\n";
    out << "    \"ic_box\": {\n";
    out << "      \"center\":    ";
    writeJsonArray( out, ic_box.center );
    out << ",\n";
    out << "      \"halfWidth\": ";
    writeJsonArray( out, ic_box.halfWidth );
    out << "\n";
    out << "    }\n";
    out << "  },\n";

    //   reference orbit — stored accepted-step grid
    out << "  \"reference_orbit\": {\n";
    out << "    \"t\":  ";
    writeJsonArray( out, ref_sol.t );
    out << ",\n";
    std::vector< double > col( ref_sol.t.size() );
    for ( int j = 0; j < 4; ++j )
    {
        for ( std::size_t i = 0; i < ref_sol.t.size(); ++i ) col[i] = ref_sol.x[i]( j );
        out << "    \"x" << j << "\": ";
        writeJsonArray( out, col );
        out << ( j + 1 < 4 ? ",\n" : "\n" );
    }
    out << "  },\n";

    //   snapshots: MC truth + ADS leaf envelope
    out << "  \"snapshots\": [\n";
    for ( int s = 0; s < kNSnaps; ++s )
    {
        const double t = times[static_cast< std::size_t >( s )];
        const auto& pts = mc_truth[static_cast< std::size_t >( s )];
        std::vector< double > xs, ys;
        xs.reserve( pts.size() );
        ys.reserve( pts.size() );
        for ( const auto& p : pts )
        {
            xs.push_back( p( 0 ) );
            ys.push_back( p( 1 ) );
        }
        out << "    {\n";
        out << "      \"t\":    " << t << ",\n";
        out << "      \"mc_x\": ";
        writeJsonArray( out, xs );
        out << ",\n";
        out << "      \"mc_y\": ";
        writeJsonArray( out, ys );
        out << ",\n";
        out << "      \"ads_leaves\": [";
        const auto& leaves = ads_leaves_per_snap[static_cast< std::size_t >( s )];
        for ( std::size_t li = 0; li < leaves.size(); ++li )
        {
            if ( li ) out << ",";
            out << "\n        { \"x\": ";
            writeJsonArray( out, leaves[li].xs );
            out << ", \"y\": ";
            writeJsonArray( out, leaves[li].ys );
            out << " }";
        }
        out << ( leaves.empty() ? "" : "\n      " ) << "]\n";
        out << "    }" << ( s + 1 < kNSnaps ? "," : "" ) << "\n";
    }
    out << "  ],\n";

    //   sweep cells
    out << "  \"sweep\": [\n";
    for ( std::size_t i = 0; i < cells.size(); ++i )
    {
        const auto& c = cells[i];
        out << "    { \"method\": \"" << c.method << "\""
            << ", \"P\": " << c.P << ", \"tol\": "
            << ( std::isnan( c.tol ) ? std::string( "null" ) : std::to_string( c.tol ) )
            << ", \"n_leaves\": " << c.n_leaves << ", \"max_pos_err\": " << c.max_pos_err
            << ", \"rms_pos_err\": " << c.rms_pos_err << ", \"elapsed_ms\": " << c.elapsed_ms
            << " }" << ( i + 1 < cells.size() ? "," : "" ) << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    // ---- Terminal banner ----------------------------------------------------
    const std::vector< std::pair< std::string, std::string > > rows{
        { "MC samples", std::to_string( kNMc ) },
        { "snapshots", std::to_string( kNSnaps ) },
        { "envelope P", std::to_string( kEnvP ) },
        { "sweep P", "[4, 6, 8]" },
        { "sweep tol", "[1e-2, 1e-4, 1e-6]" },
        { "sweep cells", std::to_string( cells.size() ) },
        { "output", "validation.json" } };
    printBanner( "validation", rows );

    // Compact table of (method, P, tol) → error + timing.
    std::cout << "  " << std::left << std::setw( 7 ) << "method"
              << "  " << std::setw( 3 ) << "P"
              << "  " << std::setw( 10 ) << "tol"
              << "  " << std::setw( 5 ) << "leafs"
              << "  " << std::setw( 11 ) << "max err"
              << "  " << std::setw( 11 ) << "rms err"
              << "  " << std::setw( 10 ) << "elapsed" << '\n';
    std::cout << "  " << std::string( 7 + 2 + 3 + 2 + 10 + 2 + 5 + 2 + 11 + 2 + 11 + 2 + 10, '-' )
              << '\n';
    for ( const auto& c : cells )
    {
        std::cout << "  " << std::setw( 7 ) << c.method << "  " << std::setw( 3 ) << c.P << "  "
                  << std::setw( 10 )
                  << ( std::isnan( c.tol )
                           ? std::string( "  -" )
                           : ( std::ostringstream{} << std::scientific << std::setprecision( 0 )
                                                    << c.tol )
                                 .str() )
                  << "  " << std::setw( 5 ) << c.n_leaves << "  " << std::setw( 11 )
                  << std::scientific << std::setprecision( 3 ) << c.max_pos_err << "  "
                  << std::setw( 11 ) << c.rms_pos_err << "  " << std::fixed
                  << std::setprecision( 1 ) << std::setw( 8 ) << c.elapsed_ms << " ms" << '\n';
    }
    return 0;
}
