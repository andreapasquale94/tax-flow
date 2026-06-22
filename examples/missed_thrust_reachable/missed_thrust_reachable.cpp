// =============================================================================
// examples/missed_thrust_reachable/missed_thrust_reachable.cpp
//
// SET-VALUED reachable set of a low-thrust spacecraft under thrust outages.
//
// The deterministic counterpart of the probabilistic dispersion example
// (examples/missed_thrust/). An outage over one revolution is parameterized by
// THREE continuous descriptors — onset tau, duration w and depth d — which are
// the expansion variables of the DA flow map (M = 3). A SINGLE ADS propagation
// per snapshot covers the whole outage box; ADS splits it where the onset
// sweeping across the snapshot turns the flow map nonlinear. The reachable set
// is the union of the leaf images: we sample each leaf's flow polynomial over
// its sub-box, pool the (x, y) points into a shared coverage grid, and the
// outer support of that grid is the reachable-set envelope. The coverage count
// doubles as a "robustness" heat — how much of the outage-parameter space maps
// into each region.
//
// Run:    ./missed_thrust_reachable [mild|moderate|severe]
// Writes: missed_thrust_reachable_<scenario>.json
// Plot:   python3 examples/missed_thrust_reachable/plot.py \
//                 missed_thrust_reachable_mild.json \
//                 missed_thrust_reachable_moderate.json \
//                 missed_thrust_reachable_severe.json \
//                 --out missed_thrust_reachable.png
// =============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>
#include <tax/ads.hpp>
#include <tax/ode.hpp>
#include <thread>
#include <vector>

#include "common.hpp"

namespace
{
using namespace example;
using namespace example::missed_thrust_reachable;
using tax::ode::methods::Verner89;

constexpr int P = 5;  // DA truncation order in the 3 outage axes
constexpr int M = 3;  // outage expansion variables (tau, w, d)
constexpr int D = 7;  // state dimension

constexpr int NX = 130, NY = 130;             // coverage grid resolution
constexpr int kSnapEvery = 2;                 // snapshot every 2 arcs (every 20 deg)
constexpr long long kSamplesTotal = 1200000;  // box samples per snapshot (volume-weighted)
constexpr int kSamplesMinLeaf = 32;           // floor of samples per leaf

// Coverage grid: a flat NX*NY count buffer with a fixed extent.
struct Grid
{
    double xmin, xmax, ymin, ymax;
    std::vector< long long > cov =
        std::vector< long long >( static_cast< std::size_t >( NX * NY ), 0 );

    // Accumulate one (x, y) hit; points outside the grid are dropped (not
    // clamped) so they cannot pile up a false high-count line on the edge.
    void add( double x, double y )
    {
        const int ix = static_cast< int >( ( x - xmin ) / ( xmax - xmin ) * NX );
        const int iy = static_cast< int >( ( y - ymin ) / ( ymax - ymin ) * NY );
        if ( ix < 0 || ix >= NX || iy < 0 || iy >= NY ) return;
        ++cov[static_cast< std::size_t >( iy * NX + ix )];
    }
};

// Sample one snapshot's reachable set into a coverage grid: walk the leaves,
// draw volume-weighted uniform samples inside each leaf's local [-1,1]^M box,
// evaluate the flow polynomial for (x, y) and accumulate the hit counts.
// Parallel over leaves; per-thread grids are merged by the caller.
template < class Tree >
Grid sampleReachable( const Tree& tree, const Grid& proto )
{
    // Collect leaf indices and the total box volume for weighting.
    std::vector< int > leaves;
    double totVol = 0.0;
    for ( int li : tree.done() )
    {
        leaves.push_back( li );
        double v = 1.0;
        for ( int a = 0; a < M; ++a ) v *= tree.leaf( li ).box.halfWidth( a );
        totVol += v;
    }
    const int nThreads = std::max( 1, adsThreads() );
    std::vector< Grid > part( static_cast< std::size_t >( nThreads ), proto );

    auto worker = [&]( int tid ) {
        Grid& g = part[static_cast< std::size_t >( tid )];
        Rng rng( 0xA5A5A5A5ULL ^
                 ( static_cast< std::uint64_t >( tid + 1 ) * 0x9E3779B97F4A7C15ULL ) );
        for ( std::size_t k = static_cast< std::size_t >( tid ); k < leaves.size();
              k += static_cast< std::size_t >( nThreads ) )
        {
            const auto& leaf = tree.leaf( leaves[k] );
            double v = 1.0;
            for ( int a = 0; a < M; ++a ) v *= leaf.box.halfWidth( a );
            const long long ns = std::max< long long >(
                kSamplesMinLeaf, std::llround( static_cast< double >( kSamplesTotal ) *
                                               ( totVol > 0 ? v / totVol : 0.0 ) ) );
            for ( long long s = 0; s < ns; ++s )
            {
                std::array< double, M > loc;
                for ( int a = 0; a < M; ++a )
                    loc[static_cast< std::size_t >( a )] = rng.symmetric();
                const double px = leaf.payload( 3 ).eval( loc );  // x = state comp 3
                const double py = leaf.payload( 4 ).eval( loc );  // y = state comp 4
                // Sanity clip: the reachable set lies between the ballistic
                // circle (r = 1) and the nominal spiral; anything well beyond
                // is an under-resolved-leaf artifact, not a reachable point.
                if ( !std::isfinite( px ) || !std::isfinite( py ) || px * px + py * py > 4.0 )
                    continue;
                g.add( px, py );
            }
        }
    };

    std::vector< std::thread > pool;
    for ( int t = 0; t < nThreads; ++t ) pool.emplace_back( worker, t );
    for ( auto& th : pool ) th.join();

    Grid out = proto;
    for ( const auto& g : part )
        for ( std::size_t i = 0; i < out.cov.size(); ++i ) out.cov[i] += g.cov[i];
    return out;
}

}  // namespace

int main( int argc, char** argv )
{
    // ---- Select outage-severity scenario from the command line --------------
    const Scenario& scenario = [&]() -> const Scenario& {
        if ( argc > 1 )
        {
            const std::string arg( argv[1] );
            if ( arg == "mild" ) return kMild;
            if ( arg == "severe" ) return kSevere;
            if ( arg == "moderate" ) return kModerate;
        }
        return kModerate;
    }();

    const Preset& preset = kSpacecraft;
    const double m_nom = aMax( preset );
    const double theta_nom = preset.thetaNomDeg * M_PI / 180.0;

    // ADS config: only the final-time flow map per leaf is needed.
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-11;
    cfg.save_steps = false;

    // Reference config: keep the accepted-step grid for smooth plotted curves.
    tax::ode::IntegratorConfig< double > cfg_ref;
    cfg_ref.abstol = cfg_ref.reltol = 1e-12;
    cfg_ref.save_steps = true;

    // Resolve the onset sweep (the dominant nonlinearity); the onset axis needs
    // several splits to bring each leaf's tau-cell down to the edge ramp width.
    const tax::ads::TruncationCriterion criterion{ /*tol=*/5e-4, /*maxDepth=*/10 };

    const auto box = outageBox( scenario );
    const auto ic = stateIC( scenario );

    // ---- Reference orbits (nominal = full thrust, ballistic = all missed) ---
    auto nominal = referenceOrbit( m_nom, theta_nom, cfg_ref );
    auto ballistic = referenceOrbit( 0.0, theta_nom, cfg_ref );

    // ---- Shared plotting extent (the nominal spiral bounds the reachable set)
    Grid proto{ 0.0, 0.0, 0.0, 0.0 };  // include the Sun at the origin
    auto extend = [&]( double x, double y ) {
        proto.xmin = std::min( proto.xmin, x );
        proto.xmax = std::max( proto.xmax, x );
        proto.ymin = std::min( proto.ymin, y );
        proto.ymax = std::max( proto.ymax, y );
    };
    for ( std::size_t i = 0; i < nominal.first.size(); ++i )
        extend( nominal.first[i], nominal.second[i] );
    for ( std::size_t i = 0; i < ballistic.first.size(); ++i )
        extend( ballistic.first[i], ballistic.second[i] );
    {
        const double mx = 0.18 * ( proto.xmax - proto.xmin );
        const double my = 0.18 * ( proto.ymax - proto.ymin );
        proto.xmin -= mx;
        proto.xmax += mx;
        proto.ymin -= my;
        proto.ymax += my;
    }

    // ---- One ADS propagation per snapshot time, then sample the leaves ------
    std::vector< double > snap_times;
    for ( int k = kSnapEvery; k <= kNArcs; k += kSnapEvery ) snap_times.push_back( k * kArc );

    Stopwatch clock;
    std::vector< Grid > grids;
    std::string leaf_counts;
    for ( double t : snap_times )
    {
        auto tree = tax::ads::propagate< P >( Verner89{}, criterion, rhs( m_nom, theta_nom ), box,
                                              ic, 0.0, t, cfg, adsThreads() );
        int n_leaves = 0;
        for ( int li : tree.done() )
        {
            (void)li;
            ++n_leaves;
        }
        leaf_counts += ( leaf_counts.empty() ? "" : "," ) + std::to_string( n_leaves );
        grids.push_back( sampleReachable( tree, proto ) );
    }
    const double elapsed_ms = clock.ms();

    // ---- Write JSON ---------------------------------------------------------
    std::ofstream out( scenario.outfile );
    out << std::setprecision( 10 );
    out << "{\n";
    out << "  \"method\": \"missed_thrust_reachable_ads\",\n";
    out << "  \"params\": {\n";
    out << "    \"scenario\": \"" << scenario.name << "\",\n";
    out << "    \"case\": \"" << preset.name << "\",\n";
    out << "    \"a_max\": " << m_nom << ",\n";
    out << "    \"theta_nom_deg\": " << preset.thetaNomDeg << ",\n";
    out << "    \"w_max_frac\": " << scenario.wMaxFrac << ",\n";
    out << "    \"d_max\": " << scenario.dMax << ",\n";
    out << "    \"n_arcs\": " << kNArcs << ",\n";
    out << "    \"P\": " << P << ",\n";
    out << "    \"M\": " << M << ",\n";
    out << "    \"D\": " << D << "\n";
    out << "  },\n";
    out << "  \"timing\": { \"elapsed_ms\": " << elapsed_ms << " },\n";
    out << "  \"grid\": { \"xmin\": " << proto.xmin << ", \"xmax\": " << proto.xmax
        << ", \"ymin\": " << proto.ymin << ", \"ymax\": " << proto.ymax << ", \"nx\": " << NX
        << ", \"ny\": " << NY << " },\n";

    auto writeCurve = [&]( const char* key,
                           const std::pair< std::vector< double >, std::vector< double > >& c,
                           bool comma ) {
        out << "  \"" << key << "\": { \"x\": [";
        for ( std::size_t i = 0; i < c.first.size(); ++i ) out << ( i ? "," : "" ) << c.first[i];
        out << "], \"y\": [";
        for ( std::size_t i = 0; i < c.second.size(); ++i ) out << ( i ? "," : "" ) << c.second[i];
        out << "] }" << ( comma ? "," : "" ) << "\n";
    };
    writeCurve( "ballistic_orbit", ballistic, true );
    writeCurve( "nominal_orbit", nominal, true );

    out << "  \"snapshots\": [\n";
    for ( std::size_t k = 0; k < snap_times.size(); ++k )
    {
        const double t = snap_times[k];
        out << "    { \"deg\": " << ( t / kArc ) * kDegPerArc << ", \"t\": " << t << ", \"cov\": [";
        const auto& cov = grids[k].cov;
        for ( std::size_t i = 0; i < cov.size(); ++i ) out << ( i ? "," : "" ) << cov[i];
        out << "] }" << ( k + 1 < snap_times.size() ? "," : "" ) << "\n";
    }
    out << "  ]\n}\n";
    out.close();

    printBanner(
        "missed_thrust_reachable — set-valued reachable set under thrust outages",
        { { "scenario", scenario.name },
          { "case", preset.name },
          { "a_max (nominal)", std::to_string( m_nom ) },
          { "outage budget", "w_max = " + std::to_string( scenario.wMaxFrac ) +
                                 " T, d_max = " + std::to_string( scenario.dMax ) },
          { "uncertainty", "M=3 (tau, w, d), P=" + std::to_string( P ) },
          { "snapshots", std::to_string( snap_times.size() ) + " (every " +
                             std::to_string( kSnapEvery * static_cast< int >( kDegPerArc ) ) +
                             " deg)" },
          { "leaves per snap", leaf_counts },
          { "elapsed", std::to_string( elapsed_ms ) + " ms" },
          { "output", scenario.outfile } } );
    return 0;
}
