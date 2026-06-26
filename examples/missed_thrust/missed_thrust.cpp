// =============================================================================
// examples/missed_thrust/missed_thrust.cpp
//
// Reachable / dispersion set of a low-thrust spacecraft under MISSED THRUST.
//
// The spacecraft follows a nominal constant-magnitude, constant-direction
// low-thrust plan over one heliocentric revolution. Reality departs from it by
//   * a small magnitude execution error  delta_m  (+-2%),
//   * a small pointing  execution error  delta_th (+-5 deg),
//   * MISSED THRUST: the delivered fraction follows a 5-state Markov chain over
//     {0,25,50,75,100}% that transitions once per 10-degree arc.
//
// Method: the two execution errors are the DA expansion variables (M = 2); the
// box is the small (delta_m, delta_th) rectangle. For each Monte-Carlo sample
// (one Markov sequence) the DA state is propagated arc-by-arc, carried across
// the 10-degree boundaries, so the integrator composes the per-arc flow maps.
// After every arc the physical state is a Taylor polynomial in (delta_m,
// delta_th); evaluating it over the box yields a cheap cloud of execution-error
// realisations (the polynomial surrogate). Pooling the clouds over many Markov
// sequences gives, at each 10-degree snapshot, the missed-thrust dispersion set
// from which confidence bands are drawn.
//
// Run:    ./missed_thrust
// Writes: missed_thrust.json
// Plot:   python3 examples/missed_thrust/plot.py missed_thrust.json \
//                 --out missed_thrust.png
// =============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>
#include <thread>
#include <tax/ads/domains/box.hpp>
#include <tax/ads/da_state.hpp>
#include <tax/ode.hpp>
#include <vector>

#include "common.hpp"

namespace
{
using namespace example;
using namespace example::missed_thrust;
using tax::ode::methods::Verner89;

constexpr int P = 6;  // DA truncation order in (delta_m, delta_th)
constexpr int M = 2;  // execution-error expansion variables
constexpr int D = 6;  // state dimension

// One Monte-Carlo trajectory snapshot point.
struct Pt
{
    double x, y;
};

// Per-snapshot accumulator: a point cloud, a running centroid sum, and the
// per-arc thrust-level histogram (how many sequences sit at each of the five
// missed-thrust levels on this arc — the Markov marginal at this time).
struct SnapAccum
{
    std::vector< Pt > pts;
    double sx = 0.0, sy = 0.0;
    std::array< long long, MarkovModel::kNStates > lvl{};
};

// ---- Markov sequence + execution draws for one Monte-Carlo sample ----------
// Returns the per-arc level indices; fills `loc` with J normalised box draws
// (xi in [-1,1]^2 ⇒ delta_m = xi0*sigma_m, delta_th = xi1*sigma_theta).
std::array< int, kNArcs > drawSequence( const MarkovModel& mk, Rng& rng )
{
    std::array< int, kNArcs > seq{};
    int state = mk.initState;
    for ( int k = 0; k < kNArcs; ++k )
    {
        seq[static_cast< std::size_t >( k )] = state;
        state = mk.next( state, rng.uniform() );
    }
    return seq;
}

// ---- Smooth reference orbit (constant thrust level) ------------------------
// Integrate one revolution at a fixed commanded magnitude (delta = 0) and
// return the (x, y) along the accepted-step grid for a clean plotted curve.
std::pair< std::vector< double >, std::vector< double > > referenceOrbit(
    double magBase, double thetaNom, const tax::ode::IntegratorConfig< double >& cfg )
{
    auto sol = tax::ode::propagate( Verner89{}, rhs( magBase, thetaNom ), stateIC(), 0.0, kPeriod,
                                    cfg );
    std::vector< double > xs, ys;
    xs.reserve( sol.x.size() );
    ys.reserve( sol.x.size() );
    for ( const auto& s : sol.x )
    {
        xs.push_back( s( 2 ) );
        ys.push_back( s( 3 ) );
    }
    return { std::move( xs ), std::move( ys ) };
}

}  // namespace

int main( int argc, char** argv )
{
    // ---- Select missed-thrust scenario from the command line -----------------
    const Scenario& scenario = [&]() -> const Scenario& {
        if ( argc > 1 )
        {
            const std::string arg( argv[1] );
            if ( arg == "optimistic" ) return kOptimistic;
            if ( arg == "pessimistic" ) return kPessimistic;
            if ( arg == "intermediate" ) return kIntermediate;
        }
        return kIntermediate;
    }();

    const Preset& preset = kSpacecraft;
    const double m_nom = aMax( preset );
    const double theta_nom = preset.thetaNomDeg * M_PI / 180.0;
    const MarkovModel markov = scenario.markov;

    constexpr int kNSeq = 10000;  // Markov sequences (Monte-Carlo samples)
    constexpr int kNDraw = 16;    // execution-error draws per sequence (surrogate evals)
    constexpr std::uint64_t kSeed = 0xC0FFEE123456789ULL;

    // DA config: only the final state of each arc is needed -> no step saving.
    tax::ode::IntegratorConfig< double > cfg_mc;
    cfg_mc.abstol = cfg_mc.reltol = 1e-11;
    cfg_mc.save_steps = false;

    // Reference config: keep the accepted-step grid for smooth plotted curves.
    tax::ode::IntegratorConfig< double > cfg_ref;
    cfg_ref.abstol = cfg_ref.reltol = 1e-12;
    cfg_ref.save_steps = true;

    const tax::ads::Box< double, M > errBox{ { 0.0, 0.0 }, { kSigmaM, kSigmaTheta } };

    Stopwatch clock;

    // ---- Monte Carlo over Markov sequences, parallel over samples -----------
    const int nThreads = adsThreads();
    std::vector< std::array< SnapAccum, kNArcs > > perThread( static_cast< std::size_t >( nThreads ) );

    auto worker = [&]( int tid ) {
        auto& acc = perThread[static_cast< std::size_t >( tid )];
        for ( int i = tid; i < kNSeq; i += nThreads )
        {
            // Per-sequence RNG keyed by sample index ⇒ threading-independent output.
            Rng rng( kSeed + static_cast< std::uint64_t >( i ) * 0x9E3779B97F4A7C15ULL );
            const auto seq = drawSequence( markov, rng );

            // Fixed execution-error bias per realisation (normalised box coords).
            std::array< std::array< double, M >, kNDraw > loc{};
            for ( int j = 0; j < kNDraw; ++j )
                loc[static_cast< std::size_t >( j )] = { rng.symmetric( 1.0 ), rng.symmetric( 1.0 ) };

            // Seed the identity DA state on the (delta_m, delta_th) box and carry
            // it forward arc by arc; the integrator composes the per-arc maps.
            auto x = tax::ads::create< P, M >( errBox, stateIC() );
            for ( int k = 0; k < kNArcs; ++k )
            {
                const int level = seq[static_cast< std::size_t >( k )];
                const double magBase = MarkovModel::levelFrac( level ) * m_nom;
                auto sol = tax::ode::propagate( Verner89{}, rhs( magBase, theta_nom ), x,
                                                k * kArc, ( k + 1 ) * kArc, cfg_mc );
                x = sol.x.back();

                auto& snap = acc[static_cast< std::size_t >( k )];
                ++snap.lvl[static_cast< std::size_t >( level )];
                for ( int j = 0; j < kNDraw; ++j )
                {
                    const double px = x( 2 ).eval( loc[static_cast< std::size_t >( j )] );
                    const double py = x( 3 ).eval( loc[static_cast< std::size_t >( j )] );
                    snap.pts.push_back( { px, py } );
                    snap.sx += px;
                    snap.sy += py;
                }
            }
        }
    };

    {
        std::vector< std::thread > pool;
        for ( int t = 0; t < nThreads; ++t ) pool.emplace_back( worker, t );
        for ( auto& th : pool ) th.join();
    }

    // ---- Merge per-thread accumulators --------------------------------------
    std::array< SnapAccum, kNArcs > snaps;
    for ( const auto& acc : perThread )
        for ( int k = 0; k < kNArcs; ++k )
        {
            auto& dst = snaps[static_cast< std::size_t >( k )];
            const auto& src = acc[static_cast< std::size_t >( k )];
            dst.pts.insert( dst.pts.end(), src.pts.begin(), src.pts.end() );
            dst.sx += src.sx;
            dst.sy += src.sy;
            for ( int s = 0; s < MarkovModel::kNStates; ++s )
                dst.lvl[static_cast< std::size_t >( s )] += src.lvl[static_cast< std::size_t >( s )];
        }

    // ---- Reference orbits (ballistic = all 0%, nominal = all 100%) ----------
    auto ballistic = referenceOrbit( 0.0, theta_nom, cfg_ref );
    auto nominal = referenceOrbit( m_nom, theta_nom, cfg_ref );

    // ---- Validation: surrogate vs direct integration on a few samples -------
    double max_err = 0.0;
    constexpr int kNCheck = 24;
    for ( int c = 0; c < kNCheck; ++c )
    {
        Rng rng( kSeed + static_cast< std::uint64_t >( c ) * 0x9E3779B97F4A7C15ULL );
        const auto seq = drawSequence( markov, rng );
        const std::array< double, M > l{ rng.symmetric( 1.0 ), rng.symmetric( 1.0 ) };
        const double dm = l[0] * kSigmaM, dth = l[1] * kSigmaTheta;

        // Surrogate path (carry DA state, eval at final time).
        auto x = tax::ads::create< P, M >( errBox, stateIC() );
        // Direct path (scalar state with the same fixed errors).
        auto xs = stateIC( dm, dth );
        for ( int k = 0; k < kNArcs; ++k )
        {
            const double magBase = MarkovModel::levelFrac( seq[static_cast< std::size_t >( k )] ) * m_nom;
            auto sa = tax::ode::propagate( Verner89{}, rhs( magBase, theta_nom ), x, k * kArc,
                                           ( k + 1 ) * kArc, cfg_mc );
            x = sa.x.back();
            auto sb = tax::ode::propagate( Verner89{}, rhs( magBase, theta_nom ), xs, k * kArc,
                                           ( k + 1 ) * kArc, cfg_mc );
            xs = sb.x.back();
        }
        const double ex = x( 2 ).eval( l ) - xs( 2 );
        const double ey = x( 3 ).eval( l ) - xs( 3 );
        max_err = std::max( max_err, std::sqrt( ex * ex + ey * ey ) );
    }

    const double elapsed_ms = clock.ms();

    // ---- Shared plotting extent over all clouds + reference orbits ----------
    double xmin = 0.0, xmax = 0.0, ymin = 0.0, ymax = 0.0;  // include Sun at origin
    auto extend = [&]( double x, double y ) {
        xmin = std::min( xmin, x );
        xmax = std::max( xmax, x );
        ymin = std::min( ymin, y );
        ymax = std::max( ymax, y );
    };
    for ( const auto& snap : snaps )
        for ( const auto& p : snap.pts ) extend( p.x, p.y );
    for ( std::size_t i = 0; i < ballistic.first.size(); ++i )
        extend( ballistic.first[i], ballistic.second[i] );
    for ( std::size_t i = 0; i < nominal.first.size(); ++i )
        extend( nominal.first[i], nominal.second[i] );
    const double mx = 0.05 * ( xmax - xmin ), my = 0.05 * ( ymax - ymin );
    xmin -= mx;
    xmax += mx;
    ymin -= my;
    ymax += my;

    constexpr int NX = 130, NY = 130;
    auto binIndex = [&]( double v, double lo, double hi, int n ) {
        int idx = static_cast< int >( ( v - lo ) / ( hi - lo ) * n );
        return idx < 0 ? 0 : ( idx >= n ? n - 1 : idx );
    };

    // ---- Write JSON ---------------------------------------------------------
    std::ofstream out( scenario.outfile );
    out << std::setprecision( 10 );
    out << "{\n";
    out << "  \"method\": \"missed_thrust_mc\",\n";
    out << "  \"params\": {\n";
    out << "    \"scenario\": \"" << scenario.name << "\",\n";
    out << "    \"case\": \"" << preset.name << "\",\n";
    out << "    \"thrust_mN\": " << preset.thrustN * 1000 << ",\n";
    out << "    \"mass_kg\": " << preset.massKg << ",\n";
    out << "    \"a_max\": " << m_nom << ",\n";
    out << "    \"theta_nom_deg\": " << preset.thetaNomDeg << ",\n";
    out << "    \"sigma_m\": " << kSigmaM << ",\n";
    out << "    \"sigma_theta_deg\": " << kSigmaTheta * 180.0 / M_PI << ",\n";
    out << "    \"n_arcs\": " << kNArcs << ",\n";
    out << "    \"n_levels\": " << MarkovModel::kNStates << ",\n";
    out << "    \"n_sequences\": " << kNSeq << ",\n";
    out << "    \"n_draws\": " << kNDraw << ",\n";
    out << "    \"P\": " << P << ",\n";
    out << "    \"markov\": { \"p_down\": " << markov.pDown << ", \"p_up\": " << markov.pUp
        << ", \"init_state\": " << markov.initState << " }\n";
    out << "  },\n";
    out << "  \"timing\": { \"elapsed_ms\": " << elapsed_ms << " },\n";
    out << "  \"validation\": { \"max_pos_err\": " << max_err << ", \"n_check\": " << kNCheck
        << " },\n";
    out << "  \"grid\": { \"xmin\": " << xmin << ", \"xmax\": " << xmax << ", \"ymin\": " << ymin
        << ", \"ymax\": " << ymax << ", \"nx\": " << NX << ", \"ny\": " << NY << " },\n";

    auto writeCurve = [&]( const char* key, const std::pair< std::vector< double >,
                                                             std::vector< double > >& c, bool comma ) {
        out << "  \"" << key << "\": { \"x\": [";
        for ( std::size_t i = 0; i < c.first.size(); ++i )
            out << ( i ? "," : "" ) << c.first[i];
        out << "], \"y\": [";
        for ( std::size_t i = 0; i < c.second.size(); ++i )
            out << ( i ? "," : "" ) << c.second[i];
        out << "] }" << ( comma ? "," : "" ) << "\n";
    };
    writeCurve( "ballistic_orbit", ballistic, true );
    writeCurve( "nominal_orbit", nominal, true );

    out << "  \"snapshots\": [\n";
    for ( int k = 0; k < kNArcs; ++k )
    {
        const auto& snap = snaps[static_cast< std::size_t >( k )];
        std::vector< int > hist( static_cast< std::size_t >( NX * NY ), 0 );
        for ( const auto& p : snap.pts )
        {
            const int ix = binIndex( p.x, xmin, xmax, NX );
            const int iy = binIndex( p.y, ymin, ymax, NY );
            ++hist[static_cast< std::size_t >( iy * NX + ix )];
        }
        const double n = static_cast< double >( snap.pts.size() );
        const double cx = n > 0 ? snap.sx / n : 0.0;
        const double cy = n > 0 ? snap.sy / n : 0.0;

        // Radial dispersion envelope: percentiles of |r - mean| over the cloud.
        std::vector< double > dist;
        dist.reserve( snap.pts.size() );
        for ( const auto& p : snap.pts )
            dist.push_back( std::hypot( p.x - cx, p.y - cy ) );
        std::sort( dist.begin(), dist.end() );
        auto pct = [&]( double q ) {
            if ( dist.empty() ) return 0.0;
            const std::size_t idx = std::min(
                dist.size() - 1, static_cast< std::size_t >( q * ( dist.size() - 1 ) ) );
            return dist[idx];
        };

        // Thrust-level marginal at this arc (fraction of sequences per level).
        long long nseq = 0;
        for ( int s = 0; s < MarkovModel::kNStates; ++s ) nseq += snap.lvl[static_cast< std::size_t >( s )];

        // Sigma envelope radii: 2-D Gaussian-equivalent coverage masses
        // (1σ = 1-e^-0.5 = 39.35%, 2σ = 1-e^-2 = 86.47%, 3σ = 1-e^-4.5 = 98.89%).
        out << "    { \"deg\": " << ( k + 1 ) * kDegPerArc << ", \"t\": " << ( k + 1 ) * kArc
            << ", \"n\": " << snap.pts.size() << ", \"mean\": [" << cx << "," << cy << "]"
            << ", \"r1s\": " << pct( 0.3935 ) << ", \"r2s\": " << pct( 0.8647 )
            << ", \"r3s\": " << pct( 0.9889 ) << ", \"level_dist\": [";
        for ( int s = 0; s < MarkovModel::kNStates; ++s )
            out << ( s ? "," : "" )
                << ( nseq > 0 ? static_cast< double >( snap.lvl[static_cast< std::size_t >( s )] ) /
                                    static_cast< double >( nseq )
                              : 0.0 );
        out << "], \"hist\": [";
        for ( std::size_t i = 0; i < hist.size(); ++i ) out << ( i ? "," : "" ) << hist[i];
        out << "] }" << ( k + 1 < kNArcs ? "," : "" ) << "\n";
    }
    out << "  ]\n}\n";
    out.close();

    printBanner( "missed_thrust — low-thrust reachable set under missed thrust (Markov chain)",
                 { { "scenario", scenario.name },
                   { "case", preset.name },
                   { "a_max (nominal)", std::to_string( m_nom ) },
                   { "exec errors", "+-" + std::to_string( kSigmaM * 100 ) + "% mag, +-" +
                                        std::to_string( preset.thetaNomDeg == 0.0 ? 5.0 : 5.0 ) +
                                        " deg" },
                   { "arcs / snapshots", std::to_string( kNArcs ) + " (every " +
                                             std::to_string( static_cast< int >( kDegPerArc ) ) +
                                             " deg)" },
                   { "Markov", "5 levels, p_down=" + std::to_string( markov.pDown ) +
                                   ", p_up=" + std::to_string( markov.pUp ) },
                   { "samples", std::to_string( kNSeq ) + " seq x " + std::to_string( kNDraw ) +
                                    " draws" },
                   { "validation", "max |dr| = " + std::to_string( max_err ) },
                   { "elapsed", std::to_string( elapsed_ms ) + " ms" },
                   { "output", scenario.outfile } } );
    return 0;
}
