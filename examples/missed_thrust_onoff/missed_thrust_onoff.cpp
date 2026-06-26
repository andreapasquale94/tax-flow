// =============================================================================
// examples/missed_thrust_onoff/missed_thrust_onoff.cpp
//
// Dispersion set of a low-thrust spacecraft with a BANG-BANG (on/off) thruster
// under MISSED THRUST.
//
// The thruster is either fully ON or fully OFF and may only switch on a 4-day
// grid (a 4-day minimum dwell). Whether each 4-day arc is ON or OFF follows a
// 2-state Markov chain (ON->OFF with p_fail, OFF->ON with p_recover). On top of
// the binary schedule each ON arc carries small execution errors:
//   * a magnitude execution error  delta_m  (+-2%),
//   * a pointing  execution error  delta_th (+-5 deg).
//
// Method: the two execution errors are the DA expansion variables (M = 2); the
// box is the small (delta_m, delta_th) rectangle. For each Monte-Carlo sample
// (one on/off schedule) the DA state is propagated arc-by-arc, carried across
// the 4-day boundaries, so the integrator composes the per-arc flow maps. After
// every arc the physical state is a Taylor polynomial in (delta_m, delta_th);
// evaluating it over the box yields a cheap cloud of execution-error
// realisations (the polynomial surrogate). Pooling the clouds over many on/off
// schedules gives, at each 4-day snapshot, the missed-thrust dispersion set
// from which confidence bands are drawn.
//
// Run:    ./missed_thrust_onoff [reliable|intermittent|unreliable]
// Writes: missed_thrust_onoff_<scenario>.json
// Plot:   python3 examples/missed_thrust_onoff/plot.py \
//                 missed_thrust_onoff_reliable.json \
//                 missed_thrust_onoff_intermittent.json \
//                 missed_thrust_onoff_unreliable.json \
//                 --out missed_thrust_onoff.png
// =============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <string>
#include <tax/ads/domains/box.hpp>
#include <tax/ads/da_state.hpp>
#include <tax/ode.hpp>
#include <thread>
#include <vector>

#include "common.hpp"

namespace
{
using namespace example;
using namespace example::missed_thrust_onoff;
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
// per-arc ON/OFF histogram (how many schedules are OFF vs ON on this arc).
struct SnapAccum
{
    std::vector< Pt > pts;
    double sx = 0.0, sy = 0.0;
    std::array< long long, ThrusterModel::kNStates > lvl{};
};

// ---- ON/OFF schedule for one Monte-Carlo sample ----------------------------
std::array< int, kNArcs > drawSchedule( const ThrusterModel& th, Rng& rng )
{
    std::array< int, kNArcs > seq{};
    int state = th.initState;
    for ( int k = 0; k < kNArcs; ++k )
    {
        seq[static_cast< std::size_t >( k )] = state;
        state = th.next( state, rng.uniform() );
    }
    return seq;
}

// ---- Smooth reference orbit (constant on/off level) ------------------------
std::pair< std::vector< double >, std::vector< double > > referenceOrbit(
    double magBase, double thetaNom, const tax::ode::IntegratorConfig< double >& cfg )
{
    auto sol =
        tax::ode::propagate( Verner89{}, rhs( magBase, thetaNom ), stateIC(), 0.0, kHorizon, cfg );
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
    // ---- Select thruster-reliability scenario from the command line ----------
    const Scenario& scenario = [&]() -> const Scenario& {
        if ( argc > 1 )
        {
            const std::string arg( argv[1] );
            if ( arg == "reliable" ) return kReliable;
            if ( arg == "unreliable" ) return kUnreliable;
            if ( arg == "intermittent" ) return kIntermittent;
        }
        return kIntermittent;
    }();

    const Preset& preset = kSpacecraft;
    const double m_nom = aMax( preset );
    const double theta_nom = preset.thetaNomDeg * M_PI / 180.0;
    const ThrusterModel thruster = scenario.thruster;

    constexpr int kNSeq = 8000;  // on/off schedules (Monte-Carlo samples)
    constexpr int kNDraw = 16;   // execution-error draws per schedule (surrogate evals)
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

    // ---- Monte Carlo over on/off schedules, parallel over samples -----------
    const int nThreads = adsThreads();
    std::vector< std::array< SnapAccum, kNArcs > > perThread(
        static_cast< std::size_t >( nThreads ) );

    auto worker = [&]( int tid ) {
        auto& acc = perThread[static_cast< std::size_t >( tid )];
        for ( int i = tid; i < kNSeq; i += nThreads )
        {
            // Per-schedule RNG keyed by sample index => threading-independent output.
            Rng rng( kSeed + static_cast< std::uint64_t >( i ) * 0x9E3779B97F4A7C15ULL );
            const auto seq = drawSchedule( thruster, rng );

            // Fixed execution-error bias per realisation (normalised box coords).
            std::array< std::array< double, M >, kNDraw > loc{};
            for ( int j = 0; j < kNDraw; ++j )
                loc[static_cast< std::size_t >( j )] = { rng.symmetric( 1.0 ),
                                                         rng.symmetric( 1.0 ) };

            // Seed the identity DA state on the (delta_m, delta_th) box and carry
            // it forward arc by arc; the integrator composes the per-arc maps.
            auto x = tax::ads::create< P, M >( errBox, stateIC() );
            for ( int k = 0; k < kNArcs; ++k )
            {
                const int state = seq[static_cast< std::size_t >( k )];
                const double magBase = ThrusterModel::levelFrac( state ) * m_nom;
                auto sol = tax::ode::propagate( Verner89{}, rhs( magBase, theta_nom ), x, k * kArc,
                                                ( k + 1 ) * kArc, cfg_mc );
                x = sol.x.back();

                auto& snap = acc[static_cast< std::size_t >( k )];
                ++snap.lvl[static_cast< std::size_t >( state )];
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
            for ( int s = 0; s < ThrusterModel::kNStates; ++s )
                dst.lvl[static_cast< std::size_t >( s )] +=
                    src.lvl[static_cast< std::size_t >( s )];
        }

    // ---- Reference orbits (ballistic = always OFF, nominal = always ON) ------
    auto ballistic = referenceOrbit( 0.0, theta_nom, cfg_ref );
    auto nominal = referenceOrbit( m_nom, theta_nom, cfg_ref );

    // ---- Realised ON duty fraction (mean over all arcs/schedules) -----------
    long long onCount = 0, totCount = 0;
    for ( const auto& snap : snaps )
    {
        onCount += snap.lvl[1];
        for ( int s = 0; s < ThrusterModel::kNStates; ++s ) totCount += snap.lvl[s];
    }
    const double onFrac = totCount > 0 ? static_cast< double >( onCount ) / totCount : 0.0;

    // ---- Validation: surrogate vs direct integration on a few samples -------
    double max_err = 0.0;
    constexpr int kNCheck = 24;
    for ( int c = 0; c < kNCheck; ++c )
    {
        Rng rng( kSeed + static_cast< std::uint64_t >( c ) * 0x9E3779B97F4A7C15ULL );
        const auto seq = drawSchedule( thruster, rng );
        const std::array< double, M > l{ rng.symmetric( 1.0 ), rng.symmetric( 1.0 ) };
        const double dm = l[0] * kSigmaM, dth = l[1] * kSigmaTheta;

        auto x = tax::ads::create< P, M >( errBox, stateIC() );  // surrogate path
        auto xs = stateIC( dm, dth );                            // direct path
        for ( int k = 0; k < kNArcs; ++k )
        {
            const double magBase =
                ThrusterModel::levelFrac( seq[static_cast< std::size_t >( k )] ) * m_nom;
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
    out << "  \"method\": \"missed_thrust_onoff_mc\",\n";
    out << "  \"params\": {\n";
    out << "    \"scenario\": \"" << scenario.name << "\",\n";
    out << "    \"case\": \"" << preset.name << "\",\n";
    out << "    \"thrust_mN\": " << preset.thrustN * 1000 << ",\n";
    out << "    \"mass_kg\": " << preset.massKg << ",\n";
    out << "    \"a_max\": " << m_nom << ",\n";
    out << "    \"theta_nom_deg\": " << preset.thetaNomDeg << ",\n";
    out << "    \"sigma_m\": " << kSigmaM << ",\n";
    out << "    \"sigma_theta_deg\": " << kSigmaTheta * 180.0 / M_PI << ",\n";
    out << "    \"arc_days\": " << kArcDays << ",\n";
    out << "    \"n_arcs\": " << kNArcs << ",\n";
    out << "    \"n_levels\": " << ThrusterModel::kNStates << ",\n";
    out << "    \"on_fraction\": " << onFrac << ",\n";
    out << "    \"n_sequences\": " << kNSeq << ",\n";
    out << "    \"n_draws\": " << kNDraw << ",\n";
    out << "    \"P\": " << P << ",\n";
    out << "    \"thruster\": { \"p_fail\": " << thruster.pFail
        << ", \"p_recover\": " << thruster.pRecover << ", \"init_state\": " << thruster.initState
        << " }\n";
    out << "  },\n";
    out << "  \"timing\": { \"elapsed_ms\": " << elapsed_ms << " },\n";
    out << "  \"validation\": { \"max_pos_err\": " << max_err << ", \"n_check\": " << kNCheck
        << " },\n";
    out << "  \"grid\": { \"xmin\": " << xmin << ", \"xmax\": " << xmax << ", \"ymin\": " << ymin
        << ", \"ymax\": " << ymax << ", \"nx\": " << NX << ", \"ny\": " << NY << " },\n";

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
        for ( const auto& p : snap.pts ) dist.push_back( std::hypot( p.x - cx, p.y - cy ) );
        std::sort( dist.begin(), dist.end() );
        auto pct = [&]( double q ) {
            if ( dist.empty() ) return 0.0;
            const std::size_t idx =
                std::min( dist.size() - 1, static_cast< std::size_t >( q * ( dist.size() - 1 ) ) );
            return dist[idx];
        };

        long long nseq = 0;
        for ( int s = 0; s < ThrusterModel::kNStates; ++s )
            nseq += snap.lvl[static_cast< std::size_t >( s )];

        // Sigma envelope radii: 2-D Gaussian-equivalent coverage masses
        // (1s = 1-e^-0.5 = 39.35%, 2s = 1-e^-2 = 86.47%, 3s = 1-e^-4.5 = 98.89%).
        out << "    { \"day\": " << ( k + 1 ) * kArcDays << ", \"t\": " << ( k + 1 ) * kArc
            << ", \"n\": " << snap.pts.size() << ", \"mean\": [" << cx << "," << cy << "]"
            << ", \"r1s\": " << pct( 0.3935 ) << ", \"r2s\": " << pct( 0.8647 )
            << ", \"r3s\": " << pct( 0.9889 ) << ", \"level_dist\": [";
        for ( int s = 0; s < ThrusterModel::kNStates; ++s )
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

    printBanner(
        "missed_thrust_onoff — on/off thruster dispersion under missed thrust",
        { { "scenario", scenario.name },
          { "case", preset.name },
          { "a_max (nominal)", std::to_string( m_nom ) },
          { "exec errors", "+-2% mag, +-5 deg (when ON)" },
          { "decision grid", std::to_string( kNArcs ) + " arcs x " +
                                 std::to_string( static_cast< int >( kArcDays ) ) + " days" },
          { "thruster", "ON/OFF, p_fail=" + std::to_string( thruster.pFail ) +
                            ", p_recover=" + std::to_string( thruster.pRecover ) },
          { "ON duty (realised)", std::to_string( onFrac * 100.0 ) + "%" },
          { "samples",
            std::to_string( kNSeq ) + " sched x " + std::to_string( kNDraw ) + " draws" },
          { "validation", "max |dr| = " + std::to_string( max_err ) },
          { "elapsed", std::to_string( elapsed_ms ) + " ms" },
          { "output", scenario.outfile } } );
    return 0;
}
