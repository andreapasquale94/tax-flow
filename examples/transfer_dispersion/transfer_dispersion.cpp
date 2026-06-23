// =============================================================================
// examples/transfer_dispersion/transfer_dispersion.cpp
//
// Dispersion sets of a low-thrust Earth -> NEA transfer (heliocentric two-body).
// One 6-D DA state s = [delta_m, delta_th, x, y, vx, vy] carries every
// uncertainty; each case (initial / thrust / both) is a choice of box
// half-widths. The flow map is carried arc-by-arc through the nominal
// thrust-coast-thrust plan; sampling its polynomial over the box at snapshots
// yields the dispersion-set cloud (its convex hull is drawn by plot.py).
//
// Run:    ./transfer_dispersion [low|med|high]
// Writes: transfer_dispersion_<level>.json
// Plot:   python3 examples/transfer_dispersion/plot.py \
//                 transfer_dispersion_low.json transfer_dispersion_med.json \
//                 transfer_dispersion_high.json --out transfer_dispersion.png
// =============================================================================

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <tax/ads/box.hpp>
#include <tax/ads/da_state.hpp>
#include <tax/ode.hpp>
#include <vector>

#include "common.hpp"

namespace
{
using namespace example;
using namespace example::transfer_dispersion;
using tax::ode::methods::Verner89;

constexpr int P = 4;  // DA truncation order
constexpr int M = 6;  // expansion variables (delta_m, delta_th, x, y, vx, vy)
constexpr int D = 6;  // state dimension

// One snapshot of the dispersion set: nominal centre + the sampled (x, y) cloud.
struct Snap
{
    double t, cx, cy;
    std::vector< double > x, y;
};

// Box sample directions in [-1, 1]^6: all 2^6 vertices + random interior points.
std::vector< std::array< double, 6 > > makeSamples()
{
    std::vector< std::array< double, 6 > > xi;
    for ( int v = 0; v < 64; ++v )
    {
        std::array< double, 6 > p{};
        for ( int a = 0; a < 6; ++a ) p[static_cast< std::size_t >( a )] = ( v >> a ) & 1 ? 1.0 : -1.0;
        xi.push_back( p );
    }
    Rng rng( 0xD1597E54ULL );
    for ( int k = 0; k < 220; ++k )
    {
        std::array< double, 6 > p{};
        for ( int a = 0; a < 6; ++a ) p[static_cast< std::size_t >( a )] = rng.symmetric();
        xi.push_back( p );
    }
    return xi;
}

}  // namespace

int main( int argc, char** argv )
{
    const Preset& preset = [&]() -> const Preset& {
        if ( argc > 1 )
        {
            const std::string a( argv[1] );
            if ( a == "low" ) return kLow;
            if ( a == "high" ) return kHigh;
            if ( a == "med" ) return kMed;
        }
        return kMed;
    }();

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-11;
    cfg.save_steps = false;

    const auto xi = makeSamples();
    Stopwatch clock;

    // Arc schedule: (magnitude, direction, duration, #sub-snapshots).
    struct Arc
    {
        double mag, phi, dur;
        int nsub;
    };
    const std::array< Arc, 3 > arcs{ Arc{ preset.aLt, preset.phi1, preset.tau1, 12 },
                                     Arc{ 0.0, preset.phi1, preset.coast, 16 },
                                     Arc{ preset.aLt, preset.phi2, preset.tau2, 12 } };
    constexpr int kCloudEvery = 6;  // cloud (set) snapshot cadence; centre saved every step

    const std::array< DispCase, 3 > cases{ kInit, kThrust, kBoth };
    std::vector< std::vector< Snap > > caseSnaps( 3 );
    std::vector< std::array< double, 2 > > nominal;  // dense centre trajectory (shared)
    std::vector< double > nominalT;                   // its times (from departure)

    for ( std::size_t ci = 0; ci < 3; ++ci )
    {
        auto x = tax::ads::create< P, M >( dispBox( cases[ci] ), stateIC() );
        int step = 0;
        auto record = [&]( double t ) {
            const std::array< double, 6 > zero{};
            const double cx = x( 2 ).eval( zero ), cy = x( 3 ).eval( zero );
            if ( ci == 0 )  // nominal is case-independent
            {
                nominal.push_back( { cx, cy } );
                nominalT.push_back( t );
            }
            if ( step % kCloudEvery == 0 )
            {
                Snap sn;
                sn.t = t;
                sn.cx = cx;
                sn.cy = cy;
                sn.x.reserve( xi.size() );
                sn.y.reserve( xi.size() );
                for ( const auto& p : xi )
                {
                    sn.x.push_back( x( 2 ).eval( p ) );
                    sn.y.push_back( x( 3 ).eval( p ) );
                }
                caseSnaps[ci].push_back( std::move( sn ) );
            }
            ++step;
        };
        record( 0.0 );
        double t = 0.0;
        for ( const auto& arc : arcs )
        {
            const double dt = arc.dur / arc.nsub;
            for ( int k = 0; k < arc.nsub; ++k )
            {
                auto sol = tax::ode::propagate( Verner89{}, rhs( arc.mag, arc.phi ), x, t, t + dt, cfg );
                x = sol.x.back();
                t += dt;
                record( t );
            }
        }
    }
    const double elapsed_ms = clock.ms();

    // ---- Write JSON ---------------------------------------------------------
    std::ofstream out( preset.outfile );
    out << std::setprecision( 10 );
    out << "{\n  \"method\": \"transfer_dispersion\",\n";
    out << "  \"params\": { \"level\": \"" << preset.name << "\", \"a_lt\": " << preset.aLt
        << ", \"T\": " << preset.T << ", \"td\": " << kTd << ", \"P\": " << P
        << ", \"pos_km\": 1000, \"vel_ms\": 1, \"sig_m\": " << kSigM
        << ", \"sig_th_deg\": 5 },\n";
    out << "  \"nea\": { \"a\": " << kNeaA << ", \"e\": " << kNeaE << ", \"w\": " << kNeaW
        << ", \"M0\": " << kNeaM0 << " },\n";
    out << "  \"timing\": { \"elapsed_ms\": " << elapsed_ms << " },\n";
    out << "  \"nominal\": { \"t\": [";
    for ( std::size_t i = 0; i < nominalT.size(); ++i ) out << ( i ? "," : "" ) << nominalT[i];
    out << "], \"x\": [";
    for ( std::size_t i = 0; i < nominal.size(); ++i ) out << ( i ? "," : "" ) << nominal[i][0];
    out << "], \"y\": [";
    for ( std::size_t i = 0; i < nominal.size(); ++i ) out << ( i ? "," : "" ) << nominal[i][1];
    out << "] },\n";
    out << "  \"cases\": [\n";
    for ( std::size_t ci = 0; ci < 3; ++ci )
    {
        out << "    { \"name\": \"" << cases[ci].name << "\", \"snapshots\": [\n";
        const auto& snaps = caseSnaps[ci];
        for ( std::size_t s = 0; s < snaps.size(); ++s )
        {
            const auto& sn = snaps[s];
            out << "      { \"t\": " << sn.t << ", \"c\": [" << sn.cx << "," << sn.cy
                << "], \"x\": [";
            for ( std::size_t i = 0; i < sn.x.size(); ++i ) out << ( i ? "," : "" ) << sn.x[i];
            out << "], \"y\": [";
            for ( std::size_t i = 0; i < sn.y.size(); ++i ) out << ( i ? "," : "" ) << sn.y[i];
            out << "] }" << ( s + 1 < snaps.size() ? "," : "" ) << "\n";
        }
        out << "    ] }" << ( ci + 1 < 3 ? "," : "" ) << "\n";
    }
    out << "  ]\n}\n";
    out.close();

    printBanner( "transfer_dispersion — low-thrust Earth->NEA dispersion sets",
                 { { "thrust level", preset.name },
                   { "a_lt", std::to_string( preset.aLt ) },
                   { "T (yr)", std::to_string( preset.T / kYear ) },
                   { "cases", "initial (+-1000 km, +-1 m/s), thrust (+-2%, +-5 deg), both" },
                   { "P, M, D", std::to_string( P ) + ", " + std::to_string( M ) + ", " +
                                    std::to_string( D ) },
                   { "snapshots/case", std::to_string( caseSnaps[0].size() ) },
                   { "elapsed", std::to_string( elapsed_ms ) + " ms" },
                   { "output", preset.outfile } } );
    return 0;
}
