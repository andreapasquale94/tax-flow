// =============================================================================
// examples/flyby/flyby.cpp
//
// Reachable set of gravity-assist OUTCOMES for a planetary flyby, in the planar
// Sun-Jupiter CR3BP.
//
// The mission's design freedom is its AIM POINT: the incoming position is a
// free 2-D variable inside a small square upstream of Jupiter (velocity fixed).
// As the aim sweeps the square the spacecraft passes Jupiter at different impact
// parameters, and the flyby deflects it by wildly different amounts -- a turn
// angle that is violently nonlinear near a grazing pass. A SINGLE ADS
// propagation maps the whole aim box through the encounter; Automatic Domain
// Splitting bisects it, concentrating its leaves on the close-approach corner.
// Sampling the leaf flow maps yields the reachable set of post-flyby outcomes:
// the achievable heliocentric energies and outgoing directions.
//
// Run:    ./flyby [slow|medium|fast]
// Writes: flyby_<scenario>.json
// Plot:   python3 examples/flyby/plot.py \
//                 flyby_slow.json flyby_medium.json flyby_fast.json \
//                 --out flyby.png
// =============================================================================

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <tax/ads.hpp>
#include <tax/ode.hpp>
#include <vector>

#include "common.hpp"

namespace
{
using namespace example;
using namespace example::flyby;
using tax::ode::methods::Verner89;

constexpr int P = 6;  // DA truncation order in the 2 aim coordinates
constexpr int M = 2;  // aim expansion variables (x, y incoming position)
constexpr int D = 4;  // state dimension
constexpr int NB = 150;  // outcome-grid resolution per aim axis
constexpr double kTend = 3.0;  // propagation time (past closest approach, downstream)

// Evaluate the ADS flow map at a global aim offset (gx, gy) in
// [-kAimHalf, kAimHalf]^2 through the containing leaf; returns the final state.
template < class Tree >
[[nodiscard]] bool evalState( const Tree& tree, double gx, double gy,
                              tax::la::VecNT< 4, double >& out )
{
    for ( int li : tree.done() )
    {
        const auto& leaf = tree.leaf( li );
        const double dx = gx - leaf.box.center( 0 );
        const double dy = gy - leaf.box.center( 1 );
        if ( std::abs( dx ) <= leaf.box.halfWidth( 0 ) + 1e-12 &&
             std::abs( dy ) <= leaf.box.halfWidth( 1 ) + 1e-12 )
        {
            const std::array< double, 2 > loc{ dx / leaf.box.halfWidth( 0 ),
                                               dy / leaf.box.halfWidth( 1 ) };
            for ( int c = 0; c < 4; ++c ) out( c ) = leaf.payload( c ).eval( loc );
            return true;
        }
    }
    return false;
}

}  // namespace

int main( int argc, char** argv )
{
    const Scenario& scenario = [&]() -> const Scenario& {
        if ( argc > 1 )
        {
            const std::string arg( argv[1] );
            if ( arg == "slow" ) return kSlow;
            if ( arg == "fast" ) return kFast;
            if ( arg == "medium" ) return kMedium;
        }
        return kMedium;
    }();

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-11;
    cfg.max_steps = 2'000'000;  // close grazing passes need many small steps
    cfg.save_steps = false;

    tax::ode::IntegratorConfig< double > cfg_traj = cfg;
    cfg_traj.save_steps = true;

    const tax::ads::TruncationCriterion criterion{ /*tol=*/1e-5, /*maxDepth=*/18 };
    const auto box = aimBox();
    const auto ic = aimCenter( scenario );

    Stopwatch clock;

    // ---- One ADS propagation of the aim box through the flyby ---------------
    auto tree = tax::ads::propagate< P >( Verner89{}, criterion, rhs(), box, ic, 0.0, kTend, cfg,
                                          adsThreads() );
    int n_leaves = 0;
    std::vector< std::array< double, 4 > > leaf_boxes;  // cx, cy, hx, hy (aim-offset units)
    for ( int li : tree.done() )
    {
        const auto& leaf = tree.leaf( li );
        leaf_boxes.push_back( { leaf.box.center( 0 ), leaf.box.center( 1 ), leaf.box.halfWidth( 0 ),
                                leaf.box.halfWidth( 1 ) } );
        ++n_leaves;
    }

    // ---- Outcome grid over the aim box (via the leaf flow maps) -------------
    // E[i] heliocentric post-flyby energy, A[i] outgoing angle (deg).
    std::vector< double > gridE( static_cast< std::size_t >( NB * NB ), 0.0 );
    std::vector< double > gridA( static_cast< std::size_t >( NB * NB ), 0.0 );
    double eMin = 1e9, eMax = -1e9;
    for ( int iy = 0; iy < NB; ++iy )
        for ( int ix = 0; ix < NB; ++ix )
        {
            const double gx = -kAimHalf + 2.0 * kAimHalf * ( ix + 0.5 ) / NB;
            const double gy = -kAimHalf + 2.0 * kAimHalf * ( iy + 0.5 ) / NB;
            tax::la::VecNT< 4, double > s;
            const std::size_t idx = static_cast< std::size_t >( iy * NB + ix );
            if ( evalState( tree, gx, gy, s ) )
            {
                const auto oc = outcome( s );
                gridE[idx] = oc.energy;
                gridA[idx] = oc.angleDeg;
                eMin = std::min( eMin, oc.energy );
                eMax = std::max( eMax, oc.energy );
            }
            else
            {
                gridE[idx] = std::nan( "" );
                gridA[idx] = std::nan( "" );
            }
        }

    // ---- Trajectory bundle: scalar sweep along the transverse aim diagonal --
    // Offset along (1,-1)/sqrt2 (perpendicular to the approach velocity) sweeps
    // the impact parameter; each path is integrated with step saving for plotting.
    struct Traj
    {
        double s;        // sweep parameter in [-1, 1]
        double dmin;     // closest approach to Jupiter (Hill radii)
        double energy;   // post-flyby heliocentric energy
        std::vector< double > x, y;
    };
    std::vector< Traj > bundle;
    constexpr int kNTraj = 41;
    for ( int i = 0; i < kNTraj; ++i )
    {
        const double t = -1.0 + 2.0 * i / ( kNTraj - 1 );
        auto s0 = aimState( scenario, t, -t );  // along (1,-1): transverse to velocity
        auto sol = tax::ode::propagate( Verner89{}, rhs(), s0, 0.0, kTend, cfg_traj );
        Traj tr;
        tr.s = t;
        tr.dmin = 1e9;
        tr.x.reserve( sol.x.size() );
        tr.y.reserve( sol.x.size() );
        for ( const auto& s : sol.x )
        {
            tr.dmin = std::min( tr.dmin, distToJupiter( s ) / kHill );
            tr.x.push_back( s( 0 ) );
            tr.y.push_back( s( 1 ) );
        }
        tr.energy = outcome( sol.x.back() ).energy;
        bundle.push_back( std::move( tr ) );
    }

    // ---- Validation: surrogate (leaf flow map) vs direct integration --------
    double max_err = 0.0;
    Rng rng( 0xBEEF1234ULL );
    for ( int c = 0; c < 24; ++c )
    {
        const double xix = rng.symmetric(), xiy = rng.symmetric();
        tax::la::VecNT< 4, double > ss;
        if ( !evalState( tree, kAimHalf * xix, kAimHalf * xiy, ss ) ) continue;
        auto s0 = aimState( scenario, xix, xiy );
        auto sol = tax::ode::propagate( Verner89{}, rhs(), s0, 0.0, kTend, cfg );
        const double ex = ss( 0 ) - sol.x.back()( 0 );
        const double ey = ss( 1 ) - sol.x.back()( 1 );
        max_err = std::max( max_err, std::hypot( ex, ey ) );
    }

    const double elapsed_ms = clock.ms();

    // ---- Write JSON ---------------------------------------------------------
    std::ofstream out( scenario.outfile );
    out << std::setprecision( 10 );
    out << "{\n";
    out << "  \"method\": \"flyby_ads\",\n";
    out << "  \"params\": {\n";
    out << "    \"scenario\": \"" << scenario.name << "\",\n";
    out << "    \"mu\": " << kMu << ",\n";
    out << "    \"jupiter_x\": " << kJupiterX << ",\n";
    out << "    \"sun_x\": " << kSunX << ",\n";
    out << "    \"hill\": " << kHill << ",\n";
    out << "    \"rho0\": " << kRho0 << ",\n";
    out << "    \"v_approach\": " << scenario.vApproach << ",\n";
    out << "    \"aim_half\": " << kAimHalf << ",\n";
    out << "    \"t_end\": " << kTend << ",\n";
    out << "    \"P\": " << P << ",\n";
    out << "    \"n_leaves\": " << n_leaves << ",\n";
    out << "    \"nb\": " << NB << ",\n";
    out << "    \"e_min\": " << eMin << ",\n";
    out << "    \"e_max\": " << eMax << "\n";
    out << "  },\n";
    out << "  \"timing\": { \"elapsed_ms\": " << elapsed_ms << " },\n";
    out << "  \"validation\": { \"max_pos_err\": " << max_err << " },\n";

    // Outcome grids (row-major NB x NB over [-aim_half, aim_half]^2).
    auto writeFlat = [&]( const char* key, const std::vector< double >& v, bool comma ) {
        out << "  \"" << key << "\": [";
        for ( std::size_t i = 0; i < v.size(); ++i )
        {
            if ( i ) out << ",";
            if ( std::isnan( v[i] ) )
                out << "null";
            else
                out << v[i];
        }
        out << "]" << ( comma ? "," : "" ) << "\n";
    };
    writeFlat( "grid_energy", gridE, true );
    writeFlat( "grid_angle", gridA, true );

    // ADS leaf boxes (aim-offset units): cx, cy, hx, hy.
    out << "  \"leaves\": [";
    for ( std::size_t i = 0; i < leaf_boxes.size(); ++i )
    {
        const auto& b = leaf_boxes[i];
        out << ( i ? "," : "" ) << "[" << b[0] << "," << b[1] << "," << b[2] << "," << b[3] << "]";
    }
    out << "],\n";

    // Trajectory bundle.
    out << "  \"bundle\": [\n";
    for ( std::size_t i = 0; i < bundle.size(); ++i )
    {
        const auto& tr = bundle[i];
        out << "    { \"s\": " << tr.s << ", \"dmin\": " << tr.dmin << ", \"energy\": " << tr.energy
            << ", \"x\": [";
        for ( std::size_t j = 0; j < tr.x.size(); ++j ) out << ( j ? "," : "" ) << tr.x[j];
        out << "], \"y\": [";
        for ( std::size_t j = 0; j < tr.y.size(); ++j ) out << ( j ? "," : "" ) << tr.y[j];
        out << "] }" << ( i + 1 < bundle.size() ? "," : "" ) << "\n";
    }
    out << "  ]\n}\n";
    out.close();

    printBanner( "flyby — gravity-assist reachable-set of outcomes (Sun-Jupiter CR3BP)",
                 { { "scenario", scenario.name },
                   { "mu (Sun-Jupiter)", std::to_string( kMu ) },
                   { "approach speed", std::to_string( scenario.vApproach ) },
                   { "aim box half-width", std::to_string( kAimHalf ) + " (" +
                                               std::to_string( kAimHalf / kHill ) + " Hill)" },
                   { "uncertainty", "M=2 aim (x,y), P=" + std::to_string( P ) },
                   { "ADS leaves", std::to_string( n_leaves ) },
                   { "energy range", std::to_string( eMin ) + " .. " + std::to_string( eMax ) },
                   { "validation", "max |dr| = " + std::to_string( max_err ) },
                   { "elapsed", std::to_string( elapsed_ms ) + " ms" },
                   { "output", scenario.outfile } } );
    return 0;
}
