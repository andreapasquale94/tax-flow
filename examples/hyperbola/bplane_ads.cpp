// =============================================================================
// examples/hyperbola/bplane_ads.cpp
//
// Stage 2 — the closest-approach event surface found while ADS is acting.
//
// Stage 1 located a single flow map's closest approach. Here the IC set is
// split by Automatic Domain Splitting, so the event surface is found *piecewise*
// — every leaf finds the closest approach of its OWN sub-region's center orbit.
//
// The trick is to hand the driver a *terminal* r . v = 0 event. Because the
// driver runs user events before its internal SplitEvent, a leaf that reaches
// periapsis terminates there (its flow map is captured at closest approach)
// rather than splitting; splits therefore happen strictly on the approach leg,
// before periapsis. Every done leaf's payload is thus its closest-approach flow
// map, and the done leaves tile the whole IC set at the event. The user
// convenience wrapper tax::ads::propagate<P> does NOT forward user events, so we
// drive an AdsDriver directly.
//
// We run the SAME oriented uncertainty two ways — as its axis-aligned bounding
// Box and as the oriented Zonotope — and overlay both leaf tilings on a
// Monte-Carlo cloud in the (B.T, B.R) plane. The oriented domain tiles the
// event surface with fewer leaves.
//
// Run:    ./hyperbola_bplane_ads
// Writes: hyperbola_bplane_ads.json  (plot with plot_hyperbola_bplane.py)
// =============================================================================

#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <tax/ads.hpp>
#include <tax/ode.hpp>
#include <tax/ode/events/root_finding_event.hpp>
#include <vector>

#include "common.hpp"

namespace
{
using namespace example;
using namespace example::hyperbola;
using namespace tax::ode::methods;

constexpr int P = 6;
constexpr int M = kM;
constexpr int D = 6;
constexpr int kNPerEdge = 24;
constexpr int kNMonte = 6000;

using TE = tax::TE< P, M >;
using DAState = tax::la::VecNT< D, TE >;
using Vec6 = tax::la::VecNT< 6, double >;
using Stepper = tax::ode::StepperType< Verner89, DAState >;
using G = decltype( radialVelocityOfCenter() );
using ExtraEvt = std::vector< std::shared_ptr< tax::ode::Event< DAState, double > > >;

// A fresh terminal closest-approach event (the driver deep-clones it per leaf).
ExtraEvt periapsisEvent()
{
    ExtraEvt e;
    e.push_back( std::make_shared< tax::ode::RootFindingEvent< DAState, double, G > >(
        radialVelocityOfCenter(), tax::ode::Direction::Increasing, "closest_approach",
        /*terminal=*/true ) );
    return e;
}

// B-plane outline of one leaf's closest-approach flow map, projected on the
// fixed nominal frame.
Polygon leafBPlane( const DAState& payload, const Frame& fr,
                    const std::vector< std::array< double, 2 > >& boundary, int id, int depth )
{
    const auto bd = bData( payload );
    const TE BT = project( bd.Bx, bd.By, bd.Bz, fr.T );
    const TE BR = project( bd.Bx, bd.By, bd.Bz, fr.R );
    Polygon p;
    p.id = id;
    p.depth = depth;
    p.x.reserve( boundary.size() );
    p.y.reserve( boundary.size() );
    for ( const auto& ab : boundary )
    {
        const auto d = boundaryToBox( ab[0], ab[1] );
        p.x.push_back( BT.eval( d ) );
        p.y.push_back( BR.eval( d ) );
    }
    return p;
}

struct DomainResult
{
    std::string name;
    int n_leaves = 0;
    std::vector< Polygon > leaves;
};

template < class Domain >
DomainResult scoreDomain( const std::string& name, const Domain& domain,
                          const tax::ads::TruncationCriterion& crit,
                          const tax::ode::IntegratorConfig< double >& cfg, const Frame& fr,
                          const std::vector< std::array< double, 2 > >& boundary, double t_final,
                          int threads )
{
    tax::ads::AdsDriver< Stepper, tax::ads::TruncationCriterion, Domain > driver{
        crit, cfg, periapsisEvent(), threads };
    auto sol = driver.run( rhs(), domain, icCenter(), 0.0, t_final );

    DomainResult r;
    r.name = name;
    const auto& tree = sol.tree();
    r.n_leaves = static_cast< int >( tree.done().size() );
    int id = 0;
    for ( int li : tree.done() )
    {
        const auto& leaf = tree.leaf( li );
        r.leaves.push_back( leafBPlane( leaf.payload, fr, boundary, id++, leaf.depth ) );
    }
    return r;
}
}  // namespace

int main()
{
    const double t_final = tFinal();
    const Vec6 center = icCenter();
    const Frame frame = nominalFrame( center );
    const auto boundary = unitSquareBoundary( kNPerEdge );

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    cfg.save_steps = false;

    const tax::ads::TruncationCriterion criterion{ /*tol=*/1e-4, /*maxDepth=*/7 };

    // ---- Monte-Carlo truth cloud (samples of the oriented zonotope set) ------
    const auto zono = icZonotope();
    tax::ode::IntegratorConfig< double > mcCfg = cfg;
    std::mt19937 rng( 4242u );
    std::uniform_real_distribution< double > unit( -1.0, 1.0 );
    std::vector< double > mc_bt, mc_br;
    mc_bt.reserve( kNMonte );
    mc_br.reserve( kNMonte );
    for ( int s = 0; s < kNMonte; ++s )
    {
        const double xi0 = unit( rng ), xi1 = unit( rng );
        Vec6 ic = center;
        ic( 0 ) += zono.generators( 0, 0 ) * xi0 + zono.generators( 0, 1 ) * xi1;
        ic( 1 ) += zono.generators( 1, 0 ) * xi0 + zono.generators( 1, 1 ) * xi1;
        auto msol = tax::ode::propagate( Verner89{}, rhs(), ic, 0.0, t_final, mcCfg );
        const auto mbd = bData( msol.x.back() );
        mc_bt.push_back( project( mbd.Bx, mbd.By, mbd.Bz, frame.T ) );
        mc_br.push_back( project( mbd.Bx, mbd.By, mbd.Bz, frame.R ) );
    }

    // ---- ADS event tiling for the box vs the oriented zonotope --------------
    Stopwatch clock;
    std::vector< DomainResult > doms;
    doms.push_back( scoreDomain( "bounding box", icBox(), criterion, cfg, frame, boundary, t_final,
                                 adsThreads() ) );
    doms.push_back( scoreDomain( "oriented zonotope", icZonotope(), criterion, cfg, frame, boundary,
                                 t_final, adsThreads() ) );
    const double elapsed_ms = clock.ms();

    const auto bd0 = bData( center );
    const double bt0 = project( bd0.Bx, bd0.By, bd0.Bz, frame.T );
    const double br0 = project( bd0.Bx, bd0.By, bd0.Bz, frame.R );

    // ---- Output --------------------------------------------------------------
    std::ofstream out( "hyperbola_bplane_ads.json" );
    out << std::setprecision( 12 );
    out << "{\n";
    out << "  \"problem\": \"hyperbola_bplane_ads\",\n";
    out << "  \"t_final\": " << t_final << ",\n";
    out << "  \"tol\": " << criterion.tol << ", \"max_depth\": " << criterion.maxDepth << ",\n";
    out << "  \"b_center\": [" << bt0 << ", " << br0 << "],\n";
    out << "  \"mc\": { \"bt\": ";
    writeJsonArray( out, mc_bt );
    out << ", \"br\": ";
    writeJsonArray( out, mc_br );
    out << " },\n";
    out << "  \"domains\": [\n";
    for ( std::size_t i = 0; i < doms.size(); ++i )
    {
        const auto& d = doms[i];
        out << "    { \"name\": \"" << d.name << "\", \"n_leaves\": " << d.n_leaves
            << ", \"leaves\": [";
        for ( std::size_t l = 0; l < d.leaves.size(); ++l )
        {
            out << "{ \"depth\": " << d.leaves[l].depth << ", \"x\": ";
            writeJsonArray( out, d.leaves[l].x );
            out << ", \"y\": ";
            writeJsonArray( out, d.leaves[l].y );
            out << " }" << ( l + 1 < d.leaves.size() ? ", " : "" );
        }
        out << "] }" << ( i + 1 < doms.size() ? ",\n" : "\n" );
    }
    out << "  ]\n}\n";

    std::string counts;
    for ( const auto& d : doms )
        counts += ( counts.empty() ? "" : ", " ) + d.name + "=" + std::to_string( d.n_leaves );

    printBanner( "hyperbola/bplane_ads — event surface tiled by ADS (box vs zonotope)",
                 { { "P, M", std::to_string( P ) + ", " + std::to_string( M ) },
                   { "criterion", "truncation, tol=1e-4, depth<=7" },
                   { "leaves at event", counts },
                   { "mc samples", std::to_string( kNMonte ) },
                   { "elapsed", std::to_string( elapsed_ms ) + " ms" },
                   { "output", "hyperbola_bplane_ads.json" } } );
    return 0;
}
