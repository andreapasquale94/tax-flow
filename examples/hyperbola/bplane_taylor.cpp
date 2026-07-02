// =============================================================================
// examples/hyperbola/bplane_taylor.cpp
//
// Stage 1 — one Taylor flow polynomial, no domain splitting.
//
// This is the "simplest case" the tutorial builds on: a single DA-valued state
// (an Eigen vector of TaylorExpansions seeded on the IC box) is integrated once
// through the flyby. Along the way we answer the four questions:
//
//   1. Taylor expansion of the EVENT.  Closest approach is r . v = 0. A scalar
//      RootFindingEvent locates the *center* orbit's periapsis time t*. Because
//      for xi != 0 the true closest approach happens at a slightly different
//      time, we recover the time-of-event surface by one DA-Newton step:
//         dt(xi) = - g / g'  with  g = r.v,  g' = v^2 - 1/r   (a DA division),
//      then evaluate the flow on that surface. The residual r.v drops by orders
//      of magnitude, confirming the surface was found.
//
//   2. Taylor expansion of the STATE at the event.  The event record already
//      carries the DA state at t*; the DA-Newton step shifts it onto the true
//      closest-approach surface X_ca(xi) = X(t*, xi) + f(X) . dt(xi).
//
//   3. B-plane representation.  From X_ca we build the impact-parameter vector B
//      and project it onto the fixed nominal (B.T, B.R) frame -> two DA
//      polynomials. For a flyby the impact parameter is a nearly *linear*
//      function of the IC, so a first-order zonotope already encloses the
//      B-plane image tightly (we report the linear fit and the cloud).
//
//   4. Enclosures.  The strong nonlinearity is in configuration space near
//      periapsis, so the box / zonotope / polynomial-zonotope hierarchy is shown
//      on the closest-approach POSITION (x, y) of X_ca, validated against a
//      Monte-Carlo cloud (mirrors examples/zonotope/representations.cpp).
//
// Run:    ./hyperbola_bplane_taylor
// Writes: hyperbola_bplane.json  (plot with examples/plot/plot_hyperbola_bplane.py)
// =============================================================================

#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <string>
#include <tax/ads/da_state.hpp>
#include <tax/ode.hpp>
#include <tax/ode/events/root_finding_event.hpp>
#include <vector>

#include "common.hpp"

namespace
{
using namespace example;
using namespace example::hyperbola;
using namespace tax::ode::methods;

constexpr int P = 6;       // DA truncation order
constexpr int M = kM;      // number of DA factors (= 2)
constexpr int D = 6;       // physical state dimension
constexpr int kNPerEdge = 40;
constexpr int kNMonte = 5000;

using TE = tax::TE< P, M >;
using DAState = tax::la::VecNT< D, TE >;
using Vec6 = tax::la::VecNT< 6, double >;

// Degree-1 coefficient of factor xi_j in a TE.
double lin( const TE& f, int j )
{
    tax::MultiIndex< M > e{};
    e[static_cast< std::size_t >( j )] = 1;
    return f[tax::flatIndex< M >( e )];
}

// Interval hull of two DA components over xi in [-1,1]^M: c0 +/- sum |coeff|.
std::array< double, 4 > boxHull( const TE& u, const TE& v )
{
    constexpr std::size_t Nc = tax::numMonomials( P, M );
    auto radius = [&]( const TE& f ) {
        double r = 0.0;
        for ( std::size_t k = 1; k < Nc; ++k ) r += std::abs( f[k] );
        return r;
    };
    return { u[0] - radius( u ), u[0] + radius( u ), v[0] - radius( v ), v[0] + radius( v ) };
}

// Scalar closest-approach (terminal) event on a plain double state.
auto scalarPeriapsis()
{
    return []( const auto& x, double ) -> double {
        return x( 0 ) * x( 3 ) + x( 1 ) * x( 4 ) + x( 2 ) * x( 5 );
    };
}
}  // namespace

int main()
{
    const double t_final = tFinal();
    const Vec6 center = icCenter();
    const auto box = icBox();
    const Frame frame = nominalFrame( center );

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    cfg.save_steps = true;

    // ---- One DA propagation with a closest-approach event -------------------
    auto x0_da = tax::ads::create< P, M >( box, center );
    using G = decltype( radialVelocityOfCenter() );
    std::vector< std::shared_ptr< tax::ode::Event< DAState, double > > > events;
    events.push_back( std::make_shared< tax::ode::RootFindingEvent< DAState, double, G > >(
        radialVelocityOfCenter(), tax::ode::Direction::Increasing, "closest_approach", false ) );

    Stopwatch clock;
    auto sol = tax::ode::propagate( Verner89{}, rhs(), x0_da, 0.0, t_final, cfg, events );

    double t_event = 0.0;
    DAState X_star = x0_da;
    for ( const auto& e : sol.events )
        if ( e.label == "closest_approach" )
        {
            t_event = e.t;
            X_star = e.x;
        }

    // ---- Time-of-event surface: one DA-Newton step on r . v = 0 -------------
    auto rdotv = []( const DAState& s ) {
        return s( 0 ) * s( 3 ) + s( 1 ) * s( 4 ) + s( 2 ) * s( 5 );
    };
    const TE g = rdotv( X_star );
    const TE r_star = sqrt( X_star( 0 ) * X_star( 0 ) + X_star( 1 ) * X_star( 1 ) +
                            X_star( 2 ) * X_star( 2 ) );
    const TE v2 = X_star( 3 ) * X_star( 3 ) + X_star( 4 ) * X_star( 4 ) + X_star( 5 ) * X_star( 5 );
    const TE gdot = v2 - TE( 1.0 ) / r_star;  // d/dt (r.v) = v^2 - 1/r
    const TE dt = -g / gdot;                   // dt(xi), constant term ~ 0

    const DAState f = rhs()( X_star, t_event );
    DAState X_ca = X_star;
    for ( int i = 0; i < D; ++i ) X_ca( i ) = X_star( i ) + f( i ) * dt;

    const double rv_before = std::abs( g[0] );
    const double rv_after = std::abs( rdotv( X_ca )[0] );

    // ---- Event quantities (Taylor expansions) -------------------------------
    const auto bd = bData( X_ca );
    const double elapsed_ms = clock.ms();

    const TE r_ca = bd.rmag;  // closest-approach distance polynomial
    const TE BT = project( bd.Bx, bd.By, bd.Bz, frame.T );
    const TE BR = project( bd.Bx, bd.By, bd.Bz, frame.R );

    // B-plane first-order (zonotope) fit — for a flyby this is nearly exact.
    const std::array< double, 2 > b_c{ BT[0], BR[0] };
    const std::array< double, 4 > b_J{ lin( BT, 0 ), lin( BT, 1 ), lin( BR, 0 ), lin( BR, 1 ) };

    // Closest-approach POSITION (x, y): box / zonotope / poly-zonotope hierarchy.
    const TE& cx = X_ca( 0 );
    const TE& cy = X_ca( 1 );
    const auto pos_hull = boxHull( cx, cy );
    const std::array< double, 2 > pos_c{ cx[0], cy[0] };
    const std::array< double, 4 > pos_J{ lin( cx, 0 ), lin( cx, 1 ), lin( cy, 0 ), lin( cy, 1 ) };

    const auto boundary = unitSquareBoundary( kNPerEdge );
    std::vector< double > pz_x, pz_y;
    pz_x.reserve( boundary.size() );
    pz_y.reserve( boundary.size() );
    for ( const auto& ab : boundary )
    {
        const auto d = boundaryToBox( ab[0], ab[1] );
        pz_x.push_back( cx.eval( d ) );
        pz_y.push_back( cy.eval( d ) );
    }

    // ---- Monte-Carlo truth: closest-approach state of each sampled IC -------
    using SG = decltype( scalarPeriapsis() );
    auto mcEvents = []() {
        std::vector< std::shared_ptr< tax::ode::Event< Vec6, double > > > ev;
        ev.push_back( std::make_shared< tax::ode::RootFindingEvent< Vec6, double, SG > >(
            scalarPeriapsis(), tax::ode::Direction::Increasing, "ca", /*terminal=*/true ) );
        return ev;
    };
    tax::ode::IntegratorConfig< double > mcCfg;
    mcCfg.abstol = mcCfg.reltol = 1e-12;
    mcCfg.save_steps = false;

    std::mt19937 rng( 2025u );
    std::uniform_real_distribution< double > unit( -1.0, 1.0 );
    std::vector< double > mc_bt, mc_br, mc_x, mc_y;
    for ( int s = 0; s < kNMonte; ++s )
    {
        Vec6 ic = center;
        ic( 0 ) += box.halfWidth( 0 ) * unit( rng );
        ic( 1 ) += box.halfWidth( 1 ) * unit( rng );
        auto msol = tax::ode::propagate( Verner89{}, rhs(), ic, 0.0, t_final, mcCfg, mcEvents() );
        const Vec6 xca = msol.x.back();  // terminal event -> closest-approach state
        const auto mbd = bData( xca );
        mc_bt.push_back( project( mbd.Bx, mbd.By, mbd.Bz, frame.T ) );
        mc_br.push_back( project( mbd.Bx, mbd.By, mbd.Bz, frame.R ) );
        mc_x.push_back( xca( 0 ) );
        mc_y.push_back( xca( 1 ) );
    }

    // ---- Output --------------------------------------------------------------
    std::ofstream out( "hyperbola_bplane.json" );
    out << std::setprecision( 12 );
    out << "{\n";
    out << "  \"problem\": \"hyperbola_bplane\",\n";
    out << "  \"P\": " << P << ", \"M\": " << M << ", \"D\": " << D << ",\n";
    out << "  \"t_event\": " << t_event << ", \"t_peri_analytic\": " << tPeri()
        << ", \"t_final\": " << t_final << ",\n";
    out << "  \"ecc\": " << bd.ecc[0] << ", \"vinf\": " << bd.vinf[0] << ",\n";
    out << "  \"r_ca_const\": " << r_ca[0] << ", \"rv_before\": " << rv_before
        << ", \"rv_after\": " << rv_after << ",\n";
    out << "  \"bplane\": { \"mc\": { \"bt\": ";
    writeJsonArray( out, mc_bt );
    out << ", \"br\": ";
    writeJsonArray( out, mc_br );
    out << " }, \"c\": [" << b_c[0] << ", " << b_c[1] << "], \"J\": [" << b_J[0] << ", " << b_J[1]
        << ", " << b_J[2] << ", " << b_J[3] << "] },\n";
    out << "  \"capos\": { \"center\": [" << pos_c[0] << ", " << pos_c[1] << "], \"mc\": { \"x\": ";
    writeJsonArray( out, mc_x );
    out << ", \"y\": ";
    writeJsonArray( out, mc_y );
    out << " }, \"box\": [" << pos_hull[0] << ", " << pos_hull[1] << ", " << pos_hull[2] << ", "
        << pos_hull[3] << "], \"zonotope\": { \"c\": [" << pos_c[0] << ", " << pos_c[1]
        << "], \"J\": [" << pos_J[0] << ", " << pos_J[1] << ", " << pos_J[2] << ", " << pos_J[3]
        << "] }, \"poly_zonotope\": { \"x\": ";
    writeJsonArray( out, pz_x );
    out << ", \"y\": ";
    writeJsonArray( out, pz_y );
    out << " } }\n}\n";

    printBanner(
        "hyperbola/bplane_taylor — one Taylor flow map, closest approach & B-plane",
        { { "P, M", std::to_string( P ) + ", " + std::to_string( M ) },
          { "ecc, vinf", std::to_string( bd.ecc[0] ) + ", " + std::to_string( bd.vinf[0] ) },
          { "t_event vs analytic", std::to_string( t_event ) + " vs " + std::to_string( tPeri() ) },
          { "r_ca (const term)", std::to_string( r_ca[0] ) },
          { "|r.v| before -> after", std::to_string( rv_before ) + " -> " +
                                         std::to_string( rv_after ) },
          { "B.T, B.R (center)", std::to_string( b_c[0] ) + ", " + std::to_string( b_c[1] ) },
          { "mc samples", std::to_string( kNMonte ) },
          { "elapsed", std::to_string( elapsed_ms ) + " ms" },
          { "output", "hyperbola_bplane.json" } } );
    return 0;
}
