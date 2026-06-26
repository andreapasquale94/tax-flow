// =============================================================================
// examples/two_body/cpz.cpp
//
// Constrained polynomial zonotope on the two-body problem: an uncertainty box
// in (y, v_y) restricted to a fixed-energy (fixed semi-major-axis) level set.
//
// The valid initial conditions are not the whole box but the curve
//   { (y, v_y) ∈ box : E(y, v_y) = E₀ },   E = ½v² − 1/r,
// a one-dimensional sub-manifold. We express E as a DA polynomial in the box
// factors, so the constraint g(ξ) = E(ξ) − E₀ is a polynomial and the set is a
// constrained polynomial zonotope (tax::ads::cpz).
//
// Two things are shown:
//   1. Dimensional collapse — subdividing the box while pruning leaves whose
//      energy range excludes E₀ keeps only the band straddling the curve. The
//      unpruned count grows like 2^depth (area); the feasible count like
//      2^(depth/2) (length).
//   2. Validation — propagate the feasible leaves to t = T/2 and overlay the
//      constrained image on 10000 Monte-Carlo samples taken ON the true energy
//      curve. The CPZ curve reproduces them; the leaf count is far below the
//      unconstrained 2-D ADS tiling of the same box.
//
// Run:    ./two_body_cpz
// Writes: cpz.json   (plot with plot_two_body_cpz.py)
// =============================================================================

#include <cmath>
#include <random>
#include <string>
#include <tax/ads.hpp>
#include <tax/ode.hpp>
#include <utility>
#include <vector>

#include "common.hpp"

namespace
{
using namespace example;
using namespace example::two_body;
using namespace tax::ode::methods;

constexpr int P = 8;
constexpr int M = 4;
constexpr int D = 4;
constexpr int kAy = 1;  // y
constexpr int kAv = 3;  // v_y

using TE = tax::TE< P, M >;
using DAState = tax::la::VecNT< D, TE >;
using Vec4 = tax::la::VecNT< 4, double >;
using CPZ = tax::ads::ConstrainedPolyZonotope< double, P, M, tax::storage::Dense, D >;

// Specific orbital energy of a state (scalar or DA), E = ½(vx²+vy²) − 1/r.
template < class S >
auto energyOf( const S& s )
{
    const auto r2 = s( 0 ) * s( 0 ) + s( 1 ) * s( 1 );
    return 0.5 * ( s( 2 ) * s( 2 ) + s( 3 ) * s( 3 ) ) - 1.0 / sqrt( r2 );
}

double nominalEnergy()
{
    const Vec4 c = icCenter();
    return 0.5 * ( c( 2 ) * c( 2 ) + c( 3 ) * c( 3 ) ) -
           1.0 / std::sqrt( c( 0 ) * c( 0 ) + c( 1 ) * c( 1 ) );
}

// v_y on the energy curve E(y, v_y) = E₀ (positive branch), or NaN if off it.
double vyOnCurve( double y, double E0 )
{
    const double r = std::sqrt( icCenter()( 0 ) * icCenter()( 0 ) + y * y );
    const double arg = 2.0 * ( E0 + 1.0 / r );
    return arg > 0.0 ? std::sqrt( arg ) : std::nan( "" );
}

// Linear coefficient ∂g/∂ξ_dim of a constraint at the origin.
double linCoeff( const TE& g, int dim )
{
    tax::MultiIndex< M > e{};
    e[static_cast< std::size_t >( dim )] = 1;
    return g[tax::flatIndex< M >( e )];
}

// Trace the curve g(ξ)=0 across a leaf (linear near its small box) and map it
// through the leaf flow map → an (x, y) polyline of the constrained image.
Polygon traceConstraint( const DAState& payload, const TE& g )
{
    Polygon p;
    const double g0 = g[0];
    const double Jy = linCoeff( g, kAy );
    const double Jv = linCoeff( g, kAv );
    const bool along_y = std::abs( Jv ) >= std::abs( Jy );  // solve for the steeper axis
    constexpr int kS = 9;
    for ( int i = 0; i < kS; ++i )
    {
        const double s = -1.0 + 2.0 * i / ( kS - 1 );
        double a = 0.0, b = 0.0;
        if ( along_y )
        {
            if ( std::abs( Jv ) < 1e-30 ) break;
            a = s;
            b = -( g0 + Jy * a ) / Jv;
        } else
        {
            if ( std::abs( Jy ) < 1e-30 ) break;
            b = s;
            a = -( g0 + Jv * b ) / Jy;
        }
        if ( std::abs( a ) > 1.0 || std::abs( b ) > 1.0 ) continue;
        Vec4 d = Vec4::Zero();
        d( kAy ) = a;
        d( kAv ) = b;
        p.x.push_back( payload( 0 ).eval( d ) );
        p.y.push_back( payload( 1 ).eval( d ) );
    }
    return p;
}
}  // namespace

int main()
{
    const double t_final = 0.5 * kPeriod;
    const double E0 = nominalEnergy();

    // ---- The uncertainty box and its energy constraint -----------------------
    Vec4 hw = Vec4::Zero();
    hw( kAy ) = 0.15;
    hw( kAv ) = 0.10;
    const tax::ads::Box< double, M > box{ icCenter(), hw };

    DAState id = tax::ads::create< P, M >( box, icCenter() );
    CPZ root;
    root.value = id;
    root.constraints.push_back( energyOf( id ) - E0 );  // g(ξ) = E(ξ) − E₀

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    cfg.save_steps = true;

    // ---- 1. Dimensional collapse: pruned subdivision, level by level ---------
    constexpr int kMaxDepth = 16;
    constexpr int kBandDepth = 8;  // shallower level whose boxes are drawable
    struct Node
    {
        tax::ads::Box< double, M > box;
        CPZ cpz;
    };
    std::vector< Node > level;
    level.push_back( Node{ box, root } );
    std::vector< double > depth_axis, full_counts, feas_counts;
    std::vector< Node > band;  // feasible leaves at kBandDepth (for the figure)
    for ( int depth = 0;; ++depth )
    {
        std::vector< Node > feas;
        for ( auto& n : level )
            if ( tax::ads::feasible( n.cpz ) ) feas.push_back( std::move( n ) );
        depth_axis.push_back( depth );
        full_counts.push_back( static_cast< double >( std::size_t{ 1 } << depth ) );
        feas_counts.push_back( static_cast< double >( feas.size() ) );
        if ( depth == kBandDepth )
            for ( const auto& n : feas ) band.push_back( n );
        if ( depth == kMaxDepth ) break;
        const int dim = ( depth % 2 == 0 ) ? kAy : kAv;
        std::vector< Node > next;
        next.reserve( feas.size() * 2 );
        for ( auto& n : feas )
        {
            auto bpr = n.box.split( dim );
            auto cpr = tax::ads::split( n.cpz, n.box, dim );
            next.push_back( Node{ bpr.first, std::move( cpr.first ) } );
            next.push_back( Node{ bpr.second, std::move( cpr.second ) } );
        }
        level = std::move( next );
    }

    // ---- 2a. Unconstrained 2-D ADS of the box (for contrast) -----------------
    const auto boundary = unitSquareBoundary( 20 );
    auto tree = tax::ads::propagate< P >( Verner89{}, tax::ads::TruncationCriterion{ 1e-6, 8 },
                                          rhs(), box, icCenter(), 0.0, t_final, cfg, adsThreads() );
    std::vector< Polygon > blob;
    for ( int li : tree.done() )
        blob.push_back( evalPolygon( tree.leaf( li ).payload, boundary, boundaryToBox, 0,
                                     tree.leaf( li ).depth ) );

    // ---- 2b. Constrained image: feasible ADS leaves, constraint traced -------
    std::vector< Polygon > cpz_curves;
    std::vector< int > feasible_leaves;
    for ( int li : tree.done() )
    {
        const auto& leaf = tree.leaf( li );
        const TE g_leaf = tax::ads::restrictConstraint( root.constraints[0], box, leaf.box );
        if ( !tax::ads::constraintFeasible( g_leaf, /*tol=*/0.0 ) ) continue;
        feasible_leaves.push_back( li );
        cpz_curves.push_back( traceConstraint( leaf.payload, g_leaf ) );
    }

    // ---- 3. Monte-Carlo on the true energy curve -----------------------------
    std::mt19937 rng( 99u );
    std::uniform_real_distribution< double > uy( icCenter()( kAy ) - hw( kAy ),
                                                 icCenter()( kAy ) + hw( kAy ) );
    std::vector< double > mc_x, mc_y;
    double sq_err = 0.0, max_err = 0.0;
    int scored = 0;
    auto fastCfg = cfg;
    fastCfg.save_steps = false;
    for ( int s = 0; s < 10000; ++s )
    {
        const double y = uy( rng );
        const double vy = vyOnCurve( y, E0 );
        if ( std::isnan( vy ) || std::abs( vy - icCenter()( kAv ) ) > hw( kAv ) ) continue;
        Vec4 ic = icCenter();
        ic( kAy ) = y;
        ic( kAv ) = vy;
        auto sol = tax::ode::propagate( Verner89{}, rhs(), ic, 0.0, t_final, fastCfg );
        const double tx = sol.x.back()( 0 ), ty = sol.x.back()( 1 );
        mc_x.push_back( tx );
        mc_y.push_back( ty );

        // Score: locate the feasible leaf holding (y, vy), evaluate its map.
        for ( int li : feasible_leaves )
        {
            const auto& lf = tree.leaf( li );
            const double a = ( y - lf.box.center( kAy ) ) / lf.box.halfWidth( kAy );
            const double b = ( vy - lf.box.center( kAv ) ) / lf.box.halfWidth( kAv );
            if ( std::abs( a ) > 1.0 + 1e-9 || std::abs( b ) > 1.0 + 1e-9 ) continue;
            Vec4 d = Vec4::Zero();
            d( kAy ) = a;
            d( kAv ) = b;
            const double e =
                std::hypot( lf.payload( 0 ).eval( d ) - tx, lf.payload( 1 ).eval( d ) - ty );
            sq_err += e * e;
            max_err = std::max( max_err, e );
            ++scored;
            break;
        }
    }
    const double rms = scored > 0 ? std::sqrt( sq_err / scored ) : 0.0;

    // ---- Reference orbit + the true energy curve in (y, v_y) -----------------
    auto ref_sol = tax::ode::propagate( Taylor< 16 >{}, rhs(), icCenter(), 0.0, t_final, cfg );
    const auto reference = sampleOrbit( ref_sol, {}, D );
    std::vector< double > curve_y, curve_v;
    for ( double yy :
          example::linspace( icCenter()( kAy ) - hw( kAy ), icCenter()( kAy ) + hw( kAy ), 120 ) )
    {
        const double vv = vyOnCurve( yy, E0 );
        if ( !std::isnan( vv ) )
        {
            curve_y.push_back( yy );
            curve_v.push_back( vv );
        }
    }

    // ---- Output --------------------------------------------------------------
    std::ofstream out( "cpz.json" );
    out << std::setprecision( 10 ) << "{\n";
    out << "  \"t_final\": " << t_final << ", \"E0\": " << E0 << ",\n";
    out << "  \"center_yv\": [" << icCenter()( kAy ) << ", " << icCenter()( kAv ) << "],\n";
    out << "  \"hw_yv\": [" << hw( kAy ) << ", " << hw( kAv ) << "],\n";
    out << "  \"n_full_ads\": " << tree.done().size() << ", \"n_cpz\": " << feasible_leaves.size()
        << ",\n";
    out << "  \"rms\": " << rms << ", \"max_err\": " << max_err << ",\n";

    out << "  \"convergence\": { \"depth\": ";
    writeJsonArray( out, depth_axis );
    out << ", \"full\": ";
    writeJsonArray( out, full_counts );
    out << ", \"feasible\": ";
    writeJsonArray( out, feas_counts );
    out << " },\n";

    out << "  \"curve_yv\": { \"y\": ";
    writeJsonArray( out, curve_y );
    out << ", \"v\": ";
    writeJsonArray( out, curve_v );
    out << " },\n";

    out << "  \"band\": [";
    for ( std::size_t i = 0; i < band.size(); ++i )
    {
        const auto& b = band[i].box;
        out << "[" << b.center( kAy ) << ", " << b.center( kAv ) << ", " << b.halfWidth( kAy )
            << ", " << b.halfWidth( kAv ) << "]" << ( i + 1 < band.size() ? ", " : "" );
    }
    out << "],\n";

    out << "  \"reference_orbit\": { \"x0\": ";
    writeJsonArray( out, reference.cols[0] );
    out << ", \"x1\": ";
    writeJsonArray( out, reference.cols[1] );
    out << " },\n";

    out << "  \"mc\": { \"x\": ";
    writeJsonArray( out, mc_x );
    out << ", \"y\": ";
    writeJsonArray( out, mc_y );
    out << " },\n";

    auto writePolys = [&]( const std::vector< Polygon >& ps ) {
        out << "[";
        for ( std::size_t i = 0; i < ps.size(); ++i )
        {
            out << "{ \"x\": ";
            writeJsonArray( out, ps[i].x );
            out << ", \"y\": ";
            writeJsonArray( out, ps[i].y );
            out << " }" << ( i + 1 < ps.size() ? ", " : "" );
        }
        out << "]";
    };
    out << "  \"blob\": ";
    writePolys( blob );
    out << ",\n  \"cpz_curves\": ";
    writePolys( cpz_curves );
    out << "\n}\n";

    printBanner(
        "two_body/cpz — fixed-energy constrained polynomial zonotope",
        { { "constraint", "E(y,vy) = E0 (fixed semi-major axis)" },
          { "E0", jsonNumber( E0 ) },
          { "box hw (y,vy)", std::to_string( hw( kAy ) ) + ", " + std::to_string( hw( kAv ) ) },
          { "subdiv depth", std::to_string( kMaxDepth ) + " -> full " +
                                std::to_string( static_cast< long >( full_counts.back() ) ) +
                                " vs feasible " +
                                std::to_string( static_cast< long >( feas_counts.back() ) ) },
          { "ADS leaves", std::to_string( tree.done().size() ) + " (2-D box) vs " +
                              std::to_string( feasible_leaves.size() ) + " (on constraint)" },
          { "MC RMS / max", jsonNumber( rms ) + " / " + jsonNumber( max_err ) },
          { "output", "cpz.json" } } );
    return 0;
}
