// =============================================================================
// examples/two_body/poly_zonotope.cpp
//
// A *curved* initial-condition set propagated as a polynomial zonotope.
//
// The uncertainty is a 3σ Gaussian in two orbital parameters — true anomaly ν
// (where along the orbit) and eccentricity e (orbit shape). In Cartesian state
// that uncertainty is not an ellipse but a **bent** set (a banana), because the
// element→Cartesian map is nonlinear. Expanding that map as a DA gives a
// genuine polynomial-zonotope initial set, carried as a tax::ads::PolyZonotope
// domain and propagated through ADS.
//
// We compare against the **linear** picture — the tangent ellipse (first-order
// element→Cartesian map) propagated the same way — and validate both against
// 10000 Monte-Carlo samples drawn from the element-space Gaussian. The curved
// polynomial zonotope tracks the cloud; the linear ellipse misses the tails.
//
// Run:    ./two_body_poly_zonotope
// Writes: poly_zonotope.json   (plot with plot_two_body_poly_zonotope.py)
// =============================================================================

#include <cmath>
#include <random>
#include <string>
#include <tax/ads.hpp>
#include <tax/ode.hpp>
#include <vector>

#include "common.hpp"

namespace
{
using namespace example;
using namespace example::two_body;
using namespace tax::ode::methods;

constexpr int P = 8;
constexpr int M = 2;  // factors: (ν, e)
constexpr int D = 4;  // Cartesian state

using TE = tax::TE< P, M >;
using DAState = tax::la::VecNT< D, TE >;
using Vec4 = tax::la::VecNT< 4, double >;
using V2 = tax::la::VecNT< 2, double >;
using PZ = tax::ads::PolyZonotope< double, P, M, tax::storage::Dense, D >;

// 3σ half-widths of the (ν, e) Gaussian — ξ = ±1 maps to ±3σ.
constexpr double kSigNu = 0.05;        // 1σ true anomaly [rad]
constexpr double kSigE = 0.03;         // 1σ eccentricity
constexpr double kE0 = kEcc;           // nominal e = 0.5
constexpr double kNu3 = 3.0 * kSigNu;  // box half-width in ν
constexpr double kE3 = 3.0 * kSigE;    // box half-width in e

// Perifocal Cartesian state of a Kepler orbit (a = 1, μ = 1) at (ν, e), generic
// over scalar / DA arguments. Matches icCenter() at (ν, e) = (0, kE0).
template < class S >
tax::la::VecNT< 4, S > orbitState( const S& nu, const S& e )
{
    const S cn = cos( nu );
    const S sn = sin( nu );
    const S p = 1.0 - e * e;           // semi-latus rectum (a = 1)
    const S r = p / ( 1.0 + e * cn );  // radius
    const S isp = 1.0 / sqrt( p );     // sqrt(μ/p)
    tax::la::VecNT< 4, S > s;
    s( 0 ) = r * cn;
    s( 1 ) = r * sn;
    s( 2 ) = isp * ( -sn );
    s( 3 ) = isp * ( e + cn );
    return s;
}

// The curved initial map x0(ξ): orbit state at ν = kNu3·ξ0, e = kE0 + kE3·ξ1.
DAState curvedMap()
{
    const TE nu = kNu3 * TE::variable( 0.0, 0 );
    const TE e = kE0 + kE3 * TE::variable( 0.0, 1 );
    const auto s = orbitState( nu, e );
    DAState m;
    for ( int i = 0; i < D; ++i ) m( i ) = s( i );
    return m;
}

// Degree-1 (tangent) approximation of a curved map: keep centre + Jacobian·ξ.
DAState linearize( const PZ& z )
{
    const auto J = z.jacobian();  // D×M
    DAState m;
    for ( int i = 0; i < D; ++i )
    {
        TE c{};
        c[0] = z.center( i );
        for ( int j = 0; j < M; ++j )
        {
            tax::MultiIndex< M > e{};
            e[static_cast< std::size_t >( j )] = 1;
            c[tax::flatIndex< M >( e )] = J( i, j );
        }
        m( i ) = c;
    }
    return m;
}

// (x, y) boundary image of a map over the unit-square factor boundary.
std::array< double, 2 > toBox2( double a, double b ) { return { a, b }; }

// Best-fit leaf lookup + flow-map evaluation for a physical IC, returning the
// predicted (x, y). Uses each leaf's linear inverse (leaves are small).
template < class Tree >
V2 predictXY( const Tree& tree, const Vec4& ic )
{
    double best = 1e30;
    V2 out{ 0, 0 };
    for ( int li : tree.done() )
    {
        const auto& leaf = tree.leaf( li );
        const auto J = leaf.box.jacobian();
        const V2 xi = J.colPivHouseholderQr().solve( Vec4{ ic - leaf.box.center } );
        const double inf = xi.cwiseAbs().maxCoeff();
        if ( inf < best )
        {
            best = inf;
            out = V2{ leaf.payload( 0 ).eval( xi ), leaf.payload( 1 ).eval( xi ) };
        }
    }
    return out;
}

Polygon mapBoundary( const DAState& m, const std::vector< std::array< double, 2 > >& bnd )
{
    Polygon p;
    for ( const auto& ab : bnd )
    {
        const V2 d{ ab[0], ab[1] };
        p.x.push_back( m( 0 ).eval( d ) );
        p.y.push_back( m( 1 ).eval( d ) );
    }
    return p;
}
}  // namespace

int main()
{
    const double t_final = 0.5 * kPeriod;
    const auto boundary = unitSquareBoundary( 48 );

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    cfg.save_steps = true;

    const PZ pz_curved = PZ::fromMap( curvedMap() );
    const PZ pz_linear = PZ::fromMap( linearize( pz_curved ) );

    // Initial sets in (x, y): the bent polynomial zonotope vs its tangent.
    const Polygon ic_curved = mapBoundary( pz_curved.map, boundary );
    const Polygon ic_linear = mapBoundary( pz_linear.map, boundary );

    const tax::ads::TruncationCriterion crit{ /*tol=*/1e-6, /*maxDepth=*/8 };
    auto tree_curved = tax::ads::propagate< P >( Verner89{}, crit, rhs(), pz_curved, icCenter(),
                                                 0.0, t_final, cfg, adsThreads() );
    auto tree_linear = tax::ads::propagate< P >( Verner89{}, crit, rhs(), pz_linear, icCenter(),
                                                 0.0, t_final, cfg, adsThreads() );

    auto tiling = [&]( auto& tree ) {
        std::vector< Polygon > ps;
        for ( int li : tree.done() )
            ps.push_back( evalPolygon( tree.leaf( li ).payload, boundary, toBox2, 0,
                                       tree.leaf( li ).depth ) );
        return ps;
    };
    const auto leaves_curved = tiling( tree_curved );
    const auto leaves_linear = tiling( tree_linear );

    // ---- Monte Carlo: 10000 draws from the (ν, e) Gaussian, within 3σ --------
    std::mt19937 rng( 2025u );
    std::normal_distribution< double > gauss( 0.0, 1.0 );
    auto fastCfg = cfg;
    fastCfg.save_steps = false;
    std::vector< double > mc0x, mc0y, mcx, mcy;
    double sq_c = 0.0, sq_l = 0.0, max_c = 0.0, max_l = 0.0;
    int n = 0;
    while ( n < 10000 )
    {
        const double zn = gauss( rng ), ze = gauss( rng );
        if ( std::abs( zn ) > 3.0 || std::abs( ze ) > 3.0 ) continue;  // inside the 3σ box
        const double nu = kSigNu * zn;
        const double e = kE0 + kSigE * ze;
        const Vec4 ic = orbitState( nu, e );
        mc0x.push_back( ic( 0 ) );
        mc0y.push_back( ic( 1 ) );
        auto sol = tax::ode::propagate( Verner89{}, rhs(), ic, 0.0, t_final, fastCfg );
        const V2 truth{ sol.x.back()( 0 ), sol.x.back()( 1 ) };
        mcx.push_back( truth( 0 ) );
        mcy.push_back( truth( 1 ) );

        const V2 pc = predictXY( tree_curved, ic );
        const V2 pl = predictXY( tree_linear, ic );
        const double ec = std::hypot( pc( 0 ) - truth( 0 ), pc( 1 ) - truth( 1 ) );
        const double el = std::hypot( pl( 0 ) - truth( 0 ), pl( 1 ) - truth( 1 ) );
        sq_c += ec * ec;
        sq_l += el * el;
        max_c = std::max( max_c, ec );
        max_l = std::max( max_l, el );
        ++n;
    }
    const double rms_c = std::sqrt( sq_c / n );
    const double rms_l = std::sqrt( sq_l / n );

    auto ref_sol = tax::ode::propagate( Taylor< 16 >{}, rhs(), icCenter(), 0.0, t_final, cfg );
    const auto reference = sampleOrbit( ref_sol, {}, D );

    // ---- Output --------------------------------------------------------------
    std::ofstream out( "poly_zonotope.json" );
    out << std::setprecision( 10 ) << "{\n";
    out << "  \"t_final\": " << t_final << ",\n";
    out << "  \"sigma\": { \"nu\": " << kSigNu << ", \"e\": " << kSigE << " },\n";
    out << "  \"n_curved\": " << tree_curved.done().size()
        << ", \"n_linear\": " << tree_linear.done().size() << ",\n";
    out << "  \"rms_curved\": " << rms_c << ", \"rms_linear\": " << rms_l << ",\n";
    out << "  \"max_curved\": " << max_c << ", \"max_linear\": " << max_l << ",\n";

    auto poly = [&]( const Polygon& p ) {
        out << "{ \"x\": ";
        writeJsonArray( out, p.x );
        out << ", \"y\": ";
        writeJsonArray( out, p.y );
        out << " }";
    };
    auto polys = [&]( const std::vector< Polygon >& ps ) {
        out << "[";
        for ( std::size_t i = 0; i < ps.size(); ++i )
        {
            poly( ps[i] );
            out << ( i + 1 < ps.size() ? ", " : "" );
        }
        out << "]";
    };
    auto cloud = [&]( const std::vector< double >& xs, const std::vector< double >& ys ) {
        out << "{ \"x\": ";
        writeJsonArray( out, xs );
        out << ", \"y\": ";
        writeJsonArray( out, ys );
        out << " }";
    };

    out << "  \"ic_curved\": ";
    poly( ic_curved );
    out << ",\n  \"ic_linear\": ";
    poly( ic_linear );
    out << ",\n  \"ic_mc\": ";
    cloud( mc0x, mc0y );
    out << ",\n  \"leaves_curved\": ";
    polys( leaves_curved );
    out << ",\n  \"leaves_linear\": ";
    polys( leaves_linear );
    out << ",\n  \"mc\": ";
    cloud( mcx, mcy );
    out << ",\n  \"reference_orbit\": { \"x0\": ";
    writeJsonArray( out, reference.cols[0] );
    out << ", \"x1\": ";
    writeJsonArray( out, reference.cols[1] );
    out << " }\n}\n";

    printBanner( "two_body/poly_zonotope — curved initial set (orbit-element Gaussian)",
                 { { "factors", "true anomaly nu, eccentricity e" },
                   { "3-sigma (nu,e)", std::to_string( kNu3 ) + ", " + std::to_string( kE3 ) },
                   { "leaves", std::to_string( tree_curved.done().size() ) + " (curved) vs " +
                                   std::to_string( tree_linear.done().size() ) + " (linear)" },
                   { "MC RMS curved", jsonNumber( rms_c ) + " (max " + jsonNumber( max_c ) + ")" },
                   { "MC RMS linear", jsonNumber( rms_l ) + " (max " + jsonNumber( max_l ) + ")" },
                   { "output", "poly_zonotope.json" } } );
    return 0;
}
