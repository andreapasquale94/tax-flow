// =============================================================================
// examples/zonotope/two_body_mc.cpp
//
// Monte-Carlo validation of the oriented / polynomial-zonotope ADS on the
// two-body problem. For two initial-condition sets — an oriented
// parallelotope and a covariance ellipsoid — we propagate the set with each
// covering domain to one full period, then draw 10000 Monte-Carlo samples of
// the SAME set, propagate each with a high-accuracy scalar integrator, and:
//
//   * overlay the true cloud on the leaf tiling (does the union of polynomial
//     zonotopes envelope the real image?);
//   * score the piecewise-polynomial prediction against the truth — for every
//     sample, locate its leaf, evaluate that leaf's flow map, and measure the
//     error (RMS and max over the 10000 samples).
//
// The headline: every covering reproduces the cloud to ~the split tolerance,
// but the flow-aligned / oriented frames need far fewer leaves to do it.
//
// Run:    ./zonotope_two_body_mc
// Writes: two_body_mc_parallelotope.json, two_body_mc_ellipse.json
// =============================================================================

#include <Eigen/Dense>
#include <cmath>
#include <optional>
#include <random>
#include <string>
#include <tax/ads.hpp>
#include <tax/ode.hpp>
#include <utility>
#include <vector>

#include "../two_body/common.hpp"

namespace
{
using namespace example;
using namespace example::two_body;
using namespace tax::ode::methods;

constexpr int P = 8;
constexpr int M = 4;
constexpr int D = 4;
constexpr int kAy = 1;  // y axis
constexpr int kAv = 3;  // v_y axis
constexpr int kNMonte = 10000;

using TE = tax::TE< P, M >;
using DAState = tax::la::VecNT< D, TE >;
using Vec4 = tax::la::VecNT< 4, double >;
using V2 = tax::la::VecNT< 2, double >;
using Mat2 = tax::la::MatNT< 2, double >;
using Mat4 = tax::la::MatNT< 4, double >;

Mat4 embed( const Mat2& B )
{
    Mat4 G = Mat4::Zero();
    G( kAy, kAy ) = B( 0, 0 );
    G( kAy, kAv ) = B( 0, 1 );
    G( kAv, kAy ) = B( 1, 0 );
    G( kAv, kAv ) = B( 1, 1 );
    return G;
}
Mat2 activeBlock( const Mat4& A )
{
    Mat2 B;
    B << A( kAy, kAy ), A( kAy, kAv ), A( kAv, kAy ), A( kAv, kAv );
    return B;
}
tax::ads::Zonotope< double, 4 > zonoFrom( const Mat2& B )
{
    tax::ads::Zonotope< double, 4 > z;
    z.center = icCenter();
    z.generators = embed( B );
    return z;
}

// The (y, v_y) generator block + centre of a leaf domain (Box or Zonotope),
// used to recover leaf-local factor coordinates without touching the singular
// pinned (x, v_x) axes.
std::pair< Mat2, V2 > leafBlock( const tax::ads::Box< double, 4 >& b )
{
    Mat2 B;
    B << b.halfWidth( kAy ), 0.0, 0.0, b.halfWidth( kAv );
    return { B, V2{ b.center( kAy ), b.center( kAv ) } };
}
std::pair< Mat2, V2 > leafBlock( const tax::ads::Zonotope< double, 4 >& z )
{
    return { activeBlock( z.generators ), V2{ z.center( kAy ), z.center( kAv ) } };
}

tax::ode::IntegratorConfig< double > fastCfg()
{
    tax::ode::IntegratorConfig< double > c;
    c.abstol = c.reltol = 1e-12;
    c.save_steps = false;
    return c;
}

// Truth: scalar propagation of one IC to the final time, returning (x, y).
V2 truthXY( const Vec4& ic, double t_final )
{
    auto sol = tax::ode::propagate( Verner89{}, rhs(), ic, 0.0, t_final, fastCfg() );
    return V2{ sol.x.back()( 0 ), sol.x.back()( 1 ) };
}

// Locate the leaf containing `ic` (by leaf-local |ξ_active|∞ ≤ 1) and evaluate
// its flow map → predicted (x, y). nullopt if no leaf claims the point.
template < class Tree >
std::optional< V2 > predictXY( const Tree& tree, const Vec4& ic )
{
    const V2 icAct{ ic( kAy ), ic( kAv ) };
    double best = 1e30;
    const DAState* bestPayload = nullptr;
    V2 bestXi;
    for ( int li : tree.done() )
    {
        const auto& leaf = tree.leaf( li );
        auto [B, c] = leafBlock( leaf.box );
        const V2 xi = B.partialPivLu().solve( V2{ icAct - c } );
        const double inf = xi.cwiseAbs().maxCoeff();
        if ( inf < best )
        {
            best = inf;
            bestXi = xi;
            bestPayload = &leaf.payload;
        }
    }
    if ( bestPayload == nullptr || best > 1.0 + 1e-6 ) return std::nullopt;
    Vec4 d = Vec4::Zero();
    d( kAy ) = bestXi( 0 );
    d( kAv ) = bestXi( 1 );
    return V2{ ( *bestPayload )( 0 ).eval( d ), ( *bestPayload )( 1 ).eval( d ) };
}

struct DomainResult
{
    std::string name;
    std::string ic_kind;         // "box" or "zono"
    std::array< double, 4 > ic;  // box: [hy, hv, 0, 0]; zono: [g11,g13,g31,g33]
    int n_leaves = 0;
    double rms = 0.0;
    double max_err = 0.0;
    std::vector< Polygon > leaves;
};

// Build the leaf tiling and the MC error scores for one covering domain.
template < class Domain >
DomainResult scoreDomain( const std::string& name, const std::string& kind,
                          const std::array< double, 4 >& ic_desc, const Domain& domain,
                          double t_final, const std::vector< Vec4 >& samples,
                          const std::vector< V2 >& truth,
                          const std::vector< std::array< double, 2 > >& boundary )
{
    DomainResult r;
    r.name = name;
    r.ic_kind = kind;
    r.ic = ic_desc;

    auto tree =
        tax::ads::propagate< P >( Verner89{}, tax::ads::TruncationCriterion{ 1e-6, 8 }, rhs(),
                                  domain, icCenter(), 0.0, t_final, fastCfg(), adsThreads() );
    r.n_leaves = static_cast< int >( tree.done().size() );
    int id = 0;
    for ( int li : tree.done() )
        r.leaves.push_back( evalPolygon( tree.leaf( li ).payload, boundary, boundaryToBox, id++,
                                         tree.leaf( li ).depth ) );

    double sq = 0.0;
    int cnt = 0;
    for ( std::size_t s = 0; s < samples.size(); ++s )
    {
        auto pred = predictXY( tree, samples[s] );
        if ( !pred ) continue;
        const double e =
            std::hypot( ( *pred )( 0 ) - truth[s]( 0 ), ( *pred )( 1 ) - truth[s]( 1 ) );
        sq += e * e;
        r.max_err = std::max( r.max_err, e );
        ++cnt;
    }
    r.rms = cnt > 0 ? std::sqrt( sq / cnt ) : 0.0;
    return r;
}

void writeScenario( const std::string& path, const std::string& scenario, double t_final,
                    const V2& center_yv, const std::vector< V2 >& cloud,
                    const std::vector< DomainResult >& doms, const OrbitSamples& ref )
{
    std::ofstream out( path );
    out << std::setprecision( 10 );
    out << "{\n  \"scenario\": \"" << scenario << "\",\n";
    out << "  \"t_final\": " << t_final << ",\n";
    out << "  \"n_monte\": " << cloud.size() << ",\n";
    out << "  \"center_yv\": [" << center_yv( 0 ) << ", " << center_yv( 1 ) << "],\n";

    std::vector< double > cx, cy;
    cx.reserve( cloud.size() );
    cy.reserve( cloud.size() );
    for ( const auto& p : cloud )
    {
        cx.push_back( p( 0 ) );
        cy.push_back( p( 1 ) );
    }
    out << "  \"mc\": { \"x\": ";
    writeJsonArray( out, cx );
    out << ", \"y\": ";
    writeJsonArray( out, cy );
    out << " },\n";

    out << "  \"reference_orbit\": { \"x0\": ";
    writeJsonArray( out, ref.cols[0] );
    out << ", \"x1\": ";
    writeJsonArray( out, ref.cols[1] );
    out << " },\n";

    out << "  \"domains\": [\n";
    for ( std::size_t i = 0; i < doms.size(); ++i )
    {
        const auto& d = doms[i];
        out << "    { \"name\": \"" << d.name << "\", \"ic_kind\": \"" << d.ic_kind
            << "\", \"ic\": [" << d.ic[0] << ", " << d.ic[1] << ", " << d.ic[2] << ", " << d.ic[3]
            << "], \"n_leaves\": " << d.n_leaves << ", \"rms\": " << d.rms
            << ", \"max_err\": " << d.max_err << ", \"leaves\": [";
        for ( std::size_t l = 0; l < d.leaves.size(); ++l )
        {
            out << "{ \"x\": ";
            writeJsonArray( out, d.leaves[l].x );
            out << ", \"y\": ";
            writeJsonArray( out, d.leaves[l].y );
            out << " }" << ( l + 1 < d.leaves.size() ? ", " : "" );
        }
        out << "] }" << ( i + 1 < doms.size() ? ",\n" : "\n" );
    }
    out << "  ]\n}\n";
}
}  // namespace

int main()
{
    // Scenario A (a fixed orientation) is shown over a favourable 0.8-orbit
    // arc, where the oriented frame wins; scenario B (the flow-aligned frame)
    // is shown over a FULL period, the harder case where a fixed frame flips
    // but the adaptive one still wins.
    const double tA = 0.8 * kPeriod;
    const double tB = kPeriod;
    const auto boundary = unitSquareBoundary( 24 );

    // Reference orbit underlay (full orbit, used by both scenarios).
    auto refCfg = fastCfg();
    refCfg.save_steps = true;
    auto ref_sol = tax::ode::propagate( Taylor< 16 >{}, rhs(), icCenter(), 0.0, kPeriod, refCfg );
    const auto reference = sampleOrbit( ref_sol, {}, D );

    std::mt19937 rng( 2024u );
    std::uniform_real_distribution< double > unit( -1.0, 1.0 );
    std::uniform_real_distribution< double > u01( 0.0, 1.0 );
    const V2 center_yv{ icCenter()( kAy ), icCenter()( kAv ) };

    Stopwatch clock;

    // ===================== Scenario A: oriented parallelotope =================
    {
        const auto zono = icZonotope();  // 45°-rotated set
        const Mat2 G = activeBlock( zono.generators );
        const auto box = icZonotopeBoundingBox();

        // MC over the parallelotope: ξ uniform in [-1,1]², ic = center + G·ξ.
        std::vector< Vec4 > samples;
        std::vector< V2 > truth;
        samples.reserve( kNMonte );
        truth.reserve( kNMonte );
        for ( int s = 0; s < kNMonte; ++s )
        {
            const V2 off = G * V2{ unit( rng ), unit( rng ) };
            Vec4 ic = icCenter();
            ic( kAy ) += off( 0 );
            ic( kAv ) += off( 1 );
            samples.push_back( ic );
            truth.push_back( truthXY( ic, tA ) );
        }

        std::vector< DomainResult > doms;
        doms.push_back( scoreDomain( "bounding box", "box",
                                     { box.halfWidth( kAy ), box.halfWidth( kAv ), 0, 0 }, box, tA,
                                     samples, truth, boundary ) );
        doms.push_back( scoreDomain( "oriented zonotope", "zono",
                                     { G( 0, 0 ), G( 0, 1 ), G( 1, 0 ), G( 1, 1 ) }, zono, tA,
                                     samples, truth, boundary ) );
        std::vector< V2 > cloud = truth;
        writeScenario( "two_body_mc_parallelotope.json", "parallelotope", tA, center_yv, cloud,
                       doms, reference );
    }

    // ===================== Scenario B: covariance ellipsoid ===================
    {
        const double sy = 0.02, sv = 0.03, rho = 0.5;
        Mat2 C;
        C << sy * sy, rho * sy * sv, rho * sy * sv, sv * sv;
        const Mat2 L = Eigen::LLT< Mat2 >( C ).matrixL();

        // Probe STM and flow-aligned frame.
        auto probe =
            tax::ads::propagate< P >( Verner89{}, tax::ads::TruncationCriterion{ 1e18, 0 }, rhs(),
                                      zonoFrom( L ), icCenter(), 0.0, tB, fastCfg() );
        const Mat2 PhiL =
            activeBlock( tax::ads::linearPart( probe.leaf( probe.done().front() ).payload ) );
        const Mat2 V = tax::ads::flowAlignedRotation( PhiL );
        const Mat2 LV = L * V;

        // MC over the 1σ ellipsoid: u uniform in the unit disk, ic = center + L·u.
        std::vector< Vec4 > samples;
        std::vector< V2 > truth;
        samples.reserve( kNMonte );
        truth.reserve( kNMonte );
        for ( int s = 0; s < kNMonte; ++s )
        {
            const double r = std::sqrt( u01( rng ) );
            const double th = 2.0 * M_PI * u01( rng );
            const V2 off = L * V2{ r * std::cos( th ), r * std::sin( th ) };
            Vec4 ic = icCenter();
            ic( kAy ) += off( 0 );
            ic( kAv ) += off( 1 );
            samples.push_back( ic );
            truth.push_back( truthXY( ic, tB ) );
        }

        Vec4 box_hw = Vec4::Zero();
        box_hw( kAy ) = L.row( 0 ).norm();
        box_hw( kAv ) = L.row( 1 ).norm();
        const tax::ads::Box< double, 4 > box{ icCenter(), box_hw };

        std::vector< DomainResult > doms;
        doms.push_back( scoreDomain( "bounding box", "box", { box_hw( kAy ), box_hw( kAv ), 0, 0 },
                                     box, tB, samples, truth, boundary ) );
        doms.push_back( scoreDomain( "Cholesky", "zono",
                                     { L( 0, 0 ), L( 0, 1 ), L( 1, 0 ), L( 1, 1 ) }, zonoFrom( L ),
                                     tB, samples, truth, boundary ) );
        doms.push_back( scoreDomain( "flow-aligned", "zono",
                                     { LV( 0, 0 ), LV( 0, 1 ), LV( 1, 0 ), LV( 1, 1 ) },
                                     zonoFrom( LV ), tB, samples, truth, boundary ) );
        writeScenario( "two_body_mc_ellipse.json", "ellipse", tB, center_yv, truth, doms,
                       reference );
    }

    const double elapsed = clock.ms();
    printBanner( "zonotope/two_body_mc — 10000-sample Monte-Carlo validation",
                 { { "scenarios", "parallelotope, ellipse" },
                   { "mc samples", std::to_string( kNMonte ) },
                   { "elapsed", std::to_string( elapsed ) + " ms" },
                   { "output", "two_body_mc_parallelotope.json, two_body_mc_ellipse.json" } } );
    return 0;
}
