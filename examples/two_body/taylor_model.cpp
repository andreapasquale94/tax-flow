// =============================================================================
// examples/two_body/taylor_model.cpp
//
// VALIDATED two-body propagation with Taylor models (requires a tax core with
// the tax::model module — see docs/ode/taylor_models.md).
//
// Three acts, one JSON:
//
//  1. SINGLE Taylor model over the IC box, propagated piecewise around the
//     orbit with tax::ode::methods::Picard. Every accepted state carries a
//     rigorous remainder interval; we record its growth, the interval hull of
//     the (x, y) enclosure, and boundary polygons of the polynomial part.
//     Near the return to periapsis the remainder demand of one global model
//     explodes — the run is stopped where verification fails.
//
//  2. ADS over Taylor-model payloads: the same box, same method, but the
//     TruncationCriterion now splits the domain whenever the polynomial
//     truncation frontier exceeds tolerance. Every leaf carries its own
//     (much smaller) validated remainder, and the orbit completes. We record
//     the leaf partition of the IC box, per-snapshot enclosure quality, and a
//     Monte-Carlo containment check: every sampled true flow must lie inside
//     the located leaf's interval enclosure.
//
//  3. DOMAIN CONVERSIONS at the final time: for one ADS leaf, the nested set
//     representations — Monte-Carlo truth cloud ⊂ zonotope enclosure ⊂
//     interval hull, plus the (non-enclosing) linear zonotope frame.
//
// Run:    ./two_body_taylor_model
// Writes: taylor_model.json  (plot with examples/plot/plot_two_body_taylor_model.py)
// =============================================================================

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <tax/ads.hpp>
#include <tax/ode.hpp>
#include <vector>

#include "common.hpp"

#if !__has_include( <tax/model.hpp>)
#include <cstdio>
int main()
{
    std::puts(
        "two_body_taylor_model: the tax core lacks the tax::model module — rebuild "
        "against a checkout with Taylor models to run this example." );
    return 0;
}
#else

namespace
{

using namespace example;
using namespace example::two_body;

constexpr int P = 5;  // Taylor-model order
constexpr int M = 4;  // factor dimension (state dimension here)
constexpr int D = 4;  // state dimension

constexpr double kOdeTol = 1e-7;  // Picard τ-truncation tolerance
constexpr double kAdsTol = 1e-3;  // ADS truncation-frontier tolerance
constexpr int kMaxDepth = 8;      // ADS depth cap
constexpr int kSegments = 64;     // piecewise recording grid for the single run
constexpr int kNSnaps = 9;        // snapshot times for polygons (0 .. T, 45°)
constexpr int kNPerEdge = 24;     // boundary samples per IC-box edge
constexpr int kNMc = 160;         // Monte-Carlo validation samples

using TM = tax::model::TaylorModel< double, P, M >;
using State = Eigen::Matrix< TM, D, 1 >;
using IV = tax::model::Interval< double >;

// ---- Small JSON helpers (schema is example-specific) ------------------------

void writeKV( std::ostream& os, const char* key, double v, bool comma = true )
{
    os << "    \"" << key << "\": " << v << ( comma ? ",\n" : "\n" );
}

template < class Range >
void writeArr( std::ostream& os, const char* key, const Range& v, bool comma = true )
{
    os << "    \"" << key << "\": ";
    writeJsonArray( os, v );
    os << ( comma ? ",\n" : "\n" );
}

// Boundary polygon of the polynomial parts of a Taylor-model state (the
// (x, y) image of the IC-box boundary — remainder NOT included; the hull
// rectangles carry it).
Polygon modelPolygon( const State& s, const std::vector< std::array< double, 2 > >& boundary,
                      int id = 0, int depth = 0 )
{
    Polygon p;
    p.id = id;
    p.depth = depth;
    p.x.reserve( boundary.size() );
    p.y.reserve( boundary.size() );
    for ( const auto& ab : boundary )
    {
        const auto d = boundaryToBox( ab[0], ab[1] );
        p.x.push_back( s( 0 ).polynomial().eval( d ) );
        p.y.push_back( s( 1 ).polynomial().eval( d ) );
    }
    return p;
}

void writePolygon( std::ostream& os, const Polygon& p, const char* indent )
{
    os << indent << "{ \"id\": " << p.id << ", \"depth\": " << p.depth << ", \"x\": ";
    writeJsonArray( os, p.x );
    os << ", \"y\": ";
    writeJsonArray( os, p.y );
    os << " }";
}

// (x, y) interval-hull rectangle of a Taylor-model state: [cx ± hx] × [cy ± hy].
struct HullRect
{
    double t = 0.0;
    double cx = 0, cy = 0, hx = 0, hy = 0;
};

HullRect hullOf( const State& s, double t )
{
    const auto hull = tax::domain::intervalHull( s, std::array< int, 2 >{ 0, 1 } );
    return { t, hull.center( 0 ), hull.center( 1 ), hull.halfWidth( 0 ), hull.halfWidth( 1 ) };
}

void writeHull( std::ostream& os, const HullRect& h, const char* indent )
{
    os << indent << "{ \"t\": " << h.t << ", \"cx\": " << h.cx << ", \"cy\": " << h.cy
       << ", \"hx\": " << h.hx << ", \"hy\": " << h.hy << " }";
}

// 2-D zonotope (center + generator columns) for the plot script to polygonize.
void writeZono2( std::ostream& os, const char* key, const Eigen::Vector2d& c,
                 const std::vector< std::array< double, 2 > >& gens, bool comma )
{
    os << "    \"" << key << "\": { \"c\": [" << c( 0 ) << ", " << c( 1 ) << "], \"gens\": [";
    for ( std::size_t j = 0; j < gens.size(); ++j )
        os << ( j ? ", " : "" ) << "[" << gens[j][0] << ", " << gens[j][1] << "]";
    os << "] }" << ( comma ? ",\n" : "\n" );
}

}  // namespace

int main()
{
    const double t_final = kPeriod;
    const auto ic_box = icBox();
    const auto boundary = unitSquareBoundary( kNPerEdge );

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = kOdeTol;
    cfg.save_steps = true;
    // Circuit breaker: when a single global model hits its remainder wall the
    // verification rejections would otherwise thrash for a long time at ever
    // smaller h before the rejection cap fires. A short segment never needs
    // more than a few dozen accepted steps, so a tight cap fails fast.
    cfg.max_steps = 400;

    // ---- Scalar centerpoint orbit (plot underlay) ---------------------------
    tax::ode::IntegratorConfig< double > ref_cfg;
    ref_cfg.abstol = ref_cfg.reltol = 1e-12;
    auto ref_sol = tax::ode::propagate( tax::ode::methods::Taylor< 16 >{}, rhs(), icCenter(), 0.0,
                                        t_final, ref_cfg );
    const auto reference = sampleOrbit( ref_sol, {}, D );

    // ================= Act 1: one validated Taylor model =====================
    //
    // Piecewise propagation so we can record the remainder growth on a fixed
    // grid AND stop gracefully where the single model's verification fails.
    State s = tax::domain::createModel< P >( ic_box, icCenter() );

    std::vector< double > seg_t{ 0.0 };
    std::vector< std::array< double, 4 > > seg_rem{ { 0.0, 0.0, 0.0, 0.0 } };
    std::vector< HullRect > single_hulls{ hullOf( s, 0.0 ) };
    std::vector< Snapshot > single_snaps;
    single_snaps.push_back( { 0.0, { modelPolygon( s, boundary ) } } );

    Stopwatch clock;
    double reached = 0.0;
    std::string fail_reason;
    const auto snap_times = example::linspace( 0.0, t_final, kNSnaps );
    for ( int k = 0; k < kSegments; ++k )
    {
        const double ta = t_final * k / kSegments;
        const double tb = t_final * ( k + 1 ) / kSegments;
        cfg.initial_step = ( tb - ta ) / 4.0;
        try
        {
            auto sol = tax::ode::propagate( tax::ode::methods::Picard{}, rhs(), s, ta, tb, cfg );
            s = sol.x.back();
            reached = tb;
        } catch ( const std::exception& ex )
        {
            fail_reason = ex.what();
            break;
        }
        seg_t.push_back( tb );
        seg_rem.push_back( { s( 0 ).remainder().width(), s( 1 ).remainder().width(),
                             s( 2 ).remainder().width(), s( 3 ).remainder().width() } );
        single_hulls.push_back( hullOf( s, tb ) );
        for ( double st : snap_times )
            if ( st > 0.0 && std::abs( st - tb ) < 0.25 * t_final / kSegments )
                single_snaps.push_back( { tb, { modelPolygon( s, boundary ) } } );
    }
    const double single_ms = clock.ms();

    // ================= Act 2: ADS over Taylor-model payloads =================
    clock = Stopwatch{};
    cfg.max_steps = 4000;  // leaves integrate long spans; keep a generous cap
    cfg.initial_step = 0.0;
    auto asol = tax::ads::propagate< P >(
        tax::ode::methods::Picard{}, tax::ads::TruncationCriterion{ kAdsTol, kMaxDepth }, rhs(),
        ic_box, icCenter(), 0.0, t_final, std::vector< double >( snap_times ), cfg );
    const double ads_ms = clock.ms();

    // Per-snapshot partitions: leaf polygons + max remainder width.
    const auto parts = asol.snapshots();
    std::vector< Snapshot > ads_snaps;
    std::vector< double > snap_rem_t, snap_rem_max;
    for ( const auto& part : parts )
    {
        Snapshot snap;
        snap.t = part.time();
        double worst = 0.0;
        for ( const auto& lv : part )
        {
            State st{ D };
            for ( int i = 0; i < D; ++i ) st( i ) = lv.flowMap( i );
            snap.leaves.push_back( modelPolygon( st, boundary, lv.id, lv.depth ) );
            for ( int i = 0; i < D; ++i )
                worst = std::max( worst, lv.flowMap( i ).remainder().width() );
        }
        ads_snaps.push_back( std::move( snap ) );
        snap_rem_t.push_back( part.time() );
        snap_rem_max.push_back( worst );
    }

    // Final leaf partition of the IC box in the (y0, vy0) plane, with each
    // leaf's remainder width (enclosure quality).
    struct LeafRect
    {
        int id, depth;
        double y_lo, y_hi, v_lo, v_hi, rem;
    };
    std::vector< LeafRect > leaf_rects;
    {
        const auto fin = asol.final();
        int id = 0;
        for ( const auto& lv : fin )
        {
            double worst = 0.0;
            for ( int i = 0; i < D; ++i )
                worst = std::max( worst, lv.flowMap( i ).remainder().width() );
            leaf_rects.push_back( { id++, lv.depth,
                                    lv.domain.center( 1 ) - lv.domain.halfWidth( 1 ),
                                    lv.domain.center( 1 ) + lv.domain.halfWidth( 1 ),
                                    lv.domain.center( 3 ) - lv.domain.halfWidth( 3 ),
                                    lv.domain.center( 3 ) + lv.domain.halfWidth( 3 ), worst } );
        }
    }

    // Monte-Carlo containment: sample the box (deterministic lattice + jitter-
    // free), propagate each IC with a scalar integrator, require the true
    // final state to lie INSIDE the located leaf's interval enclosure.
    int contained = 0, located = 0;
    std::vector< double > mc_x, mc_y;
    {
        const int side = static_cast< int >( std::round( std::sqrt( double( kNMc ) ) ) );
        for ( int a = 0; a < side; ++a )
            for ( int b = 0; b < side; ++b )
            {
                const double xi1 = -1.0 + 2.0 * ( a + 0.5 ) / side;
                const double xi2 = -1.0 + 2.0 * ( b + 0.5 ) / side;
                auto ic = icCenter();
                ic( 1 ) += ic_box.halfWidth( 1 ) * xi1;
                ic( 3 ) += ic_box.halfWidth( 3 ) * xi2;

                auto truth = tax::ode::propagate( tax::ode::methods::Taylor< 16 >{}, rhs(), ic, 0.0,
                                                  t_final, ref_cfg )
                                 .x.back();
                mc_x.push_back( truth( 0 ) );
                mc_y.push_back( truth( 1 ) );

                const auto enc = asol.evaluate( ic );
                if ( !enc ) continue;
                ++located;
                bool ok = true;
                for ( int i = 0; i < D; ++i ) ok = ok && ( *enc )( i ).contains( truth( i ) );
                contained += ok ? 1 : 0;
            }
    }

    // ================= Act 3: domain conversions on one leaf =================
    //
    // Pick the leaf whose subdomain contains the box center; emit its final
    // enclosure in every representation the domain module offers.
    std::ostringstream conv;
    conv << std::setprecision( 14 );
    {
        const auto loc = asol.tree().locate( icCenter() );
        const auto& leaf = asol.tree().leaf( *loc );
        const State& pay = leaf.payload;

        // Truth cloud from THIS leaf's subdomain.
        std::vector< double > lx, ly;
        const int side = 12;
        for ( int a = 0; a < side; ++a )
            for ( int b = 0; b < side; ++b )
            {
                const double xi1 = -1.0 + 2.0 * ( a + 0.5 ) / side;
                const double xi2 = -1.0 + 2.0 * ( b + 0.5 ) / side;
                tax::la::VecNT< 4, double > ic;
                for ( int i = 0; i < D; ++i )
                    ic( i ) =
                        leaf.domain.center( i ) +
                        leaf.domain.halfWidth( i ) * ( i == 1 ? xi1 : ( i == 3 ? xi2 : 0.0 ) );
                auto truth = tax::ode::propagate( tax::ode::methods::Taylor< 16 >{}, rhs(), ic, 0.0,
                                                  t_final, ref_cfg )
                                 .x.back();
                lx.push_back( truth( 0 ) );
                ly.push_back( truth( 1 ) );
            }

        const auto hull = tax::domain::intervalHull( pay, std::array< int, 2 >{ 0, 1 } );
        const auto zen = tax::domain::zonotopeEnclosure( pay, std::array< int, 2 >{ 0, 1 } );
        const auto frame = tax::domain::zonotopeFrame( pay );

        conv << "  \"conversions\": {\n";
        writeArr( conv, "mc_x", lx );
        writeArr( conv, "mc_y", ly );
        conv << "    \"hull\": { \"cx\": " << hull.center( 0 ) << ", \"cy\": " << hull.center( 1 )
             << ", \"hx\": " << hull.halfWidth( 0 ) << ", \"hy\": " << hull.halfWidth( 1 )
             << " },\n";
        std::vector< std::array< double, 2 > > zg;
        for ( Eigen::Index j = 0; j < zen.generators.cols(); ++j )
            zg.push_back( { zen.generators( 0, j ), zen.generators( 1, j ) } );
        writeZono2( conv, "enclosure", { zen.center( 0 ), zen.center( 1 ) }, zg, true );
        std::vector< std::array< double, 2 > > fg;
        for ( int j = 0; j < M; ++j )
            fg.push_back( { frame.generators( 0, j ), frame.generators( 1, j ) } );
        writeZono2( conv, "frame", { frame.center( 0 ), frame.center( 1 ) }, fg, true );
        Polygon lp = modelPolygon( pay, boundary );
        conv << "    \"poly\": ";
        {
            std::ostringstream px, py;
            px << std::setprecision( 14 );
            py << std::setprecision( 14 );
            writeJsonArray( px, lp.x );
            writeJsonArray( py, lp.y );
            conv << "{ \"x\": " << px.str() << ", \"y\": " << py.str() << " },\n";
        }
        conv << "    \"leaf_depth\": " << leaf.depth << "\n  },\n";
    }

    // ================= JSON ====================================================
    std::ofstream out( "taylor_model.json" );
    out << std::setprecision( 14 );
    out << "{\n  \"params\": {\n";
    writeKV( out, "P", P );
    writeKV( out, "M", M );
    writeKV( out, "ode_tol", kOdeTol );
    writeKV( out, "ads_tol", kAdsTol );
    writeKV( out, "t_final", t_final );
    writeKV( out, "single_reached", reached );
    writeKV( out, "single_ms", single_ms );
    writeKV( out, "ads_ms", ads_ms );
    writeKV( out, "ads_leaves", double( leaf_rects.size() ) );
    writeKV( out, "mc_total", double( mc_x.size() ) );
    writeKV( out, "mc_located", double( located ) );
    writeKV( out, "mc_contained", double( contained ) );
    out << "    \"ic_center\": " << jsonArray( ic_box.center ) << ",\n";
    out << "    \"ic_half_width\": " << jsonArray( ic_box.halfWidth ) << ",\n";
    out << "    \"single_fail\": \"" << fail_reason << "\"\n  },\n";

    out << "  \"reference_orbit\": { \"t\": ";
    writeJsonArray( out, reference.t );
    out << ", \"x\": ";
    writeJsonArray( out, reference.cols[0] );
    out << ", \"y\": ";
    writeJsonArray( out, reference.cols[1] );
    out << " },\n";

    // Act 1 payload.
    out << "  \"single\": {\n";
    writeArr( out, "t", seg_t );
    for ( int i = 0; i < 4; ++i )
    {
        std::vector< double > col;
        for ( const auto& r : seg_rem ) col.push_back( r[std::size_t( i )] );
        const std::string key = "rem" + std::to_string( i );
        writeArr( out, key.c_str(), col );
    }
    out << "    \"hulls\": [\n";
    for ( std::size_t i = 0; i < single_hulls.size(); ++i )
    {
        writeHull( out, single_hulls[i], "      " );
        out << ( i + 1 < single_hulls.size() ? ",\n" : "\n" );
    }
    out << "    ],\n    \"snapshots\": [\n";
    for ( std::size_t i = 0; i < single_snaps.size(); ++i )
    {
        out << "      { \"t\": " << single_snaps[i].t << ", \"poly\": ";
        {
            std::ostringstream px, py;
            px << std::setprecision( 14 );
            py << std::setprecision( 14 );
            writeJsonArray( px, single_snaps[i].leaves[0].x );
            writeJsonArray( py, single_snaps[i].leaves[0].y );
            out << "{ \"x\": " << px.str() << ", \"y\": " << py.str() << " } }";
        }
        out << ( i + 1 < single_snaps.size() ? ",\n" : "\n" );
    }
    out << "    ]\n  },\n";

    // Act 2 payload.
    out << "  \"ads\": {\n";
    writeArr( out, "snap_t", snap_rem_t );
    writeArr( out, "snap_rem_max", snap_rem_max );
    out << "    \"leaves\": [\n";
    for ( std::size_t i = 0; i < leaf_rects.size(); ++i )
    {
        const auto& l = leaf_rects[i];
        out << "      { \"id\": " << l.id << ", \"depth\": " << l.depth << ", \"y\": [" << l.y_lo
            << ", " << l.y_hi << "], \"vy\": [" << l.v_lo << ", " << l.v_hi
            << "], \"rem\": " << l.rem << " }" << ( i + 1 < leaf_rects.size() ? ",\n" : "\n" );
    }
    out << "    ],\n    \"snapshots\": [\n";
    for ( std::size_t sI = 0; sI < ads_snaps.size(); ++sI )
    {
        out << "      { \"t\": " << ads_snaps[sI].t << ", \"leaves\": [\n";
        for ( std::size_t l = 0; l < ads_snaps[sI].leaves.size(); ++l )
        {
            writePolygon( out, ads_snaps[sI].leaves[l], "        " );
            out << ( l + 1 < ads_snaps[sI].leaves.size() ? ",\n" : "\n" );
        }
        out << "      ] }" << ( sI + 1 < ads_snaps.size() ? ",\n" : "\n" );
    }
    out << "    ],\n";
    writeArr( out, "mc_x", mc_x );
    writeArr( out, "mc_y", mc_y, false );
    out << "  },\n";

    out << conv.str();
    out << "  \"end\": true\n}\n";

    // ---- Banner -----------------------------------------------------------------
    printBanner( "two-body / validated Taylor models",
                 { { "order P", std::to_string( P ) },
                   { "single reached", jsonNumber( reached ) + " of " + jsonNumber( t_final ) },
                   { "single time", jsonNumber( single_ms ) + " ms" },
                   { "ADS leaves", std::to_string( leaf_rects.size() ) },
                   { "ADS time", jsonNumber( ads_ms ) + " ms" },
                   { "MC contained", std::to_string( contained ) + "/" + std::to_string( located ) +
                                         " located of " + std::to_string( mc_x.size() ) },
                   { "output", "taylor_model.json" } } );
    return 0;
}

#endif  // __has_include(<tax/model.hpp>)
