// =============================================================================
// examples/zonotope/representations.cpp
//
// Set representations on a SIMPLE domain — the hierarchy a tutorial needs.
//
// We take one explicit nonlinear map φ and one square of initial conditions,
// and emit four ways of describing the image φ(box):
//
//   * MC cloud          — 10000 samples φ(ξ), the ground-truth image set;
//   * box               — the axis-aligned interval hull of the image
//                         (sum of |coefficients|, a rigorous enclosure);
//   * linear image      — the first-order term c + J·ξ (a parallelotope); note
//                         this is NOT an enclosure on its own — it misses the
//                         curvature (a zonotope method would add a remainder);
//   * polynomial zonotope — the full DA image, the curved set that the
//                         library actually stores as a leaf payload.
//
// Because φ is degree 2 and the expansion order P ≥ 2, the polynomial zonotope
// is *exact*: the MC cloud fills it exactly, while the box is a loose
// enclosure. A second block shows what splitting buys: the union of the
// per-sub-box linear images (4, then 16 pieces) converges onto the curved
// set — this is exactly what ADS does, piece by piece.
//
// Run:    ./zonotope_representations
// Writes: representations.json   (plot with plot_representations.py)
// =============================================================================

#include <array>
#include <random>
#include <tax/ads.hpp>
#include <tax/tax.hpp>
#include <utility>
#include <vector>

#include "../common/output.hpp"

namespace
{
constexpr int P = 4;
constexpr int M = 2;
constexpr int D = 2;

using TE = tax::TE< P, M >;
using DAState = tax::la::VecNT< D, TE >;
using V2 = tax::la::VecNT< 2, double >;

// A genuinely nonlinear (degree-2) map, generic over scalar / DA arguments.
//   φ(p, q) = ( p + 0.5 q + 0.35 q²,  q + 0.45 p² - 0.25 p q )
template < class S >
std::array< S, 2 > phi( const S& p, const S& q )
{
    return { p + 0.5 * q + 0.35 * q * q, q + 0.45 * p * p - 0.25 * p * q };
}

// Linear-zonotope frame of a 2-component DA map: centre = constant term,
// generators J(i,j) = ∂out_i/∂ξ_j (the degree-1 coefficients).
struct Frame
{
    V2 c;
    tax::la::MatNT< 2, double > J;
};

Frame frameOf( const DAState& s )
{
    Frame f;
    for ( int i = 0; i < 2; ++i )
    {
        f.c( i ) = s( i )[0];
        for ( int j = 0; j < 2; ++j )
        {
            tax::MultiIndex< 2 > e{};
            e[static_cast< std::size_t >( j )] = 1;
            f.J( i, j ) = s( i )[tax::flatIndex< 2 >( e )];
        }
    }
    return f;
}

// Interval hull (axis-aligned box) of a polynomial zonotope over ξ ∈ [-1,1]²:
// |ξ^α| ≤ 1, so the half-extent of component i is Σ_{α≠0} |c_α|.
std::array< double, 4 > boxHull( const DAState& s )
{
    std::array< double, 4 > lohi{};  // x_lo, x_hi, y_lo, y_hi
    constexpr std::size_t Nc = tax::numMonomials( P, M );
    for ( int i = 0; i < 2; ++i )
    {
        const double c0 = s( i )[0];
        double rad = 0.0;
        for ( std::size_t k = 1; k < Nc; ++k ) rad += std::abs( s( i )[k] );
        lohi[static_cast< std::size_t >( 2 * i )] = c0 - rad;
        lohi[static_cast< std::size_t >( 2 * i + 1 )] = c0 + rad;
    }
    return lohi;
}

// Emit a polygon (closed boundary curve) for a DA map over the unit-square
// boundary — the polynomial-zonotope outline.
example::Polygon polyOutline( const DAState& s, const std::vector< std::array< double, 2 > >& bnd )
{
    example::Polygon p;
    for ( const auto& ab : bnd )
    {
        V2 d{ ab[0], ab[1] };
        p.x.push_back( s( 0 ).eval( d ) );
        p.y.push_back( s( 1 ).eval( d ) );
    }
    return p;
}

void writeFrames( std::ostream& out, const std::vector< Frame >& frames )
{
    out << '[';
    for ( std::size_t i = 0; i < frames.size(); ++i )
    {
        const auto& f = frames[i];
        out << "{ \"c\": [" << f.c( 0 ) << ", " << f.c( 1 ) << "], \"J\": [" << f.J( 0, 0 ) << ", "
            << f.J( 0, 1 ) << ", " << f.J( 1, 0 ) << ", " << f.J( 1, 1 ) << "] }"
            << ( i + 1 < frames.size() ? ", " : "" );
    }
    out << ']';
}
}  // namespace

int main()
{
    // Identity DA state on the box ξ ∈ [-1,1]² with physical half-widths
    // (1.0, 0.6): the polynomial zonotope of the *identity* set.
    tax::ads::Zonotope< double, M > dom;
    dom.center = V2{ 0.0, 0.0 };
    dom.generators = ( tax::la::VecNT< 2, double >{ 1.0, 0.6 } ).asDiagonal();
    DAState id = tax::ads::create< P, M >( dom, V2{ 0.0, 0.0 } );

    // Push the identity through φ → polynomial zonotope of φ(box).
    auto img = phi( id( 0 ), id( 1 ) );
    DAState pz;
    pz( 0 ) = img[0];
    pz( 1 ) = img[1];

    const Frame zono = frameOf( pz );
    const auto box = boxHull( pz );
    const auto boundary = example::unitSquareBoundary( 40 );
    const example::Polygon outline = polyOutline( pz, boundary );

    // ---- Monte-Carlo ground truth: 10000 samples φ(physical(ξ)) -------------
    std::mt19937 rng( 7u );
    std::uniform_real_distribution< double > unit( -1.0, 1.0 );
    std::vector< double > mc_x, mc_y;
    mc_x.reserve( 10000 );
    mc_y.reserve( 10000 );
    for ( int s = 0; s < 10000; ++s )
    {
        const double xi0 = unit( rng ), xi1 = unit( rng );
        const double px = dom.generators( 0, 0 ) * xi0;  // physical IC (diagonal)
        const double py = dom.generators( 1, 1 ) * xi1;
        auto v = phi( px, py );
        mc_x.push_back( v[0] );
        mc_y.push_back( v[1] );
    }

    // ---- Splitting: union of per-sub-box linear zonotopes -------------------
    // Recursively bisect the factor box (da_state::split re-expands the DA on
    // each half) and take each sub-leaf's linear frame. level k → 2^k boxes.
    auto subFrames = [&]( int n_splits ) {
        std::vector< DAState > leaves{ pz };
        for ( int s = 0; s < n_splits; ++s )
        {
            std::vector< DAState > next;
            const int dim = s % M;  // alternate split axis
            for ( const auto& leaf : leaves )
            {
                auto pr = tax::ads::split( leaf, dom, dim );
                next.push_back( std::move( pr.first ) );
                next.push_back( std::move( pr.second ) );
            }
            leaves = std::move( next );
        }
        std::vector< Frame > frames;
        frames.reserve( leaves.size() );
        for ( const auto& leaf : leaves ) frames.push_back( frameOf( leaf ) );
        return frames;
    };
    const auto split2 = subFrames( 2 );  // 4 pieces
    const auto split4 = subFrames( 4 );  // 16 pieces

    // ---- Output --------------------------------------------------------------
    std::ofstream out( "representations.json" );
    out << std::setprecision( 10 );
    out << "{\n";
    out << "  \"half_widths\": [" << dom.generators( 0, 0 ) << ", " << dom.generators( 1, 1 )
        << "],\n";
    out << "  \"mc\": { \"x\": ";
    example::writeJsonArray( out, mc_x );
    out << ", \"y\": ";
    example::writeJsonArray( out, mc_y );
    out << " },\n";
    out << "  \"box\": [" << box[0] << ", " << box[1] << ", " << box[2] << ", " << box[3] << "],\n";
    out << "  \"zonotope\": { \"c\": [" << zono.c( 0 ) << ", " << zono.c( 1 ) << "], \"J\": ["
        << zono.J( 0, 0 ) << ", " << zono.J( 0, 1 ) << ", " << zono.J( 1, 0 ) << ", "
        << zono.J( 1, 1 ) << "] },\n";
    out << "  \"poly_zonotope\": { \"x\": ";
    example::writeJsonArray( out, outline.x );
    out << ", \"y\": ";
    example::writeJsonArray( out, outline.y );
    out << " },\n";
    out << "  \"split2\": ";
    writeFrames( out, split2 );
    out << ",\n  \"split4\": ";
    writeFrames( out, split4 );
    out << "\n}\n";

    example::printBanner(
        "zonotope/representations — box vs zonotope vs polynomial zonotope",
        { { "map", "phi(p,q) = (p+0.5q+0.35q^2, q+0.45p^2-0.25pq)" },
          { "box (xlo,xhi)", std::to_string( box[0] ) + ", " + std::to_string( box[1] ) },
          { "mc samples", "10000" },
          { "split pieces", "4, 16" },
          { "output", "representations.json" } } );
    return 0;
}
