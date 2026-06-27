// =============================================================================
// examples/two_body/zonotope_adaptive.cpp
//
// Adaptive oriented-domain ADS: choose the covering parallelotope's
// orientation from the *flow* instead of fixing it.
//
// The static zonotope example (zonotope.cpp) wins over most of an orbit but
// loses at the periapsis return, because a fixed 45° frame is arbitrary. When
// the initial uncertainty is an ELLIPSOID (a covariance), the covering
// parallelotope is free to orient — so we can align it with the dynamics:
//
//   1. probe: propagate the (un-split) identity once to read the flow's
//      linear map Φ over the horizon (linearPart of the propagated map);
//   2. align: V = right-singular vectors of Φ·L (L = chol of the covariance);
//      the covering generators G = L·V make the propagated set Φ·L·V = U·Σ
//      have orthogonal generators — no thin diagonal sliver to over-split.
//
// We compare three coverings of the SAME ellipsoid over a FULL period:
//   (a) axis-aligned bounding Box,
//   (b) fixed Cholesky parallelotope (oriented by the covariance only),
//   (c) flow-aligned parallelotope (oriented by the dynamics).
//
// Run:    ./two_body_zonotope_adaptive
// Writes: zonotope_adaptive.json   (the three leaf-count series + IC frames)
// =============================================================================

#include <Eigen/Dense>
#include <tax/ads.hpp>
#include <tax/ode.hpp>

#include "common.hpp"

namespace
{
using Mat4 = tax::la::MatNT< 4, double >;
using Mat2 = tax::la::MatNT< 2, double >;
using Vec4 = tax::la::VecNT< 4, double >;

// The two uncertain axes of the two-body state: y and v_y.
constexpr int kAy = 1;
constexpr int kAv = 3;

// Embed a 2×2 block (acting on the (y, v_y) subspace) into the full 4×4
// generator matrix; the pinned axes (x, v_x) stay zero.
Mat4 embed( const Mat2& B )
{
    Mat4 G = Mat4::Zero();
    G( kAy, kAy ) = B( 0, 0 );
    G( kAy, kAv ) = B( 0, 1 );
    G( kAv, kAy ) = B( 1, 0 );
    G( kAv, kAv ) = B( 1, 1 );
    return G;
}

// Extract the (y, v_y) block from a full 4×4 matrix.
Mat2 active( const Mat4& A )
{
    Mat2 B;
    B << A( kAy, kAy ), A( kAy, kAv ), A( kAv, kAy ), A( kAv, kAv );
    return B;
}

tax::ads::Zonotope< double, 4 > zonoFrom( const Mat2& block )
{
    tax::ads::Zonotope< double, 4 > z;
    z.center = example::two_body::icCenter();
    z.generators = embed( block );
    return z;
}
}  // namespace

int main()
{
    using namespace example;
    using namespace example::two_body;
    using namespace tax::ode::methods;

    constexpr int P = 8;
    constexpr int M = 4;
    constexpr int D = 4;
    constexpr int kNSnaps = 9;
    const double t_final = kPeriod;  // the full period — where the static frame flipped

    // ---- Ellipsoidal IC uncertainty on (y, v_y): a correlated covariance ----
    const double sy = 0.02;  // 1σ in y
    const double sv = 0.03;  // 1σ in v_y
    const double rho = 0.5;  // correlation
    Mat2 C;
    C << sy * sy, rho * sy * sv, rho * sy * sv, sv * sv;
    const Mat2 L = Eigen::LLT< Mat2 >( C ).matrixL();  // C = L Lᵀ

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    cfg.save_steps = true;

    const tax::ads::TruncationCriterion criterion{ /*tol=*/1e-6, /*maxDepth=*/8 };

    // ---- Probe: one un-split propagation to read the horizon STM Φ -----------
    // maxDepth = 0 never splits, so done() is a single leaf carrying the
    // full-period flow map. Only its linear part is used (the STM is exact
    // regardless of higher-order truncation), so a single leaf is enough.
    auto probe =
        tax::ads::propagate< P >( Verner89{}, tax::ads::TruncationCriterion{ /*tol=*/1e18, 0 },
                                  rhs(), zonoFrom( L ), icCenter(), 0.0, t_final, cfg );
    const auto& probe_tree = probe.tree();
    const auto& probe_map = probe_tree.leaf( probe_tree.done().front() ).payload;
    const Mat4 Phi = tax::ads::linearPart( probe_map );    // ∂x/∂ξ at t_final
    const Mat2 PhiL = active( Phi );                       // = Φ_act · L (probe used G = L)
    const Mat2 V = tax::ads::flowAlignedRotation( PhiL );  // align the frame to the flow

    // ---- The three coverings of the same ellipsoid ---------------------------
    // (a) axis-aligned bounding box: half-width on each axis is that row's
    //     2-norm of L (the ellipse's marginal extent).
    Vec4 box_hw = Vec4::Zero();
    box_hw( kAy ) = L.row( 0 ).norm();
    box_hw( kAv ) = L.row( 1 ).norm();
    const tax::ads::Box< double, 4 > cover_box{ icCenter(), box_hw };
    const auto cover_chol = zonoFrom( L );         // (b) fixed Cholesky frame
    const auto cover_aligned = zonoFrom( L * V );  // (c) flow-aligned frame

    // ---- reorientState demonstration -----------------------------------------
    // Re-express the probe flow map in the aligned frame and confirm the
    // deformation is diagonalised: the off-diagonal of (Φ_act)ᵀΦ_act drops to ~0.
    Mat4 R4 = Mat4::Identity();
    R4( kAy, kAy ) = V( 0, 0 );
    R4( kAy, kAv ) = V( 0, 1 );
    R4( kAv, kAy ) = V( 1, 0 );
    R4( kAv, kAv ) = V( 1, 1 );
    const auto probe_aligned = tax::ads::reorientState( probe_map, R4 );
    const Mat2 PhiLV = active( tax::ads::linearPart( probe_aligned ) );
    const double offdiag_before = std::abs( ( PhiL.transpose() * PhiL )( 0, 1 ) );
    const double offdiag_after = std::abs( ( PhiLV.transpose() * PhiLV )( 0, 1 ) );

    // ---- Leaf counts per snapshot for each covering --------------------------
    auto leafCounts = [&]( auto domain ) {
        std::vector< double > counts;
        std::vector< Snapshot > snaps;
        for ( double t : example::linspace( 0.0, t_final, kNSnaps ) )
        {
            if ( t <= 0.0 )
            {
                counts.push_back( 1.0 );
                snaps.push_back( Snapshot{ t, {} } );
                continue;
            }
            auto sol = tax::ads::propagate< P >( Verner89{}, criterion, rhs(), domain, icCenter(),
                                                 0.0, t, cfg, adsThreads() );
            const auto& tree = sol.tree();
            counts.push_back( static_cast< double >( tree.done().size() ) );
            snaps.push_back( Snapshot{ t, {} } );
        }
        return std::make_pair( counts, snaps );
    };

    Stopwatch clock;
    auto [box_counts, box_snaps] = leafCounts( cover_box );
    auto [chol_counts, chol_snaps] = leafCounts( cover_chol );
    auto [aligned_counts, aligned_snaps] = leafCounts( cover_aligned );
    const double elapsed_ms = clock.ms();

    auto countStr = []( const std::vector< double >& c ) {
        std::string s;
        for ( double v : c )
            s += ( s.empty() ? "" : ", " ) + std::to_string( static_cast< int >( v ) );
        return s;
    };

    // ---- Reference orbit + final-time leaf tiling for the aligned covering ---
    auto ref_sol = tax::ode::propagate( Taylor< 16 >{}, rhs(), icCenter(), 0.0, t_final, cfg );
    const auto reference = sampleOrbit( ref_sol, example::linspace( 0.0, t_final, 200 ), D );

    const auto boundary = unitSquareBoundary( 24 );
    std::vector< Snapshot > aligned_tiling;
    {
        auto sol = tax::ads::propagate< P >( Verner89{}, criterion, rhs(), cover_aligned,
                                             icCenter(), 0.0, t_final, cfg, adsThreads() );
        const auto& tree = sol.tree();
        Snapshot snap{ t_final, {} };
        int id = 0;
        for ( int li : tree.done() )
        {
            const auto& leaf = tree.leaf( li );
            snap.leaves.push_back(
                evalPolygon( leaf.payload, boundary, boundaryToBox, id++, leaf.depth ) );
        }
        aligned_tiling.push_back( std::move( snap ) );
    }

    const std::array< double, 4 > Lblk{ L( 0, 0 ), L( 0, 1 ), L( 1, 0 ), L( 1, 1 ) };
    const Mat2 LV = L * V;
    const std::array< double, 4 > LVblk{ LV( 0, 0 ), LV( 0, 1 ), LV( 1, 0 ), LV( 1, 1 ) };
    const std::array< double, 2 > center_yv{ icCenter()( kAy ), icCenter()( kAv ) };
    const std::array< double, 2 > box_hw_yv{ box_hw( kAy ), box_hw( kAv ) };

    writeRunJson( "zonotope_adaptive.json", "zonotope_adaptive",
                  { { "P", std::to_string( P ) },
                    { "M", std::to_string( M ) },
                    { "t_final", jsonNumber( t_final ) },
                    { "tol", jsonNumber( criterion.tol ) },
                    { "ic_center_yv", jsonArray( center_yv ) },
                    { "cov", jsonArray( std::array< double, 4 >{ C( 0, 0 ), C( 0, 1 ), C( 1, 0 ),
                                                                 C( 1, 1 ) } ) },
                    { "chol_yv", jsonArray( Lblk ) },
                    { "aligned_yv", jsonArray( LVblk ) },
                    { "box_hw_yv", jsonArray( box_hw_yv ) },
                    { "box_leaf_counts", jsonArray( box_counts ) },
                    { "chol_leaf_counts", jsonArray( chol_counts ) },
                    { "aligned_leaf_counts", jsonArray( aligned_counts ) } },
                  reference, aligned_tiling, elapsed_ms );

    printBanner(
        "two_body/zonotope_adaptive — flow-aligned covering over a full period",
        { { "P, M", std::to_string( P ) + ", " + std::to_string( M ) },
          { "ellipse 1σ (y,vy)",
            std::to_string( sy ) + ", " + std::to_string( sv ) + "  rho=" + std::to_string( rho ) },
          { "box leaves/snap", countStr( box_counts ) },
          { "chol leaves/snap", countStr( chol_counts ) },
          { "aligned leaves/snap", countStr( aligned_counts ) },
          { "final leaves",
            std::to_string( static_cast< int >( box_counts.back() ) ) + " (box), " +
                std::to_string( static_cast< int >( chol_counts.back() ) ) + " (chol), " +
                std::to_string( static_cast< int >( aligned_counts.back() ) ) + " (aligned)" },
          { "STM offdiag", "before=" + std::to_string( offdiag_before ) +
                               "  after reorient=" + std::to_string( offdiag_after ) },
          { "elapsed", std::to_string( elapsed_ms ) + " ms" },
          { "output", "zonotope_adaptive.json" } } );
    return 0;
}
