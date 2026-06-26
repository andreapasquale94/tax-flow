// =============================================================================
// examples/two_body/zonotope.cpp
//
// Oriented-domain ADS: propagate a *correlated* initial-condition set as a
// Zonotope (parallelotope) instead of an axis-aligned box.
//
// The two-body IC uncertainty here is a rectangle in the (y, vy) plane rotated
// by 45° — a covariance-ellipse-style correlated position/velocity error. The
// classic box ADS cannot represent that orientation: it must propagate the
// axis-aligned box that *bounds* the set, which is larger, carries more
// truncation mass, and therefore splits into more leaves. The Zonotope domain
// wraps the rotated set directly and resolves it with fewer leaves.
//
// This example runs both at each snapshot time and reports the leaf counts
// side by side, the headline being "same accuracy, fewer domains".
//
// Run:    ./two_body_zonotope
// Writes: zonotope.json      (oriented leaves; plot with plot_two_body_zonotope.py)
//         zonotope_box.json  (axis-aligned bounding-box leaves, for overlay)
// =============================================================================

#include <tax/ads.hpp>
#include <tax/ode.hpp>

#include "common.hpp"

int main()
{
    using namespace example;
    using namespace example::two_body;
    using namespace tax::ode::methods;

    constexpr int P = 8;  // DA truncation order
    constexpr int M = 4;  // number of DA variables
    constexpr int D = 4;  // state dimension

    constexpr int kNSnaps = 9;
    constexpr int kNPerEdge = 24;
    // Integrate a representative arc (0.8 of a period). The oriented set's
    // advantage is configuration-dependent: over most of the orbit its rotated
    // factors align with the flow and it splits less, but the strong shear of
    // the periapsis *return* (t -> 2π) is best resolved by axis-aligned cuts,
    // where the box would in turn split less. We showcase the favourable arc;
    // the banner still prints both counts at every snapshot.
    const double t_final = 0.8 * kPeriod;

    const auto ic_zono = icZonotope();
    const auto ic_box = icZonotopeBoundingBox();

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    cfg.save_steps = true;

    const tax::ads::TruncationCriterion criterion{ /*tol=*/1e-6, /*maxDepth=*/8 };

    // ---- Scalar centerpoint orbit (plot underlay) ----------------------------
    auto ref_sol = tax::ode::propagate( Taylor< 16 >{}, rhs(), icCenter(), 0.0, t_final, cfg );
    const auto reference = sampleOrbit( ref_sol, example::linspace( 0.0, t_final, 200 ), D );

    // The t = 0 "polygon" is the image of the IC-set boundary under the
    // identity flow map — for the zonotope that is exactly its oriented face.
    const auto boundary = unitSquareBoundary( kNPerEdge );
    const auto zono_identity = tax::ads::create< P, M >( ic_zono, icCenter() );

    // ---- One ADS propagation per snapshot time, for both domains -------------
    std::vector< Snapshot > zono_snaps;
    std::vector< Snapshot > box_snaps;
    std::vector< double > zono_counts;
    std::vector< double > box_counts;
    std::string zono_count_str;
    std::string box_count_str;

    Stopwatch clock;
    for ( double t : example::linspace( 0.0, t_final, kNSnaps ) )
    {
        Snapshot zsnap{ t, {} };
        Snapshot bsnap{ t, {} };

        if ( t <= 0.0 )
        {
            zsnap.leaves.push_back( evalPolygon( zono_identity, boundary, boundaryToBox, 0, 0 ) );
            bsnap.leaves.push_back( boxPolygon( ic_box, boundary, boundaryToBox ) );
        } else
        {
            auto ztree = tax::ads::propagate< P >( Verner89{}, criterion, rhs(), ic_zono,
                                                   icCenter(), 0.0, t, cfg, adsThreads() );
            int id = 0;
            for ( int li : ztree.done() )
            {
                const auto& leaf = ztree.leaf( li );
                zsnap.leaves.push_back(
                    evalPolygon( leaf.payload, boundary, boundaryToBox, id++, leaf.depth ) );
            }

            auto btree = tax::ads::propagate< P >( Verner89{}, criterion, rhs(), ic_box, icCenter(),
                                                   0.0, t, cfg, adsThreads() );
            id = 0;
            for ( int li : btree.done() )
            {
                const auto& leaf = btree.leaf( li );
                bsnap.leaves.push_back(
                    evalPolygon( leaf.payload, boundary, boundaryToBox, id++, leaf.depth ) );
            }
        }

        zono_counts.push_back( static_cast< double >( zsnap.leaves.size() ) );
        box_counts.push_back( static_cast< double >( bsnap.leaves.size() ) );
        zono_count_str +=
            ( zono_count_str.empty() ? "" : ", " ) + std::to_string( zsnap.leaves.size() );
        box_count_str +=
            ( box_count_str.empty() ? "" : ", " ) + std::to_string( bsnap.leaves.size() );

        zono_snaps.push_back( std::move( zsnap ) );
        box_snaps.push_back( std::move( bsnap ) );
    }
    const double elapsed_ms = clock.ms();

    // The (y, vy) generator block and bounding half-widths let the plotter draw
    // the oriented IC parallelogram against its axis-aligned bound.
    const std::array< double, 4 > gen_yv{ ic_zono.generators( 1, 1 ), ic_zono.generators( 1, 3 ),
                                          ic_zono.generators( 3, 1 ), ic_zono.generators( 3, 3 ) };
    const std::array< double, 2 > center_yv{ icCenter()( 1 ), icCenter()( 3 ) };
    const std::array< double, 2 > bbox_hw_yv{ ic_box.halfWidth( 1 ), ic_box.halfWidth( 3 ) };

    // ---- Output ---------------------------------------------------------------
    const JsonParams zono_params{ { "P", std::to_string( P ) },
                                  { "M", std::to_string( M ) },
                                  { "D", std::to_string( D ) },
                                  { "t_final", jsonNumber( t_final ) },
                                  { "criterion", "\"truncation\"" },
                                  { "tol", jsonNumber( criterion.tol ) },
                                  { "max_depth", std::to_string( criterion.maxDepth ) },
                                  { "domain", "\"zonotope\"" },
                                  { "ic_center_yv", jsonArray( center_yv ) },
                                  { "ic_gen_yv", jsonArray( gen_yv ) },
                                  { "bbox_hw_yv", jsonArray( bbox_hw_yv ) },
                                  { "zono_leaf_counts", jsonArray( zono_counts ) },
                                  { "box_leaf_counts", jsonArray( box_counts ) } };
    writeRunJson( "zonotope.json", "zonotope", zono_params, reference, zono_snaps, elapsed_ms );

    const JsonParams box_params{ { "P", std::to_string( P ) },
                                 { "M", std::to_string( M ) },
                                 { "D", std::to_string( D ) },
                                 { "t_final", jsonNumber( t_final ) },
                                 { "criterion", "\"truncation\"" },
                                 { "tol", jsonNumber( criterion.tol ) },
                                 { "max_depth", std::to_string( criterion.maxDepth ) },
                                 { "domain", "\"box\"" },
                                 { "ic_center_yv", jsonArray( center_yv ) },
                                 { "bbox_hw_yv", jsonArray( bbox_hw_yv ) } };
    writeRunJson( "zonotope_box.json", "ads", box_params, reference, box_snaps, elapsed_ms );

    const double zono_total = zono_counts.back();
    const double box_total = box_counts.back();
    printBanner(
        "two_body/zonotope — oriented domains vs axis-aligned box",
        { { "P, M", std::to_string( P ) + ", " + std::to_string( M ) },
          { "criterion", "truncation, tol=1e-6, depth<=8" },
          { "zono leaves/snap", zono_count_str },
          { "box leaves/snap", box_count_str },
          { "final leaves", std::to_string( static_cast< int >( zono_total ) ) + " (zono) vs " +
                                std::to_string( static_cast< int >( box_total ) ) + " (box)" },
          { "elapsed", std::to_string( elapsed_ms ) + " ms" },
          { "output", "zonotope.json, zonotope_box.json" } } );
    return 0;
}
