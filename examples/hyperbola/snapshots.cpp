// =============================================================================
// examples/hyperbola/snapshots.cpp
//
// Stage 2 (continued) — enclosure snapshots along the hyperbolic trajectory.
//
// The same incoming uncertainty is propagated with two domain options and the
// leaf partition is recorded at several times along the flyby:
//
//   * Zonotope             — the oriented (parallelotope) IC set;
//   * PolynomialZonotope   — a curved IC set (the box plus a small quadratic
//                            term), the lowest tier of the domain interface.
//
// A single ADS propagation per domain uses a snapshot grid (the convenience
// tax::ads::propagate<P> overload); each snapshot is the active partition (a
// tiling of the IC set) at that time. Approaching periapsis the flow stretches
// and the partition refines; the polygons are the (x, y) projection of each
// leaf's flow map.
//
// Run:    ./hyperbola_snapshots
// Writes: hyperbola_snapshots_zono.json, hyperbola_snapshots_pz.json
//         (plot with examples/plot/plot_hyperbola_snapshots.py)
// =============================================================================

#include <string>
#include <tax/ads.hpp>
#include <tax/ode.hpp>
#include <vector>

#include "common.hpp"

int main()
{
    using namespace example;
    using namespace example::hyperbola;
    using namespace tax::ode::methods;

    constexpr int P = 6;
    constexpr int M = kM;
    constexpr int D = 6;
    constexpr int kNSnaps = 9;
    constexpr int kNPerEdge = 24;
    const double t_final = tFinal();

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    cfg.save_steps = true;

    const tax::ads::TruncationCriterion criterion{ /*tol=*/1e-5, /*maxDepth=*/7 };
    const auto boundary = unitSquareBoundary( kNPerEdge );
    const auto grid_times = example::linspace( 0.0, t_final, kNSnaps );

    // ---- Scalar centerpoint orbit (plot underlay), all 6 components ----------
    auto ref_sol = tax::ode::propagate( Taylor< 16 >{}, rhs(), icCenter(), 0.0, t_final, cfg );
    const auto reference = sampleOrbit( ref_sol, example::linspace( 0.0, t_final, 300 ), D );

    // Collect the per-snapshot (x, y) leaf polygons of one ADS solution.
    auto collect = [&]( auto& sol, std::string* count_str ) {
        std::vector< Snapshot > snaps;
        for ( const auto& part : sol.snapshots() )
        {
            Snapshot snap{ part.time(), {} };
            for ( const auto& leaf : part )
                snap.leaves.push_back(
                    evalPolygon( leaf.flowMap, boundary, boundaryToBox, leaf.id, leaf.depth ) );
            if ( count_str )
                *count_str += ( count_str->empty() ? "" : ", " ) +
                              std::to_string( snap.leaves.size() );
            snaps.push_back( std::move( snap ) );
        }
        return snaps;
    };

    const JsonParams base{ { "P", std::to_string( P ) },
                           { "M", std::to_string( M ) },
                           { "D", std::to_string( D ) },
                           { "t_final", jsonNumber( t_final ) },
                           { "t_peri", jsonNumber( tPeri() ) },
                           { "criterion", "\"truncation\"" },
                           { "tol", jsonNumber( criterion.tol ) },
                           { "max_depth", std::to_string( criterion.maxDepth ) },
                           { "ecc", jsonNumber( kEcc ) },
                           { "rp", jsonNumber( kRp ) } };

    // ---- Zonotope domain -----------------------------------------------------
    Stopwatch clock;
    std::string zono_counts;
    auto zono_sol = tax::ads::propagate< P >( Verner89{}, criterion, rhs(), icZonotope(),
                                              icCenter(), 0.0, t_final, grid_times, cfg,
                                              adsThreads() );
    auto zono_snaps = collect( zono_sol, &zono_counts );
    JsonParams zono_params = base;
    zono_params.push_back( { "domain", "\"zonotope\"" } );
    writeRunJson( "hyperbola_snapshots_zono.json", "zonotope", zono_params, reference, zono_snaps,
                  clock.ms() );

    // ---- PolynomialZonotope domain (single-threaded: safe on large DA) -------
    clock = Stopwatch{};
    std::string pz_counts;
    auto pz_sol = tax::ads::propagate< P >( Verner89{}, criterion, rhs(), icPolyZono< P >(),
                                            icCenter(), 0.0, t_final, grid_times, cfg,
                                            /*num_threads=*/1 );
    auto pz_snaps = collect( pz_sol, &pz_counts );
    JsonParams pz_params = base;
    pz_params.push_back( { "domain", "\"polynomial_zonotope\"" } );
    writeRunJson( "hyperbola_snapshots_pz.json", "polynomial_zonotope", pz_params, reference,
                  pz_snaps, clock.ms() );

    printBanner( "hyperbola/snapshots — enclosure tilings along the flyby (Z and PZ)",
                 { { "P, M", std::to_string( P ) + ", " + std::to_string( M ) },
                   { "criterion", "truncation, tol=1e-5, depth<=7" },
                   { "zono leaves/snap", zono_counts },
                   { "pz leaves/snap", pz_counts },
                   { "output", "hyperbola_snapshots_zono.json, hyperbola_snapshots_pz.json" } } );
    return 0;
}
