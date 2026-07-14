// =============================================================================
// examples/two_body/polynomial_zonotope.cpp
//
// Curved (polynomial) IC domain ADS: propagate an initial-condition set whose
// boundary is a *parabolic* curve rather than a flat face.
//
// A Box or Zonotope IC describes a polytope — all generator terms are linear in
// the factor coordinates ξ.  A PolynomialZonotope allows higher-degree
// generators: x(ξ) = value(ξ) where value is a polynomial map from [-1,1]^M.
// This is the lowest tier of the domain interface — it models Domain but NOT
// LocatableDomain, so merge() is deliberately unavailable.
//
// Here the IC set is the Kepler periapsis box with one small ξ₁² term added
// to the y component, producing a gentle quadratic curvature along the y-axis.
// The ADS propagation subdivides the set under the same TruncationCriterion
// used in ads.cpp; the final partition (tree.done() leaves) is dumped to JSON.
//
// Run:    ./two_body_polynomial_zonotope
// Writes: polynomial_zonotope.json
// =============================================================================

#include <tax/ads.hpp>
#include <tax/ode.hpp>

#include "common.hpp"

int main()
{
    using namespace example;
    using namespace example::two_body;
    using namespace tax::ode::methods;

    // P is the DA truncation order — it MUST equal the PolynomialZonotope degree N.
    constexpr int P = 8;
    constexpr int M = 4;  // factor dimension (= number of DA variables)
    constexpr int D = 4;  // physical state dimension

    constexpr int kNPerEdge = 24;
    const double t_final = kPeriod;

    // ---- Build the polynomial zonotope IC set ---------------------------------
    //
    // Start from the axis-aligned box (degree-1 generators, identical to a Box).
    // Then overwrite one higher-degree coefficient to make the IC set genuinely
    // *curved*: we add a small ξ₁² term to the y-component (value[1]), which
    // bends the y boundary of the initial set into a parabola.  The coefficient
    // is kept to 10 % of the linear half-width so the set stays close to the box.
    auto pz = tax::domain::PolynomialZonotope< double, P, M >::fromBox( icBox() );
    {
        // Monomial ξ₁² in value[1] (the y-position component).
        tax::MultiIndex< M > a{};
        a[1] = 2;  // total degree 2 on the ξ₁ axis
        pz.value[1][tax::flatIndex< M >( a )] = 0.1 * kIcBoxHalfWidth( 1 );  // 8e-4
    }

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    cfg.save_steps = true;

    const tax::ads::TruncationCriterion criterion{ /*tol=*/1e-6, /*maxDepth=*/8 };

    // ---- Scalar centerpoint orbit (plot underlay) ----------------------------
    auto ref_sol = tax::ode::propagate( Taylor< 16 >{}, rhs(), icCenter(), 0.0, t_final, cfg );
    const auto reference = sampleOrbit( ref_sol, example::linspace( 0.0, t_final, 200 ), D );

    // ---- Single ADS propagation, single-threaded ----------------------------
    //
    // num_threads=1: the parallel driver may SIGBUS on large DA states; the
    // single-threaded path is always safe.  Do NOT call merge() on this
    // solution: PolynomialZonotope is not LocatableDomain, so merge() refuses
    // to compile on a PZ tree by design.
    Stopwatch clock;
    auto sol = tax::ads::propagate< P >( Verner89{}, criterion, rhs(), pz, icCenter(), 0.0, t_final,
                                         cfg, /*num_threads=*/1 );

    const auto& tree = sol.tree();
    const auto boundary = unitSquareBoundary( kNPerEdge );

    // Collect the final partition: iterate the done leaves and evaluate each
    // leaf's DA flow map on the active (y, vy) face of the unit cube.
    Snapshot final_snap{ t_final, {} };
    int id = 0;
    for ( int li : tree.done() )
    {
        const auto& leaf = tree.leaf( li );
        final_snap.leaves.push_back(
            evalPolygon( leaf.payload, boundary, boundaryToBox, id++, leaf.depth ) );
    }
    const double elapsed_ms = clock.ms();
    const int n_leaves = static_cast< int >( final_snap.leaves.size() );

    // ---- Output ---------------------------------------------------------------
    writeRunJson( "polynomial_zonotope.json", "polynomial_zonotope",
                  { { "P", std::to_string( P ) },
                    { "M", std::to_string( M ) },
                    { "D", std::to_string( D ) },
                    { "t_final", jsonNumber( t_final ) },
                    { "criterion", "\"truncation\"" },
                    { "tol", jsonNumber( criterion.tol ) },
                    { "max_depth", std::to_string( criterion.maxDepth ) },
                    { "domain", "\"polynomial_zonotope\"" },
                    { "ic_center", jsonArray( icBox().center ) },
                    { "ic_half_width", jsonArray( icBox().halfWidth ) } },
                  reference, { final_snap }, elapsed_ms );

    printBanner( "two_body/polynomial_zonotope — curved (parabolic) IC domain via ADS",
                 { { "P, M", std::to_string( P ) + ", " + std::to_string( M ) },
                   { "criterion", "truncation, tol=1e-6, depth<=8" },
                   { "IC domain", "PolynomialZonotope (box + quadratic xi_1^2 term)" },
                   { "final leaves", std::to_string( n_leaves ) },
                   { "elapsed", std::to_string( elapsed_ms ) + " ms" },
                   { "output", "polynomial_zonotope.json" } } );
    return 0;
}
