// tests/ads/test_cpz.cpp
//
// Constrained polynomial zonotope: the interval-range emptiness test, the
// feasibility/pruning predicate, and the split that re-expands both the value
// map and the constraints on each child sub-box.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <tax/ads/box.hpp>
#include <tax/ads/cpz.hpp>
#include <tax/ads/da_state.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>

namespace
{
constexpr int P = 4;
constexpr int M = 2;
constexpr int D = 2;

using TE = tax::TE< P, M >;
using CPZ = tax::ads::ConstrainedPolyZonotope< double, P, M, tax::storage::Dense, D >;
using V2 = tax::la::VecNT< 2, double >;

// g(ξ) = c0 + Σ linear/quadratic terms, built from DA variables.
TE linConstraint( double c0, double a0, double a1 )
{
    TE g{};
    g[0] = c0;
    tax::MultiIndex< 2 > e0{};
    e0[0] = 1;
    tax::MultiIndex< 2 > e1{};
    e1[1] = 1;
    g[tax::flatIndex< 2 >( e0 )] = a0;
    g[tax::flatIndex< 2 >( e1 )] = a1;
    return g;
}
}  // namespace

TEST( Cpz, IntervalRadiusIsL1OfNonConstant )
{
    // g = 0.5 + 0.2 ξ0 - 0.3 ξ1 ⇒ radius = 0.2 + 0.3 = 0.5.
    const TE g = linConstraint( 0.5, 0.2, -0.3 );
    EXPECT_DOUBLE_EQ( tax::ads::intervalRadius( g ), 0.5 );
}

TEST( Cpz, FeasibilityIsZeroInRange )
{
    // |c0| = 0.4 < radius 0.5 ⇒ can vanish ⇒ feasible.
    EXPECT_TRUE( tax::ads::constraintFeasible( linConstraint( 0.4, 0.2, -0.3 ) ) );
    // |c0| = 0.6 > radius 0.5 ⇒ cannot vanish ⇒ infeasible (prune).
    EXPECT_FALSE( tax::ads::constraintFeasible( linConstraint( 0.6, 0.2, -0.3 ) ) );
}

TEST( Cpz, SplitReExpandsValueAndConstraints )
{
    // Value: identity on the box [-1,1]^2 (centre 0, unit generators).
    tax::ads::Box< double, M > box{ V2{ 0.0, 0.0 }, V2{ 1.0, 1.0 } };
    CPZ z;
    z.value = tax::ads::create< P, M >( box, V2{ 0.0, 0.0 } );
    // A constraint that is curved enough to exercise the substitution.
    TE g{};
    g[0] = 0.1;
    {
        tax::MultiIndex< 2 > e{};
        e[0] = 1;
        g[tax::flatIndex< 2 >( e )] = 0.7;  // 0.7 ξ0
    }
    {
        tax::MultiIndex< 2 > e{};
        e[1] = 2;
        g[tax::flatIndex< 2 >( e )] = -0.4;  // -0.4 ξ1^2
    }
    z.constraints.push_back( g );

    auto [L, R] = tax::ads::split( z, box, /*dim=*/0 );

    // The right child covers ξ0 ∈ [0,1] of the parent: local ξ0' maps to
    // parent ξ0 = 0.5 + 0.5 ξ0'. Check both the value and the constraint match
    // the parent evaluated at the mapped point.
    const std::array< V2, 4 > pts{
        { V2{ 0.0, 0.0 }, V2{ 0.5, -0.6 }, V2{ -0.3, 0.8 }, V2{ 0.9, 0.2 } } };
    for ( const auto& p : pts )
    {
        const V2 parent_xi{ 0.5 + 0.5 * p( 0 ), p( 1 ) };
        EXPECT_NEAR( R.constraints[0].eval( p ), g.eval( parent_xi ), 1e-12 );
        EXPECT_NEAR( R.value( 0 ).eval( p ), z.value( 0 ).eval( parent_xi ), 1e-12 );
        EXPECT_NEAR( R.value( 1 ).eval( p ), z.value( 1 ).eval( parent_xi ), 1e-12 );
    }
}

// A leaf whose sub-box is far from the constraint surface must be pruned, while
// the half straddling it survives.
TEST( Cpz, SplittingPrunesInfeasibleHalf )
{
    tax::ads::Box< double, M > box{ V2{ 0.0, 0.0 }, V2{ 1.0, 1.0 } };
    CPZ z;
    z.value = tax::ads::create< P, M >( box, V2{ 0.0, 0.0 } );
    // Constraint ξ0 - 0.5 = 0: the surface ξ0 = 0.5 lives only in the right
    // half (ξ0 ∈ [0,1]); the left half (ξ0 ∈ [-1,0]) cannot satisfy it.
    z.constraints.push_back( linConstraint( -0.5, 1.0, 0.0 ) );

    auto [L, R] = tax::ads::split( z, box, /*dim=*/0 );
    EXPECT_FALSE( tax::ads::feasible( L ) );  // left half pruned
    EXPECT_TRUE( tax::ads::feasible( R ) );   // right half kept
}
