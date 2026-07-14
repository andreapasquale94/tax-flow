// tests/ads/test_zonotope.cpp
//
// Zonotope (parallelotope) domain: geometry primitives, identity-state
// seeding, and end-to-end ADS propagation over an oriented initial set.
// The end-to-end cases reuse the mildly nonlinear oscillator of
// test_driver.cpp and verify against high-accuracy scalar references, then
// show that an oriented parallelotope resolves with fewer leaves than the
// axis-aligned box that bounds the same set.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <tax/ads/da_state.hpp>
#include <tax/ads/merge.hpp>
#include <tax/ads/propagate.hpp>
#include <tax/ads/split_criteria.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/domain/box.hpp>
#include <tax/domain/zonotope.hpp>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <tax/tax.hpp>
#include <utility>

using tax::ads::TruncationCriterion;
using tax::domain::Box;
using tax::domain::Zonotope;
using tax::ode::IntegratorConfig;

// Permanent compile-time guards: Zonotope must model both domain tiers.
static_assert( tax::domain::Domain< tax::domain::Zonotope< double, 2 > >,
               "Zonotope must model Domain" );
static_assert( tax::domain::LocatableDomain< tax::domain::Zonotope< double, 2 > >,
               "Zonotope must model LocatableDomain" );

namespace
{
constexpr int P = 6;
constexpr int M = 2;
constexpr int D = 2;

using TE = tax::TE< P, M >;
using DAState = tax::la::VecNT< D, TE >;
using ScState = tax::la::VecNT< D, double >;
using V2 = tax::la::VecNT< M, double >;
using Mat2 = tax::la::MatNT< M, double >;

// f(x, v) = (v, -x - 0.1 x^3) — same problem as test_driver.cpp.
auto rhs()
{
    return []( const auto& x, double ) {
        using S = std::decay_t< decltype( x ) >;
        S out{ x.size() };
        out( 0 ) = x( 1 );
        out( 1 ) = -x( 0 ) - 0.1 * x( 0 ) * x( 0 ) * x( 0 );
        return out;
    };
}

ScState scalarReference( ScState x0, double t1 )
{
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    tax::ode::Verner89< ScState > integ{ rhs(), cfg };
    auto sol = integ.integrate( x0, 0.0, t1 );
    return sol.x.back();
}

// Evaluate a DA flow-map payload at local factor coordinates xi ∈ [-1, 1]^M.
ScState evalPayload( const DAState& payload, const std::array< double, M >& xi )
{
    ScState out;
    for ( int row = 0; row < D; ++row )
    {
        double acc = 0.0;
        constexpr std::size_t Nc = tax::numMonomials( P, M );
        for ( std::size_t k = 0; k < Nc; ++k )
        {
            const auto alpha = tax::unflatIndex< M >( k );
            double term = payload( row )[k];
            for ( int j = 0; j < M; ++j )
                for ( int p = 0; p < alpha[static_cast< std::size_t >( j )]; ++p )
                    term *= xi[static_cast< std::size_t >( j )];
            acc += term;
        }
        out( row ) = acc;
    }
    return out;
}

// A 45°-rotated parallelotope: long along one diagonal, thin across it.
Zonotope< double, M > orientedThinZonotope()
{
    const double c = std::cos( M_PI / 4.0 );
    const double s = std::sin( M_PI / 4.0 );
    Mat2 rot;
    rot << c, -s, s, c;
    Mat2 scale;
    scale << 0.8, 0.0, 0.0, 0.03;  // long axis 0.8, thin axis 0.03
    Zonotope< double, M > z;
    z.center = V2{ 1.0, 0.0 };
    z.generators = rot * scale;
    return z;
}

// Axis-aligned box that tightly bounds a parallelotope: per-axis half-width
// is the L1 norm of the corresponding generator row.
Box< double, M > boundingBox( const Zonotope< double, M >& z )
{
    V2 hw;
    for ( int i = 0; i < M; ++i ) hw( i ) = z.generators.row( i ).cwiseAbs().sum();
    return Box< double, M >{ z.center, hw };
}
}  // namespace

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

TEST( Zonotope, AxisAlignedMatchesBox )
{
    auto z = Zonotope< double, M >::axisAligned( V2{ 1.0, 2.0 }, V2{ 0.5, 0.25 } );
    EXPECT_DOUBLE_EQ( z.generators( 0, 0 ), 0.5 );
    EXPECT_DOUBLE_EQ( z.generators( 1, 1 ), 0.25 );
    EXPECT_DOUBLE_EQ( z.generators( 0, 1 ), 0.0 );
    EXPECT_DOUBLE_EQ( z.generators( 1, 0 ), 0.0 );

    // denormalize agrees with the Box mapping center + halfWidth ⊙ d.
    const V2 d{ 0.5, -1.0 };
    const V2 mapped = z.denormalize( d );
    EXPECT_DOUBLE_EQ( mapped( 0 ), 1.0 + 0.5 * 0.5 );
    EXPECT_DOUBLE_EQ( mapped( 1 ), 2.0 + 0.25 * -1.0 );
}

TEST( Zonotope, ContainsOrientedSet )
{
    auto z = orientedThinZonotope();
    // Corners ξ = (±1, ±1) lie on the boundary (inside up to tol).
    for ( double a : { -1.0, 1.0 } )
        for ( double b : { -1.0, 1.0 } ) EXPECT_TRUE( z.contains( z.denormalize( V2{ a, b } ) ) );

    // A point well outside the thin set: step off the long axis by the full
    // long half-length along the thin-axis normal.
    EXPECT_FALSE(
        z.contains( z.center + 0.45 * z.generators.col( 0 ).normalized() + V2{ 0.45, -0.45 } ) );
}

TEST( Zonotope, SplitBisectsGenerator )
{
    auto z = orientedThinZonotope();
    auto [L, R] = z.split( 0 );

    // The split generator is halved; the other is untouched.
    EXPECT_TRUE( L.generators.col( 0 ).isApprox( 0.5 * z.generators.col( 0 ) ) );
    EXPECT_TRUE( R.generators.col( 0 ).isApprox( 0.5 * z.generators.col( 0 ) ) );
    EXPECT_TRUE( L.generators.col( 1 ).isApprox( z.generators.col( 1 ) ) );

    // Centres shift by ∓ half the (old) generator column.
    const V2 half = 0.5 * z.generators.col( 0 );
    EXPECT_TRUE( L.center.isApprox( z.center - half ) );
    EXPECT_TRUE( R.center.isApprox( z.center + half ) );

    // Children tile the parent: parent ξ=+0.5 maps to the same physical point
    // as the right child's local ξ'=0.
    EXPECT_TRUE( z.denormalize( V2{ 0.5, 0.0 } ).isApprox( R.denormalize( V2{ 0.0, 0.0 } ) ) );

    // splitOrdinate orders the pair left < right along the split direction.
    EXPECT_LT( L.splitOrdinate( 0 ), R.splitOrdinate( 0 ) );
}

// ---------------------------------------------------------------------------
// DA state seeding
// ---------------------------------------------------------------------------

TEST( Zonotope, CreateSeedsGeneratorColumns )
{
    auto z = orientedThinZonotope();
    V2 x0{ 1.0, 0.0 };
    DAState state = tax::domain::create< P, M >( z, x0 );

    for ( int i = 0; i < D; ++i )
    {
        // Constant term is the IC centre.
        EXPECT_DOUBLE_EQ( state( i )[0], x0( i ) );
        // First-order coefficient w.r.t. ξ_j is generators(i, j).
        for ( int j = 0; j < M; ++j )
        {
            tax::MultiIndex< M > alpha{};
            alpha[static_cast< std::size_t >( j )] = 1;
            EXPECT_DOUBLE_EQ( state( i )[tax::flatIndex< M >( alpha )], z.generators( i, j ) );
        }
    }
}

// ---------------------------------------------------------------------------
// End-to-end propagation over an oriented set
// ---------------------------------------------------------------------------

TEST( Zonotope, PropagateMatchesReference )
{
    const double t1 = 2.0 * M_PI;
    auto z = orientedThinZonotope();
    V2 center{ 1.0, 0.0 };

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    auto sol = tax::ads::propagate< P >( tax::ode::methods::Verner89{},
                                         TruncationCriterion{ /*tol=*/1e-4, /*maxDepth=*/8 }, rhs(),
                                         z, center, /*t0=*/0.0, t1, cfg );
    const auto& tree = sol.tree();
    ASSERT_GE( tree.done().size(), 1u );

    // Sample physical points strictly inside the root set (interior ξ), find
    // their leaf, recover the leaf-local factor coords, and evaluate.
    const std::array< V2, 5 > xis{ {
        V2{ 0.0, 0.0 },
        V2{ 0.7, 0.5 },
        V2{ -0.6, 0.4 },
        V2{ 0.5, -0.7 },
        V2{ -0.8, -0.3 },
    } };

    for ( const auto& xi : xis )
    {
        const V2 phys = z.denormalize( xi );

        auto idx = tree.locate( phys );
        ASSERT_TRUE( idx.has_value() ) << "no leaf contains physical sample";
        const auto& leaf = tree.leaf( *idx );

        // Leaf-local factor coordinates: ξ_local = G_leaf⁻¹ (phys - c_leaf).
        const V2 xi_local_v =
            leaf.domain.generators.partialPivLu().solve( V2{ phys - leaf.domain.center } );
        std::array< double, M > xi_local{ xi_local_v( 0 ), xi_local_v( 1 ) };

        const ScState x_pred = evalPayload( leaf.payload, xi_local );

        ScState ic;
        ic( 0 ) = phys( 0 );
        ic( 1 ) = phys( 1 );
        const ScState x_ref = scalarReference( ic, t1 );

        EXPECT_NEAR( x_pred( 0 ), x_ref( 0 ), 1e-3 );
        EXPECT_NEAR( x_pred( 1 ), x_ref( 1 ), 1e-3 );
    }
}

// The motivation for oriented domains: a thin, rotated initial set is covered
// by a single parallelotope, whereas the axis-aligned box bounding the same
// set is ~√2× wider on each axis, carries more truncation mass, and therefore
// splits into more leaves. Same problem, same tolerance — fewer domains.
TEST( Zonotope, FewerLeavesThanBoundingBox )
{
    const double t1 = 2.0 * M_PI;
    auto z = orientedThinZonotope();
    auto bb = boundingBox( z );
    V2 center{ 1.0, 0.0 };

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    const TruncationCriterion crit{ /*tol=*/1e-4, /*maxDepth=*/8 };

    auto zono_sol = tax::ads::propagate< P >( tax::ode::methods::Verner89{}, crit, rhs(), z, center,
                                              0.0, t1, cfg );
    auto box_sol = tax::ads::propagate< P >( tax::ode::methods::Verner89{}, crit, rhs(), bb, center,
                                             0.0, t1, cfg );
    const auto& zono_tree = zono_sol.tree();
    const auto& box_tree = box_sol.tree();

    std::cerr << "[ DOMAINS  ] oriented zonotope leaves=" << zono_tree.done().size()
              << "  axis-aligned bounding-box leaves=" << box_tree.done().size() << "\n";

    EXPECT_LT( zono_tree.done().size(), box_tree.done().size() )
        << "oriented zonotope leaves=" << zono_tree.done().size()
        << " bounding-box leaves=" << box_tree.done().size();
}

// merge() over a Zonotope tree: a split followed by a tolerant merge must
// reduce the leaf count (exercising the oriented splitOrdinate sibling
// ordering) while the surviving flow maps still predict the reference.
TEST( Zonotope, SplitThenMergeRoundTrip )
{
    const double t1 = 2.0 * M_PI;
    auto z = orientedThinZonotope();
    V2 center{ 1.0, 0.0 };

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    // Propagate with a tight tolerance so the set splits...
    auto sol = tax::ads::propagate< P >( tax::ode::methods::Verner89{},
                                         TruncationCriterion{ /*tol=*/1e-5, /*maxDepth=*/8 }, rhs(),
                                         z, center, 0.0, t1, cfg );
    auto& tree = sol.tree();
    const std::size_t before = tree.done().size();
    ASSERT_GT( before, 1u ) << "expected the tight tolerance to force splits";

    // ...then merge back with a looser tolerance.
    auto stats = tax::ads::merge( tree, TruncationCriterion{ /*tol=*/1e-2, /*maxDepth=*/8 } );
    EXPECT_GT( stats.merges, 0 );
    EXPECT_LT( tree.done().size(), before );

    // The merged leaves still cover the set and predict the reference.
    const std::array< V2, 3 > xis{ { V2{ 0.0, 0.0 }, V2{ 0.6, 0.5 }, V2{ -0.7, -0.4 } } };
    for ( const auto& xi : xis )
    {
        const V2 phys = z.denormalize( xi );
        auto idx = tree.locate( phys );
        ASSERT_TRUE( idx.has_value() );
        const auto& leaf = tree.leaf( *idx );

        const V2 xi_local_v =
            leaf.domain.generators.partialPivLu().solve( V2{ phys - leaf.domain.center } );
        const ScState x_pred = evalPayload(
            leaf.payload, std::array< double, M >{ xi_local_v( 0 ), xi_local_v( 1 ) } );

        ScState ic;
        ic( 0 ) = phys( 0 );
        ic( 1 ) = phys( 1 );
        const ScState x_ref = scalarReference( ic, t1 );
        EXPECT_NEAR( x_pred( 0 ), x_ref( 0 ), 1e-2 );
        EXPECT_NEAR( x_pred( 1 ), x_ref( 1 ), 1e-2 );
    }
}
