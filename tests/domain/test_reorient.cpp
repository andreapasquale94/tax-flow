// tests/ads/test_reorient.cpp
//
// Re-orientation primitive: the linear change of factor variables on a DA
// flow map (reorientState), the local STM extractor (linearPart), and the
// flow-aligned rotation (flowAlignedRotation). The defining property is the
// eval-equivalence reorientState(x, R)(η) == x(R·η).

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <tax/ads/da_state.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/domain/reorient.hpp>
#include <tax/domain/zonotope.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>

namespace
{
constexpr int P = 5;
constexpr int M = 2;
constexpr int D = 2;

using TE = tax::TE< P, M >;
using DAState = tax::la::VecNT< D, TE >;
using V2 = tax::la::VecNT< M, double >;
using Mat2 = tax::la::MatNT< M, double >;

// A genuinely nonlinear two-variable map, as DA about ξ = 0.
//   x0(ξ) = 1 + ξ0 + 0.3 ξ0 ξ1 - 0.1 ξ1^3
//   x1(ξ) = 2 - 0.5 ξ1 + 0.2 ξ0^2
DAState sampleMap()
{
    TE xi0 = TE::variable( 0.0, 0 );
    TE xi1 = TE::variable( 0.0, 1 );
    DAState s;
    s( 0 ) = 1.0 + xi0 + 0.3 * xi0 * xi1 - 0.1 * xi1 * xi1 * xi1;
    s( 1 ) = 2.0 - 0.5 * xi1 + 0.2 * xi0 * xi0;
    return s;
}

Mat2 rot( double th )
{
    Mat2 r;
    r << std::cos( th ), -std::sin( th ), std::sin( th ), std::cos( th );
    return r;
}
}  // namespace

TEST( Reorient, EvalEquivalence )
{
    const DAState x = sampleMap();
    const Mat2 R = rot( 0.6 );
    const DAState y = tax::domain::reorientState( x, R );

    // y(η) must equal x(R·η) for arbitrary η in the domain.
    const std::array< V2, 5 > etas{
        { V2{ 0.0, 0.0 }, V2{ 0.3, -0.7 }, V2{ -0.9, 0.2 }, V2{ 0.5, 0.5 }, V2{ -0.4, -0.6 } } };
    for ( const auto& eta : etas )
    {
        const V2 xi = R * eta;
        for ( int i = 0; i < D; ++i )
            EXPECT_NEAR( y( i ).eval( eta ), x( i ).eval( xi ), 1e-12 )
                << "component " << i << " at eta = (" << eta( 0 ) << ", " << eta( 1 ) << ")";
    }
}

TEST( Reorient, IdentityRotationIsNoOp )
{
    const DAState x = sampleMap();
    const DAState y = tax::domain::reorientState( x, Mat2::Identity() );
    constexpr std::size_t Nc = tax::numMonomials( P, M );
    for ( int i = 0; i < D; ++i )
        for ( std::size_t k = 0; k < Nc; ++k ) EXPECT_NEAR( y( i )[k], x( i )[k], 1e-12 );
}

TEST( Reorient, LinearPartIsTheJacobian )
{
    const DAState x = sampleMap();
    const auto A = tax::domain::linearPart( x );
    // ∂x0/∂ξ0 = 1, ∂x0/∂ξ1 = 0, ∂x1/∂ξ0 = 0, ∂x1/∂ξ1 = -0.5.
    EXPECT_NEAR( A( 0, 0 ), 1.0, 1e-12 );
    EXPECT_NEAR( A( 0, 1 ), 0.0, 1e-12 );
    EXPECT_NEAR( A( 1, 0 ), 0.0, 1e-12 );
    EXPECT_NEAR( A( 1, 1 ), -0.5, 1e-12 );
}

// flowAlignedRotation(A) returns V from SVD(A); A·V must have orthogonal
// columns (zero off-diagonal in (A·V)ᵀ(A·V) = Σ²).
TEST( Reorient, FlowAlignedRotationOrthogonalizesColumns )
{
    Mat2 A;
    A << 1.0, 0.8, 0.2, 1.3;  // sheared — columns not orthogonal
    const Mat2 V = tax::domain::flowAlignedRotation( A );
    const Mat2 AV = A * V;
    EXPECT_NEAR( AV.col( 0 ).dot( AV.col( 1 ) ), 0.0, 1e-12 );
    // V is orthogonal.
    EXPECT_TRUE( ( V.transpose() * V ).isApprox( Mat2::Identity(), 1e-12 ) );
}

// Re-orienting the identity flow map and its zonotope generators by the same R
// describes the same physical IC: create(z)(η) reorients to create(z·R)(η)·...
// Concretely, reorientState(create(z, c)) == create(reorientZonotope(z, R), c)
// as polynomials in η (both equal c + G·R·η).
TEST( Reorient, IdentityAndZonotopeStayConsistent )
{
    const Mat2 R = rot( 0.4 );
    Mat2 G;
    G << 0.5, 0.1, -0.2, 0.4;

    tax::domain::Zonotope< double, M > z;
    z.center = V2{ 0.0, 0.0 };
    z.generators = G;
    const tax::la::VecNT< D, double > c{ 1.0, 2.0 };

    const DAState id = tax::domain::create< P, M >( z, c );
    const DAState id_re = tax::domain::reorientState( id, R );

    const auto zr = tax::domain::reorientZonotope( z, R );
    const DAState id_zr = tax::domain::create< P, M >( zr, c );

    constexpr std::size_t Nc = tax::numMonomials( P, M );
    for ( int i = 0; i < D; ++i )
        for ( std::size_t k = 0; k < Nc; ++k ) EXPECT_NEAR( id_re( i )[k], id_zr( i )[k], 1e-12 );
}
