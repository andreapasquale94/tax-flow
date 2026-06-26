// tests/ads/test_da_state.cpp
//
// create identity property; split round-trip on the parent.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <tax/ads/domains/box.hpp>
#include <tax/ads/da_state.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>
#include <utility>

using tax::ads::Box;
using tax::ads::create;
using tax::ads::split;

namespace
{
constexpr int P = 4;
constexpr int M = 2;
using TE = tax::TE< P, M >;
using State = tax::la::VecNT< 2, TE >;

// Evaluate a TE at ξ ∈ R^M (treating it as a polynomial in ξ).
double evalTe( const TE& f, std::array< double, M > xi )
{
    double acc = 0.0;
    constexpr std::size_t Nc = tax::numMonomials( P, M );
    for ( std::size_t k = 0; k < Nc; ++k )
    {
        const auto alpha = tax::unflatIndex< M >( k );
        double term = f[k];
        for ( int j = 0; j < M; ++j )
            for ( int p = 0; p < alpha[static_cast< std::size_t >( j )]; ++p )
                term *= xi[static_cast< std::size_t >( j )];
        acc += term;
    }
    return acc;
}
}  // namespace

TEST( AdsDaState, CreateMapsZeroDeviationToCenter )
{
    Box< double, M > box{ tax::la::VecNT< M, double >{ 1.0, 2.0 },
                          tax::la::VecNT< M, double >{ 0.5, 0.25 } };
    tax::la::VecNT< 2, double > x0;
    x0( 0 ) = 1.0;
    x0( 1 ) = 2.0;
    State F = create< P, M >( box, x0 );

    // Constant terms = x0_i.
    EXPECT_DOUBLE_EQ( F( 0 )[0], 1.0 );
    EXPECT_DOUBLE_EQ( F( 1 )[0], 2.0 );

    // First-order coefficient w.r.t. ξ_i should be halfWidth_i on row i.
    tax::MultiIndex< M > alpha_x{ 1, 0 };
    tax::MultiIndex< M > alpha_y{ 0, 1 };
    EXPECT_DOUBLE_EQ( F( 0 )[tax::flatIndex< M >( alpha_x )], 0.5 );
    EXPECT_DOUBLE_EQ( F( 1 )[tax::flatIndex< M >( alpha_y )], 0.25 );
}

TEST( AdsDaState, SplitRoundTripPreservesValue )
{
    // Build a state on parent box, split, and verify that evaluating
    // each child at ξ'=0 reproduces the parent at the center of that
    // child's domain (ξ = ±0.5 along the split axis).
    Box< double, M > parent{ tax::la::VecNT< M, double >{ 0.0, 0.0 },
                             tax::la::VecNT< M, double >{ 1.0, 1.0 } };
    tax::la::VecNT< 2, double > x0;
    x0( 0 ) = 0.0;
    x0( 1 ) = 0.0;
    State F = create< P, M >( parent, x0 );

    auto pr = split( F, /*dim=*/0 );
    auto& FL = pr.first;
    auto& FR = pr.second;

    // Row 0 is x_0(ξ) = ξ_0.
    // Parent at ξ_0 = -0.5, ξ_1 = 0   ↔   left  child at ξ'_0 = 0, ξ'_1 = 0.
    // Parent at ξ_0 = +0.5, ξ_1 = 0   ↔   right child at ξ'_0 = 0, ξ'_1 = 0.
    EXPECT_NEAR( evalTe( FL( 0 ), { 0.0, 0.0 } ), -0.5, 1e-12 );
    EXPECT_NEAR( evalTe( FR( 0 ), { 0.0, 0.0 } ), 0.5, 1e-12 );

    // Row 1 (= ξ_1) is unaffected by a dim-0 split.
    EXPECT_NEAR( evalTe( FL( 1 ), { 0.0, 0.7 } ), 0.7, 1e-12 );
    EXPECT_NEAR( evalTe( FR( 1 ), { 0.0, 0.7 } ), 0.7, 1e-12 );
}
