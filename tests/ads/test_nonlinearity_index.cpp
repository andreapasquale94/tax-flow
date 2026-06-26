// tests/ads/test_nonlinearity_index.cpp
//
// jacobianVariationBound, linRowBound, nonlinearityIndex, nliSplitDim.

#include <gtest/gtest.h>

#include <cmath>
#include <tax/ads/detail/nonlinearity_index.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>

using tax::TEn;
using tax::ads::detail::jacobianVariationBound;
using tax::ads::detail::linRowBound;
using tax::ads::detail::nliSplitDim;
using tax::ads::detail::nonlinearityIndex;

// Convenience: create two variables for TEn<N,2> at origin
template < int N >
static auto make_xy()
{
    Eigen::Vector2d p{ 0.0, 0.0 };
    auto v = tax::la::variables< tax::TE< N, 2 > >( p );
    return std::make_pair( v[0], v[1] );
}

TEST( AdsNli, LinRowBoundPureLinear )
{
    auto [x, y] = make_xy< 3 >();
    auto f = 2.0 * x + 3.0 * y;
    EXPECT_DOUBLE_EQ( linRowBound( f ), 5.0 );
}

TEST( AdsNli, JacobianVariationBoundQuadratic )
{
    // f(x, y) = 1 + x + y + 0.5*x*y + 2*x*x
    // ∂f/∂x = 1 + 0.5*y + 4*x → bound: 0.5 + 4 = 4.5
    // ∂f/∂y = 1 + 0.5*x       → bound: 0.5
    auto [x, y] = make_xy< 3 >();
    auto f = 1.0 + x + y + 0.5 * x * y + 2.0 * x * x;
    const auto bnd = jacobianVariationBound( f );
    EXPECT_DOUBLE_EQ( bnd[0], 4.5 );
    EXPECT_DOUBLE_EQ( bnd[1], 0.5 );
}

TEST( AdsNli, NonlinearityIndexLinearIsZero )
{
    using TE = tax::TE< 3, 2 >;
    auto [x, y] = make_xy< 3 >();
    tax::la::VecNT< 2, TE > F;
    F( 0 ) = 2.0 * x + 3.0 * y;
    F( 1 ) = x + y;
    EXPECT_DOUBLE_EQ( nonlinearityIndex( F ), 0.0 );
}

TEST( AdsNli, NonlinearityIndexQuadratic )
{
    using TE = tax::TE< 3, 2 >;
    auto [x, y] = make_xy< 3 >();
    tax::la::VecNT< 2, TE > F;
    F( 0 ) = x + 0.5 * x * x;  // lin = 1, var = 1 → NLI row0 = 1.0
    F( 1 ) = y;                // lin = 1, var = 0 → NLI row1 = 0
    EXPECT_DOUBLE_EQ( nonlinearityIndex( F ), 1.0 );
}

TEST( AdsNli, SplitDimPicksDominantAxis )
{
    using TE = tax::TE< 3, 2 >;
    auto [x, y] = make_xy< 3 >();
    tax::la::VecNT< 2, TE > F;
    F( 0 ) = 3.0 * x * x;  // contributes 6 to dim-0 var
    F( 1 ) = 1.0 * y * y;  // contributes 2 to dim-1 var
    EXPECT_EQ( nliSplitDim( F ), 0 );
}
