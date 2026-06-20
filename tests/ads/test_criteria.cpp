// tests/ads/test_criteria.cpp
//
// TruncationCriterion + NliCriterion — shouldSplit / splitDim decisions
// on crafted DA-valued states.

#include <gtest/gtest.h>

#include <tax/ads/criteria.hpp>
#include <tax/la/types.hpp>
#include <tax/tax.hpp>

using tax::ads::NliCriterion;
using tax::ads::TruncationCriterion;

namespace
{
template < int N >
static auto make_xy()
{
    Eigen::Vector2d p{ 0.0, 0.0 };
    auto v = tax::la::variables< tax::TE< N, 2 > >( p );
    return std::make_pair( v[0], v[1] );
}
}  // namespace

TEST( AdsCriteria, TruncationDoesNotSplitBelowTol )
{
    using TE = tax::TE< 3, 2 >;
    auto [x, y] = make_xy< 3 >();
    tax::la::VecNT< 2, TE > F;
    F( 0 ) = x + y;
    F( 1 ) = x - y;
    TruncationCriterion crit{ /*tol=*/1e-8 };
    EXPECT_FALSE( crit.shouldSplit( F, /*depth=*/0 ) );
}

TEST( AdsCriteria, TruncationSplitsAboveTol )
{
    using TE = tax::TE< 3, 2 >;
    auto [x, y] = make_xy< 3 >();
    tax::la::VecNT< 1, TE > F;
    F( 0 ) = 1.0 * x * x * x + 0.5 * y * y * y;  // degree-3 mass = 1.5
    TruncationCriterion crit{ /*tol=*/1e-3 };
    EXPECT_TRUE( crit.shouldSplit( F, 0 ) );
}

TEST( AdsCriteria, TruncationRespectsMaxDepth )
{
    using TE = tax::TE< 3, 2 >;
    auto [x, y] = make_xy< 3 >();
    tax::la::VecNT< 1, TE > F;
    F( 0 ) = 1.0 * x * x * x;
    TruncationCriterion crit{ /*tol=*/1e-12, /*maxDepth=*/3 };
    EXPECT_TRUE( crit.shouldSplit( F, 2 ) );
    EXPECT_FALSE( crit.shouldSplit( F, 3 ) );  // at the cap → don't split
}

TEST( AdsCriteria, NliBelowTolNoSplit )
{
    using TE = tax::TE< 3, 2 >;
    auto [x, y] = make_xy< 3 >();
    tax::la::VecNT< 2, TE > F;
    F( 0 ) = x + y;
    F( 1 ) = x - y;
    NliCriterion crit{ /*tol=*/0.1 };
    EXPECT_FALSE( crit.shouldSplit( F, 0 ) );
}

TEST( AdsCriteria, NliAboveTolSplitsAtDominantDim )
{
    using TE = tax::TE< 3, 2 >;
    auto [x, y] = make_xy< 3 >();
    tax::la::VecNT< 2, TE > F;
    F( 0 ) = x + 0.5 * x * x;  // big nonlinearity on dim 0
    F( 1 ) = y;
    NliCriterion crit{ /*tol=*/0.1 };
    EXPECT_TRUE( crit.shouldSplit( F, 0 ) );
    EXPECT_EQ( crit.splitDim( F ), 0 );
}
