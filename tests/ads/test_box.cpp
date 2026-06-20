// tests/ads/test_box.cpp
//
// Box<T, M> — construction, contains, split, denormalize.

#include <gtest/gtest.h>

#include <tax/ads/box.hpp>
#include <tax/la/types.hpp>

using tax::ads::Box;
using V2 = tax::la::VecNT< 2, double >;

TEST( AdsBox, DefaultCtorZero )
{
    Box< double, 2 > b{};
    EXPECT_EQ( b.center( 0 ), 0.0 );
    EXPECT_EQ( b.center( 1 ), 0.0 );
    EXPECT_EQ( b.halfWidth( 0 ), 0.0 );
    EXPECT_EQ( b.halfWidth( 1 ), 0.0 );
}

TEST( AdsBox, AggregateBraceInit )
{
    Box< double, 2 > b{ V2{ 1.0, 2.0 }, V2{ 0.5, 0.25 } };
    EXPECT_EQ( b.center( 0 ), 1.0 );
    EXPECT_EQ( b.center( 1 ), 2.0 );
    EXPECT_EQ( b.halfWidth( 0 ), 0.5 );
    EXPECT_EQ( b.halfWidth( 1 ), 0.25 );
}

TEST( AdsBox, ContainsInclusiveBoundary )
{
    Box< double, 2 > b{ V2{ 0.0, 0.0 }, V2{ 1.0, 1.0 } };
    EXPECT_TRUE( b.contains( V2{ 0.5, -0.5 } ) );
    EXPECT_TRUE( b.contains( V2{ 1.0, 1.0 } ) );  // on boundary
    EXPECT_TRUE( b.contains( V2{ -1.0, -1.0 } ) );
    EXPECT_FALSE( b.contains( V2{ 1.001, 0.0 } ) );
    EXPECT_FALSE( b.contains( V2{ 0.0, -1.001 } ) );
}

TEST( AdsBox, SplitHalvesOnlyRequestedAxis )
{
    Box< double, 2 > b{ V2{ 0.0, 0.0 }, V2{ 1.0, 2.0 } };
    auto pr = b.split( 0 );
    const auto& L = pr.first;
    const auto& R = pr.second;
    EXPECT_DOUBLE_EQ( L.center( 0 ), -0.5 );
    EXPECT_DOUBLE_EQ( R.center( 0 ), 0.5 );
    EXPECT_DOUBLE_EQ( L.halfWidth( 0 ), 0.5 );
    EXPECT_DOUBLE_EQ( R.halfWidth( 0 ), 0.5 );
    // Untouched axis (dim 1).
    EXPECT_DOUBLE_EQ( L.center( 1 ), 0.0 );
    EXPECT_DOUBLE_EQ( R.center( 1 ), 0.0 );
    EXPECT_DOUBLE_EQ( L.halfWidth( 1 ), 2.0 );
    EXPECT_DOUBLE_EQ( R.halfWidth( 1 ), 2.0 );
}

TEST( AdsBox, Denormalize )
{
    Box< double, 2 > b{ V2{ 1.0, 2.0 }, V2{ 0.5, 0.25 } };
    auto pt = b.denormalize( V2{ 1.0, -1.0 } );
    EXPECT_DOUBLE_EQ( pt( 0 ), 1.5 );
    EXPECT_DOUBLE_EQ( pt( 1 ), 1.75 );
}
