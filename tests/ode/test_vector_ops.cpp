// tests/ode/testVectorOps.cpp
//
// Round-trip the VectorOps<S> trait on the four state shapes we ship:
//   - scalar double
//   - scalar TEn<P,M>
//   - tax::la::VecNT< D, double >
//   - Eigen::Matrix<TEn<P,M>, D, 1>
// For each shape, verify:
//   norm(x)             == infinity-norm(x)
//   scale_assign(y,a,x) => y == a*x
//   axpy(y,a,x)         => y_new == y_old + a*x

#include <gtest/gtest.h>

#include <cmath>
#include <tax/la/types.hpp>
#include <tax/ode/vector_ops.hpp>
#include <tax/tax.hpp>

using tax::ode::VectorOps;

TEST( OdeVectorOps, ScalarDouble )
{
    double y = 0.0;
    VectorOps< double >::scale_assign( y, 2.5, 3.0 );
    EXPECT_DOUBLE_EQ( y, 7.5 );

    VectorOps< double >::axpy( y, -1.0, 2.0 );
    EXPECT_DOUBLE_EQ( y, 5.5 );

    EXPECT_DOUBLE_EQ( VectorOps< double >::norm( -4.2 ), 4.2 );
}

TEST( OdeVectorOps, ScalarTaylorExpansion )
{
    using DA = tax::TEn< 2, 2 >;
    DA x = DA::variable( 0.5, 0 );  // coeff[0]=0.5, coeff[var0]=1.0
    DA y{};                         // zero
    VectorOps< DA >::scale_assign( y, 3.0, x );

    EXPECT_DOUBLE_EQ( y[0], 1.5 );                        // 3.0 * 0.5
    EXPECT_DOUBLE_EQ( VectorOps< DA >::norm( y ), 3.0 );  // 3.0 * 1.0 from var0

    VectorOps< DA >::axpy( y, -1.0, x );
    EXPECT_DOUBLE_EQ( y[0], 1.5 - 0.5 );
}

TEST( OdeVectorOps, EigenVectorDouble )
{
    using V = tax::la::VecNT< 3, double >;
    V x;
    x << 1.0, -2.0, 0.5;
    V y = V::Zero();

    VectorOps< V >::scale_assign( y, 2.0, x );
    EXPECT_DOUBLE_EQ( y( 0 ), 2.0 );
    EXPECT_DOUBLE_EQ( y( 1 ), -4.0 );
    EXPECT_DOUBLE_EQ( y( 2 ), 1.0 );

    EXPECT_DOUBLE_EQ( VectorOps< V >::norm( y ), 4.0 );

    VectorOps< V >::axpy( y, -0.5, x );
    EXPECT_DOUBLE_EQ( y( 0 ), 2.0 - 0.5 );
    EXPECT_DOUBLE_EQ( y( 1 ), -4.0 + 1.0 );
    EXPECT_DOUBLE_EQ( y( 2 ), 1.0 - 0.25 );
}

TEST( OdeVectorOps, EigenVectorOfTaylorExpansion )
{
    using DA = tax::TEn< 2, 2 >;
    using V = tax::la::VecNT< 2, DA >;

    V x;
    x( 0 ) = DA::variable( 1.0, 0 );   // coeff[0]=1.0, coeff[e_0]=1.0
    x( 1 ) = DA::variable( -0.5, 1 );  // coeff[0]=-0.5, coeff[e_1]=1.0
    V y = V::Zero();

    VectorOps< V >::scale_assign( y, 2.0, x );
    EXPECT_DOUBLE_EQ( y( 0 )[0], 2.0 );   // 2*1.0
    EXPECT_DOUBLE_EQ( y( 1 )[0], -1.0 );  // 2*(-0.5)

    // sup-norm: max of (|2*1.0|, |2*1.0|, |2*(-0.5)|, |2*1.0|) = 2.0
    EXPECT_DOUBLE_EQ( VectorOps< V >::norm( y ), 2.0 );

    VectorOps< V >::axpy( y, -1.0, x );
    EXPECT_DOUBLE_EQ( y( 0 )[0], 2.0 - 1.0 );
    EXPECT_DOUBLE_EQ( y( 1 )[0], -1.0 + 0.5 );
}
