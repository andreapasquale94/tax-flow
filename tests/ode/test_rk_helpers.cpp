// tests/ode/testRKHelpers.cpp
//
// Direct unit tests for the shared adaptive_rk_step driver and the
// cubic-Hermite interpolator. Both are foundational for the five
// Plan B RK steppers.

#include <gtest/gtest.h>

#include <tax/la/types.hpp>
#include <array>
#include <cmath>

#include <tax/ode/detail/adaptive_rk_step.hpp>
#include <tax/ode/detail/hermite_interp.hpp>

namespace
{

struct RK4Tab
{
    static constexpr int n_stages   = 4;
    static constexpr int order      = 4;
    static constexpr int order_emb  = 4;
    static constexpr bool fsal      = false;

    static constexpr std::array< double, 4 > c{ 0.0, 0.5, 0.5, 1.0 };
    static constexpr std::array< double, 6 > a{
        0.5,
        0.0, 0.5,
        0.0, 0.0, 1.0
    };
    static constexpr std::array< double, 4 > b{ 1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6 };
    static constexpr std::array< double, 4 > b_emb = b;
};

}  // namespace

TEST( OdeRKHelpers, RK4OneStepOnExp )
{
    using State = tax::la::VecNT< 1, double >;
    State x; x( 0 ) = 1.0;

    auto f = []( const State& y, double ) { return y; };

    tax::ode::detail::RKStepData< State, 4 > stages;
    auto out = tax::ode::detail::adaptive_rk_step< RK4Tab >( f, x, 0.0, 0.1, stages );

    EXPECT_NEAR( out.x_new( 0 ), std::exp( 0.1 ), 1e-7 );
    EXPECT_DOUBLE_EQ( out.err_norm, 0.0 );
}

TEST( OdeRKHelpers, HermiteReproducesBoundaries )
{
    using State = tax::la::VecNT< 2, double >;
    const double h = 1.0;
    State x0; x0( 0 ) = 0.5; x0( 1 ) = -1.0;
    State x1; x1( 0 ) = 0.7; x1( 1 ) = -0.3;
    State f0; f0( 0 ) = 0.2; f0( 1 ) =  0.8;
    State f1; f1( 0 ) = 0.1; f1( 1 ) = -0.4;

    const State at_t0 = tax::ode::detail::hermite_interp( x0, x1, f0, f1, h, 0.0 );
    const State at_t1 = tax::ode::detail::hermite_interp( x0, x1, f0, f1, h, h );
    EXPECT_NEAR( ( at_t0 - x0 ).norm(), 0.0, 1e-14 );
    EXPECT_NEAR( ( at_t1 - x1 ).norm(), 0.0, 1e-14 );
}
