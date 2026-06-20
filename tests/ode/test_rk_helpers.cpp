// tests/ode/testRKHelpers.cpp
//
// Direct unit tests for the shared adaptive_rk_step driver.
// Foundational for the five RK steppers.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <tax/la/types.hpp>
#include <tax/ode/detail/adaptive_rk_step.hpp>

namespace
{

struct RK4Tab
{
    static constexpr int n_stages = 4;
    static constexpr int order = 4;
    static constexpr int order_emb = 4;
    static constexpr bool fsal = false;

    static constexpr std::array< double, 4 > c{ 0.0, 0.5, 0.5, 1.0 };
    static constexpr std::array< double, 6 > a{ 0.5, 0.0, 0.5, 0.0, 0.0, 1.0 };
    static constexpr std::array< double, 4 > b{ 1.0 / 6, 1.0 / 3, 1.0 / 3, 1.0 / 6 };
    static constexpr std::array< double, 4 > b_emb = b;
};

}  // namespace

TEST( OdeRKHelpers, RK4OneStepOnExp )
{
    using State = tax::la::VecNT< 1, double >;
    State x;
    x( 0 ) = 1.0;

    auto f = []( const State& y, double ) { return y; };

    tax::ode::detail::RKStepData< State, 4 > stages;
    auto out = tax::ode::detail::adaptive_rk_step< RK4Tab >( f, x, 0.0, 0.1, stages );

    EXPECT_NEAR( out.x_new( 0 ), std::exp( 0.1 ), 1e-7 );
    EXPECT_DOUBLE_EQ( out.err_norm, 0.0 );
}
