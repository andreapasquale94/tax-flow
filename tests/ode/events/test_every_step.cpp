// tests/ode/testEventsEveryStep.cpp
//
// EveryStep trigger paired with Continue / Custom actions.
// Verifies: every accepted step fires the trigger exactly once;
// Custom can stop the integration by returning Terminate.

#include <gtest/gtest.h>

#include <cmath>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <vector>

using tax::ode::ControlFlow;
using tax::ode::Custom;
using tax::ode::Event;
using tax::ode::EveryStep;
using tax::ode::IntegratorConfig;
using tax::ode::TaylorStepper;

TEST( OdeEventsEveryStep, FiresOncePerStep )
{
    constexpr int N = 12;
    using State = tax::la::VecNT< 1, double >;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const auto f = []( const auto& x, const auto& ) { return x; };

    int counter = 0;
    using Stepper = TaylorStepper< N, State >;
    std::vector< Event< Stepper > > events;
    events.emplace_back( EveryStep(), Custom( [&counter]( const auto&, double, auto& ) {
                             ++counter;
                             return ControlFlow::Continue;
                         } ) );

    tax::ode::Taylor< N, State, tax::ode::controllers::JorbaZou< double >, decltype( f ) > integ{
        f, cfg, events };
    State x0;
    x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    // Counter should equal the number of accepted steps == sol.size() - 1.
    EXPECT_EQ( static_cast< std::size_t >( counter ), sol.size() - 1 );
    EXPECT_GE( counter, 1 );
}

TEST( OdeEventsEveryStep, CustomCanTerminate )
{
    constexpr int N = 12;
    using State = tax::la::VecNT< 1, double >;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const auto f = []( const auto& x, const auto& ) { return x; };

    using Stepper = TaylorStepper< N, State >;
    std::vector< Event< Stepper > > events;
    events.emplace_back( EveryStep(), Custom( []( const auto& ctx, double, auto& ) {
                             return ( ctx.t_old + ctx.h_used > 0.3 ) ? ControlFlow::Terminate
                                                                     : ControlFlow::Continue;
                         } ) );

    tax::ode::Taylor< N, State, tax::ode::controllers::JorbaZou< double >, decltype( f ) > integ{
        f, cfg, events };
    State x0;
    x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    EXPECT_LT( sol.t.back(), 1.0 );
    EXPECT_GE( sol.t.back(), 0.3 );
}
