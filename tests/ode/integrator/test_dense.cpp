// tests/ode/testIntegratorDense.cpp
//
// Dense-mode (Dense=true) integration smoke tests. sol(t_query) for
// any t in [t0, tmax] must agree with the closed-form solution
// within the step's local truncation tolerance.

#include <gtest/gtest.h>

#include <tax/la/types.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <tax/ode.hpp>

using tax::ode::IntegratorConfig;

TEST( OdeIntegratorDense, ExpDenseInside )
{
    constexpr int N = 16;
    using State = tax::la::VecNT< 1, double >;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const auto f = []( const auto& x, const auto& /*t*/ ) { return x; };

    tax::ode::Taylor< N, State, tax::ode::controllers::JorbaZou< double >, /*Dense=*/true, decltype( f ) > integ{ f, cfg };

    State x0; x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    // Query at multiple intermediate times.
    for ( const double tq : { 0.07, 0.23, 0.5, 0.83, 0.99 } )
    {
        State x_at_tq = sol( tq );
        EXPECT_NEAR( x_at_tq( 0 ), std::exp( tq ), 1e-10 )
            << "tq=" << tq;
    }

    // Boundaries.
    EXPECT_NEAR( sol( 0.0 )( 0 ), 1.0,              1e-12 );
    EXPECT_NEAR( sol( 1.0 )( 0 ), std::exp( 1.0 ),  1e-11 );
}

TEST( OdeIntegratorDense, OutOfRangeThrows )
{
    constexpr int N = 8;
    using State = tax::la::VecNT< 1, double >;

    IntegratorConfig< double > cfg;
    const auto f = []( const auto& x, const auto& /*t*/ ) { return x; };

    tax::ode::Taylor< N, State, tax::ode::controllers::JorbaZou< double >, /*Dense=*/true, decltype( f ) > integ{ f, cfg };

    State x0; x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 0.5 );

    EXPECT_THROW( sol( -0.1 ), std::out_of_range );
    EXPECT_THROW( sol(  0.6 ), std::out_of_range );
}

TEST( OdeIntegratorDense, TerminatePreservesDenseInvariant )
{
    constexpr int N = 16;
    using State = tax::la::VecNT< 1, double >;

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const auto f = []( const auto& x, const auto& /*t*/ ) { return x; };

    using Stepper = tax::ode::TaylorStepper< N, State >;
    std::vector< tax::ode::Event< Stepper > > events;
    events.emplace_back(
        tax::ode::ZeroCrossing(
            []( const auto& x, const auto& ) { return x( 0 ) - 2.0; },
            tax::ode::Direction::Increasing ),
        tax::ode::Terminate() );

    tax::ode::Taylor< N, State, tax::ode::controllers::JorbaZou< double >, /*Dense=*/true, decltype( f ) > integ{ f, cfg, events };

    State x0; x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 5.0 );

    // Invariant: dense.size() + 1 == t.size().
    ASSERT_EQ( sol.dense.size() + 1, sol.t.size() );
    // Terminated early at x = 2 (which is exp(t) = 2 ⇒ t = ln(2) ≈ 0.693).
    EXPECT_NEAR( sol.t.back(), std::log( 2.0 ), 1e-9 );
    // Dense query just before the event should be valid and ≈ exp(tq).
    const double tq = sol.t.back() - 0.01;
    State x_at_tq = sol( tq );
    EXPECT_NEAR( x_at_tq( 0 ), std::exp( tq ), 1e-10 );
    // Dense query AT the event boundary should also be valid.
    State x_at_event = sol( sol.t.back() );
    EXPECT_NEAR( x_at_event( 0 ), 2.0, 1e-10 );
}
