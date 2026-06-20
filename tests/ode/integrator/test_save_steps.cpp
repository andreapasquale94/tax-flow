// tests/ode/integrator/test_save_steps.cpp
//
// Tests for IntegratorConfig::save_steps flag.
//   1. SaveStepsTrueKeepsGrid      — full grid recorded (default).
//   2. SaveStepsFalseKeepsEndpointsOnly — only initial + final stored.
//   3. EventsRecordedRegardlessOfSaveSteps — events pushed to sol.events
//      regardless of save_steps value.

#include <gtest/gtest.h>

#include <cmath>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <vector>

using tax::ode::Direction;
using tax::ode::Event;
using tax::ode::IntegratorConfig;
using tax::ode::Record;
using tax::ode::TaylorStepper;
using tax::ode::ZeroCrossing;

// x' = x  ⟹  x(t) = exp(t)
static const auto exp_rhs = []( const auto& x, const auto& ) { return x; };

TEST( OdeSaveSteps, SaveStepsTrueKeepsGrid )
{
    constexpr int N = 16;
    using State = tax::la::VecNT< 1, double >;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-10;
    // save_steps defaults to true

    tax::ode::Taylor< N, State, tax::ode::controllers::JorbaZou< double >, decltype( exp_rhs ) >
        integ{ exp_rhs, cfg };

    State x0;
    x0( 0 ) = 1.0;
    const double t0 = 0.0, tmax = 3.0;
    auto sol = integ.integrate( x0, t0, tmax );

    // Adaptive integration over [0, 3] takes more than one step.
    EXPECT_GT( sol.size(), 2u );
    EXPECT_DOUBLE_EQ( sol.t.front(), t0 );
    EXPECT_DOUBLE_EQ( sol.t.back(), tmax );
}

TEST( OdeSaveSteps, SaveStepsFalseKeepsEndpointsOnly )
{
    constexpr int N = 16;
    using State = tax::la::VecNT< 1, double >;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-10;
    cfg.save_steps = false;

    tax::ode::Taylor< N, State, tax::ode::controllers::JorbaZou< double >, decltype( exp_rhs ) >
        integ{ exp_rhs, cfg };

    State x0;
    x0( 0 ) = 1.0;
    const double t0 = 0.0, tmax = 3.0;
    auto sol = integ.integrate( x0, t0, tmax );

    ASSERT_EQ( sol.size(), 2u );
    EXPECT_DOUBLE_EQ( sol.t[0], t0 );
    EXPECT_DOUBLE_EQ( sol.t[1], tmax );
    // x(tmax) = exp(tmax); tolerance matched to cfg.abstol/reltol.
    EXPECT_NEAR( sol.x[1]( 0 ), std::exp( tmax ), 1e-9 );
}

TEST( OdeSaveSteps, EventsRecordedRegardlessOfSaveSteps )
{
    // x' = x, x(0) = 1.  x(t) = exp(t) crosses 2 at t = ln(2) ≈ 0.693.
    constexpr int N = 16;
    using State = tax::la::VecNT< 1, double >;

    // Guard function: x - 2 (increasing zero crossing at t = ln 2).
    auto guard = []( const auto& x, const auto& ) { return x( 0 ) - 2.0; };

    auto make_events = []( auto& g ) {
        using Stepper = TaylorStepper< N, State >;
        std::vector< Event< Stepper > > evs;
        evs.emplace_back( ZeroCrossing( g, Direction::Increasing ), Record( "x_eq_2" ) );
        return evs;
    };

    auto run = [&]( bool save_steps_flag ) {
        IntegratorConfig< double > cfg;
        cfg.abstol = cfg.reltol = 1e-12;
        cfg.save_steps = save_steps_flag;

        auto events = make_events( guard );

        tax::ode::Taylor< N, State, tax::ode::controllers::JorbaZou< double >, decltype( exp_rhs ) >
            integ{ exp_rhs, cfg, std::move( events ) };

        State x0;
        x0( 0 ) = 1.0;
        return integ.integrate( x0, 0.0, 3.0 );
    };

    auto sol_with = run( true );
    auto sol_without = run( false );

    // Exactly one event recorded in both cases.
    ASSERT_EQ( sol_with.events.size(), 1u );
    ASSERT_EQ( sol_without.events.size(), 1u );

    // Event time ≈ ln(2).
    EXPECT_NEAR( sol_with.events[0].t, std::log( 2.0 ), 1e-9 );
    EXPECT_NEAR( sol_without.events[0].t, std::log( 2.0 ), 1e-9 );

    // Recorded states agree between the two runs.
    EXPECT_NEAR( sol_with.events[0].x( 0 ), sol_without.events[0].x( 0 ), 1e-12 );
}
