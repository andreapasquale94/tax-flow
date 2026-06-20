// tests/ode/testEventsZeroCrossing.cpp
//
// RootFindingEvent semantics on TaylorStepper and RK steppers. Five scenarios:
//   1. Harmonic oscillator, terminate when x crosses 0 going down.
//   2. Same RHS, record both apoapsis (v=0 down) and periapsis (v=0 up).
//   3. Multiple events: record event at t=1, terminate at t=2.
//   4. Sanity: no event registered ⇒ integration runs to tmax.
//   5. Terminate across all RK methods.

#include <gtest/gtest.h>

#include <cmath>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <vector>

using tax::ode::Direction;
using tax::ode::IntegratorConfig;

TEST( OdeEventsZeroCrossing, HarmonicTerminateAtZero )
{
    constexpr int N = 16;
    using State = tax::la::VecNT< 2, double >;

    const auto f = []( const auto& x, const auto& ) {
        using S = std::decay_t< decltype( x ) >;
        S out;
        out( 0 ) = x( 1 );
        out( 1 ) = -x( 0 );
        return out;
    };

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    tax::ode::Taylor< N, State, tax::ode::controllers::JorbaZou< double >, decltype( f ) > integ{
        f, cfg };
    integ.addRootFindingEvent( []( const auto& x, const auto& ) { return x( 0 ); },
                               Direction::Decreasing, "x_zero", /*terminal=*/true );
    State x0;
    x0( 0 ) = 1.0;
    x0( 1 ) = 0.0;
    // x(t) = cos t, so x(0)=1, decreasing through 0 at t = π/2.
    auto sol = integ.integrate( x0, 0.0, 5.0 );

    EXPECT_NEAR( sol.t.back(), M_PI / 2, 1e-9 );
    EXPECT_LT( sol.t.back(), 5.0 );  // ended early
}

TEST( OdeEventsZeroCrossing, HarmonicVZeroRecord )
{
    constexpr int N = 16;
    using State = tax::la::VecNT< 2, double >;

    const auto f = []( const auto& x, const auto& ) {
        using S = std::decay_t< decltype( x ) >;
        S out;
        out( 0 ) = x( 1 );
        out( 1 ) = -x( 0 );
        return out;
    };

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    tax::ode::Taylor< N, State, tax::ode::controllers::JorbaZou< double >, decltype( f ) > integ{
        f, cfg };
    integ.addRootFindingEvent( []( const auto& x, const auto& ) { return x( 1 ); }, Direction::Any,
                               "v_zero", /*terminal=*/false );
    State x0;
    x0( 0 ) = 1.0;
    x0( 1 ) = 0.0;
    // v(t) = -sin t : zero at t = 0 (boundary), π, 2π. Over (0, 2π]
    // we expect events at π and 2π (the t=0 boundary is filtered by
    // the strict sign-change requirement inside findRoot).
    auto sol = integ.integrate( x0, 0.0, 2 * M_PI );

    EXPECT_GE( sol.events.size(), 1u );
    EXPECT_LE( sol.events.size(), 3u );
    for ( const auto& e : sol.events )
    {
        EXPECT_GE( e.t, 0.0 );
        EXPECT_LE( e.t, 2 * M_PI + 1e-9 );
        // The event time is found via Brent on flow(τ). Record uses
        // flow(τ) for the recorded state — accurate to method order.
        EXPECT_NEAR( std::abs( e.x( 1 ) ), 0.0, 1e-8 );
    }
}

// When several events fire in one step, termination must truncate at the
// *terminating* event's time — not the earliest fired event — and events
// scheduled after the terminating one must not run.
TEST( OdeEventsZeroCrossing, TerminationUsesTerminatingEventTime )
{
    constexpr int N = 8;
    using State = tax::la::VecNT< 1, double >;

    // x' = 1  =>  x(t) = t  (exact for any Taylor order).
    const auto f = []( const auto& x, const auto& ) {
        using S = std::decay_t< decltype( x ) >;
        S o;
        o( 0 ) = typename S::Scalar( 1.0 );
        return o;
    };

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    cfg.initial_step = 10.0;  // force a single step clamped to [0, tmax]

    tax::ode::Taylor< N, State, tax::ode::controllers::JorbaZou< double >, decltype( f ) > integ{
        f, cfg };
    // Under registration-order semantics, the earlier-registered record event (t=1)
    // runs before the terminating one (t=2) within the step, so the t=1 record is
    // present and truncation is at t=2.
    // Earlier crossing at t = 1: record, non-terminal (registered first).
    integ.addRootFindingEvent( []( const auto& x, const auto& ) { return x( 0 ) - 1.0; },
                               Direction::Increasing, "g1", /*terminal=*/false );
    // Later crossing at t = 2: terminate (registered second).
    integ.addRootFindingEvent( []( const auto& x, const auto& ) { return x( 0 ) - 2.0; },
                               Direction::Increasing, "g2", /*terminal=*/true );
    State x0;
    x0( 0 ) = 0.0;
    auto sol = integ.integrate( x0, 0.0, 3.0 );

    // Terminate at the terminating event (t = 2), not the earlier Record (t = 1).
    EXPECT_NEAR( sol.t.back(), 2.0, 1e-9 );
    EXPECT_NEAR( sol.x.back()( 0 ), 2.0, 1e-9 );

    // The earlier Record event did happen (at t = 1).
    ASSERT_GE( sol.events.size(), 1u );
    EXPECT_NEAR( sol.events.front().t, 1.0, 1e-9 );
}

TEST( OdeEventsZeroCrossing, EmptyEventListRunsToTmax )
{
    constexpr int N = 12;
    using State = tax::la::VecNT< 1, double >;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const auto f = []( const auto& x, const auto& ) { return x; };

    tax::ode::Taylor< N, State, tax::ode::controllers::JorbaZou< double >, decltype( f ) > integ{
        f, cfg };
    State x0;
    x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    EXPECT_DOUBLE_EQ( sol.t.back(), 1.0 );
    EXPECT_NEAR( sol.x.back()( 0 ), std::exp( 1.0 ), 1e-10 );
    EXPECT_EQ( sol.events.size(), 0u );
}

TEST( OdeEventsZeroCrossing, HarmonicTerminateAcrossAllMethods )
{
    using State = tax::la::VecNT< 2, double >;
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const auto f = []( const auto& x, const auto& ) {
        State out;
        out( 0 ) = x( 1 );
        out( 1 ) = -x( 0 );
        return out;
    };

    State x0;
    x0( 0 ) = 1.0;
    x0( 1 ) = 0.0;
    const double tmax = 5.0;
    const double t_expected = M_PI / 2;
    // All RK methods use Brent on full-order re-steps (step). Accuracy
    // is determined by the step size at event time and Brent convergence;
    // 1e-6 covers all five methods comfortably.
    const double tol = 1e-6;

    {
        tax::ode::Verner78< State > integ{ f, cfg };
        integ.addRootFindingEvent( []( const auto& x, const auto& ) { return x( 0 ); },
                                   Direction::Decreasing, "x_zero", /*terminal=*/true );
        auto sol = integ.integrate( x0, 0.0, tmax );
        EXPECT_NEAR( sol.t.back(), t_expected, tol ) << "Verner78";
    }

    {
        tax::ode::Verner89< State > integ{ f, cfg };
        integ.addRootFindingEvent( []( const auto& x, const auto& ) { return x( 0 ); },
                                   Direction::Decreasing, "x_zero", /*terminal=*/true );
        auto sol = integ.integrate( x0, 0.0, tmax );
        EXPECT_NEAR( sol.t.back(), t_expected, tol ) << "Verner89";
    }

    {
        tax::ode::Fehlberg78< State > integ{ f, cfg };
        integ.addRootFindingEvent( []( const auto& x, const auto& ) { return x( 0 ); },
                                   Direction::Decreasing, "x_zero", /*terminal=*/true );
        auto sol = integ.integrate( x0, 0.0, tmax );
        EXPECT_NEAR( sol.t.back(), t_expected, tol ) << "Fehlberg78";
    }

    {
        tax::ode::Feagin12< State > integ{ f, cfg };
        integ.addRootFindingEvent( []( const auto& x, const auto& ) { return x( 0 ); },
                                   Direction::Decreasing, "x_zero", /*terminal=*/true );
        auto sol = integ.integrate( x0, 0.0, tmax );
        EXPECT_NEAR( sol.t.back(), t_expected, tol ) << "Feagin12";
    }

    {
        tax::ode::Feagin14< State > integ{ f, cfg };
        integ.addRootFindingEvent( []( const auto& x, const auto& ) { return x( 0 ); },
                                   Direction::Decreasing, "x_zero", /*terminal=*/true );
        auto sol = integ.integrate( x0, 0.0, tmax );
        EXPECT_NEAR( sol.t.back(), t_expected, tol ) << "Feagin14";
    }
}
