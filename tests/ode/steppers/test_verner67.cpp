// tests/ode/steppers/test_verner67.cpp
//
// Stepper-level correctness of Verner67Stepper (Verner's "most efficient"
// 7(6) pair) on two RHS:
//   - dx/dt = x       (analytic exp)
//   - harmonic        (cos t, -sin t)
// Plus the step round-trip assertion at step boundaries and an
// interior point (full-order re-step, not cubic-Hermite).

#include <gtest/gtest.h>

#include <cmath>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>

using tax::ode::IntegratorConfig;
using tax::ode::Verner67Stepper;
using tax::ode::controllers::I;
using tax::ode::controllers::PI;

TEST( OdeVerner67Stepper, ExponentialOneStep )
{
    using State = tax::la::VecNT< 1, double >;
    Verner67Stepper< State > stepper;

    State x0;
    x0( 0 ) = 1.0;
    const auto f = []( const auto& x, double ) { return x; };

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    auto r = stepper.step( f, x0, 0.0, 0.05, cfg );

    EXPECT_TRUE( r.accepted );
    EXPECT_NEAR( r.x_new( 0 ), std::exp( 0.05 ), 1e-12 );

    // step at τ=0 reproduces x0.
    auto x_at_t0 = Verner67Stepper< State >::step( f, x0, 0.0, 0.0 );
    EXPECT_NEAR( x_at_t0( 0 ), x0( 0 ), 1e-14 );
    // step at τ=h_used reproduces x_new (full-order re-step).
    auto x_at_t1 = Verner67Stepper< State >::step( f, x0, 0.0, r.h_used );
    EXPECT_NEAR( x_at_t1( 0 ), r.x_new( 0 ), 1e-14 );
    // Interior point τ=h/2: step matches analytic exp(h/2).
    auto x_mid = Verner67Stepper< State >::step( f, x0, 0.0, r.h_used / 2.0 );
    EXPECT_NEAR( x_mid( 0 ), std::exp( r.h_used / 2.0 ), 1e-10 );
}

TEST( OdeVerner67Stepper, HarmonicOneStep )
{
    using State = tax::la::VecNT< 2, double >;
    Verner67Stepper< State > stepper;

    State x0;
    x0( 0 ) = 1.0;
    x0( 1 ) = 0.0;
    const auto f = []( const auto& x, double ) {
        State out;
        out( 0 ) = x( 1 );
        out( 1 ) = -x( 0 );
        return out;
    };

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    auto r = stepper.step( f, x0, 0.0, 0.05, cfg );

    EXPECT_TRUE( r.accepted );
    EXPECT_NEAR( r.x_new( 0 ), std::cos( 0.05 ), 1e-12 );
    EXPECT_NEAR( r.x_new( 1 ), -std::sin( 0.05 ), 1e-12 );
}

TEST( OdeVerner67Stepper, ControllerIVariant )
{
    using State = tax::la::VecNT< 1, double >;
    Verner67Stepper< State, I< double > > stepper;

    State x0;
    x0( 0 ) = 1.0;
    const auto f = []( const auto& x, double ) { return x; };

    IntegratorConfig< double > cfg;
    auto r = stepper.step( f, x0, 0.0, 0.05, cfg );
    EXPECT_TRUE( r.accepted );
    EXPECT_NEAR( r.x_new( 0 ), std::exp( 0.05 ), 1e-12 );
}
