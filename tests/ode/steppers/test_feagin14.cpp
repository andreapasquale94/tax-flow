// tests/ode/testFeagin14Stepper.cpp
//
// Stepper-level correctness of Feagin14Stepper on three RHS:
//   - dx/dt = x       (analytic exp)
//   - harmonic        (cos t, -sin t)
//   - exp single-step under the I controller
// Plus step round-trip and interior point check.
// Tolerances are tightened to 1e-13 to reflect the order-14
// propagation accuracy.

#include <gtest/gtest.h>

#include <cmath>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>

using tax::ode::Feagin14Stepper;
using tax::ode::IntegratorConfig;
using tax::ode::controllers::I;
using tax::ode::controllers::PI;

TEST( OdeFeagin14Stepper, ExponentialOneStep )
{
    using State = tax::la::VecNT< 1, double >;
    Feagin14Stepper< State > stepper;

    State x0;
    x0( 0 ) = 1.0;
    const auto f = []( const auto& x, double ) { return x; };

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    auto r = stepper.step( f, x0, 0.0, 0.1, cfg );

    EXPECT_TRUE( r.accepted );
    EXPECT_NEAR( r.x_new( 0 ), std::exp( 0.1 ), 1e-13 );

    // step at τ=0 reproduces x0.
    auto x_at_t0 = Feagin14Stepper< State >::step( f, x0, 0.0, 0.0 );
    EXPECT_NEAR( x_at_t0( 0 ), x0( 0 ), 1e-14 );
    // step at τ=h_used reproduces x_new (full-order re-step).
    auto x_at_t1 = Feagin14Stepper< State >::step( f, x0, 0.0, r.h_used );
    EXPECT_NEAR( x_at_t1( 0 ), r.x_new( 0 ), 1e-14 );
    // Interior point τ=h/2: step matches analytic exp(h/2).
    auto x_mid = Feagin14Stepper< State >::step( f, x0, 0.0, r.h_used / 2.0 );
    EXPECT_NEAR( x_mid( 0 ), std::exp( r.h_used / 2.0 ), 1e-10 );
}

TEST( OdeFeagin14Stepper, HarmonicOneStep )
{
    using State = tax::la::VecNT< 2, double >;
    Feagin14Stepper< State > stepper;

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
    EXPECT_NEAR( r.x_new( 0 ), std::cos( 0.05 ), 1e-13 );
    EXPECT_NEAR( r.x_new( 1 ), -std::sin( 0.05 ), 1e-13 );
}

TEST( OdeFeagin14Stepper, ControllerIVariant )
{
    using State = tax::la::VecNT< 1, double >;
    Feagin14Stepper< State, I< double > > stepper;

    State x0;
    x0( 0 ) = 1.0;
    const auto f = []( const auto& x, double ) { return x; };

    IntegratorConfig< double > cfg;
    auto r = stepper.step( f, x0, 0.0, 0.05, cfg );
    EXPECT_TRUE( r.accepted );
    EXPECT_NEAR( r.x_new( 0 ), std::exp( 0.05 ), 1e-13 );
}
