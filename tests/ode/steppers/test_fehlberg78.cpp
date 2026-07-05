// tests/ode/testFehlberg78Stepper.cpp
//
// Stepper-level correctness of Fehlberg78Stepper on three RHS:
//   - dx/dt = x       (analytic exp)
//   - harmonic        (cos t, -sin t)
// Plus the step round-trip assertion at step boundaries and an
// interior point (full-order re-step, not cubic-Hermite).
//
// Tolerances are set slightly looser than Verner78 to accommodate
// the documented "Fehlberg coincidence" (embedded estimator can be
// zero on certain steps, leading to occasional large steps that
// accumulate slightly more error).

#include <gtest/gtest.h>

#include <cmath>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>

using tax::ode::Fehlberg78Stepper;
using tax::ode::IntegratorConfig;
using tax::ode::controllers::I;
using tax::ode::controllers::PI;

TEST( OdeFehlberg78Stepper, ExponentialOneStep )
{
    using State = tax::la::VecNT< 1, double >;
    Fehlberg78Stepper< State > stepper;

    State x0;
    x0( 0 ) = 1.0;
    const auto f = []( const auto& x, double ) { return x; };

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    auto r = stepper.step( f, x0, 0.0, 0.1, cfg );

    EXPECT_TRUE( r.accepted );
    EXPECT_NEAR( r.x_new( 0 ), std::exp( 0.1 ), 1e-11 );

    // step at τ=0 reproduces x0.
    auto x_at_t0 = Fehlberg78Stepper< State >::step( f, x0, 0.0, 0.0 );
    EXPECT_NEAR( x_at_t0( 0 ), x0( 0 ), 1e-14 );
    // step at τ=h_used reproduces x_new (full-order re-step).
    auto x_at_t1 = Fehlberg78Stepper< State >::step( f, x0, 0.0, r.h_used );
    EXPECT_NEAR( x_at_t1( 0 ), r.x_new( 0 ), 1e-14 );
    // Interior point τ=h/2: step matches analytic exp(h/2).
    auto x_mid = Fehlberg78Stepper< State >::step( f, x0, 0.0, r.h_used / 2.0 );
    EXPECT_NEAR( x_mid( 0 ), std::exp( r.h_used / 2.0 ), 1e-10 );
}

TEST( OdeFehlberg78Stepper, HarmonicOneStep )
{
    using State = tax::la::VecNT< 2, double >;
    Fehlberg78Stepper< State > stepper;

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
    EXPECT_NEAR( r.x_new( 0 ), std::cos( 0.05 ), 1e-10 );
    EXPECT_NEAR( r.x_new( 1 ), -std::sin( 0.05 ), 1e-10 );
}

// Regression (O1): the method propagates at order 8 (local extrapolation) with
// an embedded order-7 estimator. The metadata (which drives the step-controller
// exponent 1/(order_emb+1) = 1/8) must reflect that, not the swapped 7/8.
TEST( OdeFehlberg78Stepper, OrderMetadataIsEightSevenPair )
{
    using State = tax::la::VecNT< 1, double >;
    EXPECT_EQ( Fehlberg78Stepper< State >::order_v, 8 );
    EXPECT_EQ( Fehlberg78Stepper< State >::order_emb_v, 7 );
}

TEST( OdeFehlberg78Stepper, ControllerIVariant )
{
    using State = tax::la::VecNT< 1, double >;
    Fehlberg78Stepper< State, I< double > > stepper;

    State x0;
    x0( 0 ) = 1.0;
    const auto f = []( const auto& x, double ) { return x; };

    IntegratorConfig< double > cfg;
    auto r = stepper.step( f, x0, 0.0, 0.05, cfg );
    EXPECT_TRUE( r.accepted );
    EXPECT_NEAR( r.x_new( 0 ), std::exp( 0.05 ), 1e-11 );
}
