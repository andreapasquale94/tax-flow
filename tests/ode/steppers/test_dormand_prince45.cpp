// tests/ode/steppers/test_dormand_prince45.cpp
//
// Stepper-level correctness of DormandPrince45Stepper (the classical
// "RK45" 5(4) pair) on two RHS:
//   - dx/dt = x       (analytic exp)
//   - harmonic        (cos t, -sin t)
// Plus the step round-trip assertion at step boundaries and an
// interior point (full-order re-step, not cubic-Hermite).
//
// Tolerances are looser than the higher-order Verner/Feagin steppers:
// a single order-5 step accumulates ~1e-12 local error at h=0.05, so the
// acceptance threshold is relaxed to 1e-8 to keep the step accepted.

#include <gtest/gtest.h>

#include <cmath>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>

using tax::ode::DormandPrince45Stepper;
using tax::ode::IntegratorConfig;
using tax::ode::controllers::I;
using tax::ode::controllers::PI;

TEST( OdeDormandPrince45Stepper, ExponentialOneStep )
{
    using State = tax::la::VecNT< 1, double >;
    DormandPrince45Stepper< State > stepper;

    State x0;
    x0( 0 ) = 1.0;
    const auto f = []( const auto& x, double ) { return x; };

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-8;

    auto r = stepper.step( f, x0, 0.0, 0.05, cfg );

    EXPECT_TRUE( r.accepted );
    EXPECT_NEAR( r.x_new( 0 ), std::exp( 0.05 ), 1e-9 );

    // step at τ=0 reproduces x0.
    auto x_at_t0 = DormandPrince45Stepper< State >::step( f, x0, 0.0, 0.0 );
    EXPECT_NEAR( x_at_t0( 0 ), x0( 0 ), 1e-14 );
    // step at τ=h_used reproduces x_new (full-order re-step).
    auto x_at_t1 = DormandPrince45Stepper< State >::step( f, x0, 0.0, r.h_used );
    EXPECT_NEAR( x_at_t1( 0 ), r.x_new( 0 ), 1e-14 );
    // Interior point τ=h/2: step matches analytic exp(h/2).
    auto x_mid = DormandPrince45Stepper< State >::step( f, x0, 0.0, r.h_used / 2.0 );
    EXPECT_NEAR( x_mid( 0 ), std::exp( r.h_used / 2.0 ), 1e-8 );
}

TEST( OdeDormandPrince45Stepper, HarmonicOneStep )
{
    using State = tax::la::VecNT< 2, double >;
    DormandPrince45Stepper< State > stepper;

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
    cfg.abstol = cfg.reltol = 1e-8;

    auto r = stepper.step( f, x0, 0.0, 0.05, cfg );

    EXPECT_TRUE( r.accepted );
    EXPECT_NEAR( r.x_new( 0 ), std::cos( 0.05 ), 1e-9 );
    EXPECT_NEAR( r.x_new( 1 ), -std::sin( 0.05 ), 1e-9 );
}

TEST( OdeDormandPrince45Stepper, ControllerIVariant )
{
    using State = tax::la::VecNT< 1, double >;
    DormandPrince45Stepper< State, I< double > > stepper;

    State x0;
    x0( 0 ) = 1.0;
    const auto f = []( const auto& x, double ) { return x; };

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-8;
    auto r = stepper.step( f, x0, 0.0, 0.05, cfg );
    EXPECT_TRUE( r.accepted );
    EXPECT_NEAR( r.x_new( 0 ), std::exp( 0.05 ), 1e-9 );
}
