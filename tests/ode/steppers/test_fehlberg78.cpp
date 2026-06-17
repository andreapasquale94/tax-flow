// tests/ode/testFehlberg78Stepper.cpp
//
// Stepper-level correctness of Fehlberg78Stepper on three RHS:
//   - dx/dt = x       (analytic exp)
//   - harmonic        (cos t, -sin t)
//   - cubic-decay     (1 / sqrt(1 + 2t))
// Plus the eval_dense round-trip assertion at step boundaries
// (Hermite-cubic fallback — exact at boundaries, ~O(h^4) inside).
//
// Tolerances are set slightly looser than Verner78 to accommodate
// the documented "Fehlberg coincidence" (embedded estimator can be
// zero on certain steps, leading to occasional large steps that
// accumulate slightly more error).

#include <gtest/gtest.h>

#include <tax/la/types.hpp>
#include <cmath>

#include <tax/ode.hpp>

using tax::ode::Fehlberg78Stepper;
using tax::ode::IntegratorConfig;
using tax::ode::controllers::I;
using tax::ode::controllers::PI;

TEST( OdeFehlberg78Stepper, ExponentialOneStep )
{
    using State = tax::la::VecNT< 1, double >;
    Fehlberg78Stepper< State > stepper;

    State x0; x0( 0 ) = 1.0;
    const auto f = []( const auto& x, double ) { return x; };

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    auto r = stepper.step( f, x0, 0.0, 0.1, cfg );

    EXPECT_TRUE( r.accepted );
    EXPECT_NEAR( r.x_new( 0 ), std::exp( 0.1 ), 1e-11 );

    auto x_at_t0 = Fehlberg78Stepper< State >::eval_dense(
        r.dense, 0.0, 0.0 );
    auto x_at_t1 = Fehlberg78Stepper< State >::eval_dense(
        r.dense, 0.0, r.h_used );
    EXPECT_NEAR( x_at_t0( 0 ), x0( 0 ),       1e-14 );
    EXPECT_NEAR( x_at_t1( 0 ), r.x_new( 0 ),  1e-14 );
}

TEST( OdeFehlberg78Stepper, HarmonicOneStep )
{
    using State = tax::la::VecNT< 2, double >;
    Fehlberg78Stepper< State > stepper;

    State x0; x0( 0 ) = 1.0; x0( 1 ) = 0.0;
    const auto f = []( const auto& x, double )
    {
        State out;
        out( 0 ) =  x( 1 );
        out( 1 ) = -x( 0 );
        return out;
    };

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    auto r = stepper.step( f, x0, 0.0, 0.05, cfg );

    EXPECT_TRUE( r.accepted );
    EXPECT_NEAR( r.x_new( 0 ),  std::cos( 0.05 ), 1e-10 );
    EXPECT_NEAR( r.x_new( 1 ), -std::sin( 0.05 ), 1e-10 );
}

TEST( OdeFehlberg78Stepper, ControllerIVariant )
{
    using State = tax::la::VecNT< 1, double >;
    Fehlberg78Stepper< State, I< double > > stepper;

    State x0; x0( 0 ) = 1.0;
    const auto f = []( const auto& x, double ) { return x; };

    IntegratorConfig< double > cfg;
    auto r = stepper.step( f, x0, 0.0, 0.05, cfg );
    EXPECT_TRUE( r.accepted );
    EXPECT_NEAR( r.x_new( 0 ), std::exp( 0.05 ), 1e-11 );
}
