// tests/ode/testTaylorStepper.cpp
//
// Stepper-level correctness. We exercise dx/dt = x (analytic
// solution exp(t)) and the harmonic oscillator dx/dt = (v, -x) with
// analytic solution (cos t, -sin t) starting from (1, 0).

#include <gtest/gtest.h>

#include <cmath>
#include <tax/la.hpp>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>

using tax::ode::IntegratorConfig;
using tax::ode::TaylorStepper;
using tax::ode::controllers::JorbaZou;
using tax::ode::controllers::PI;

// Regression (O6): a Taylor integrator with a generic I/PI controller on a
// polynomial RHS (exact zero top-order coefficients) must not stall. Before the
// fix the zero truncation indicator made the controller SHRINK every step until
// it died with "step size below min_step". x' = 2t, x(0)=0 ⇒ x(t)=t² exactly.
TEST( OdeTaylorStepper, ZeroErrorPolynomialRhsCompletesWithPI )
{
    using State = tax::la::VecNT< 1, double >;
    const auto f = []( const auto& x, const auto& t ) {
        using S = std::decay_t< decltype( x ) >;
        S out{ x.size() };
        out( 0 ) = 2.0 * t;  // note: independent of x, degree-1 in t
        return out;
    };

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    tax::ode::Taylor< 8, State, PI< double >, decltype( f ) > integ{ f, cfg };
    State x0;
    x0( 0 ) = 0.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );
    EXPECT_DOUBLE_EQ( sol.t.back(), 1.0 );
    EXPECT_NEAR( sol.x.back()( 0 ), 1.0, 1e-10 );  // x(1) = 1
}

TEST( OdeTaylorStepper, ExponentialOneStep )
{
    using State = tax::la::VecNT< 1, double >;
    TaylorStepper< 12, State > stepper;

    State x0;
    x0( 0 ) = 1.0;
    const auto f = []( const auto& x, const auto& /*t*/ ) { return x; };

    IntegratorConfig< double > cfg;
    cfg.abstol = 1e-12;

    auto r = stepper.step( f, x0, /*t=*/0.0, /*h=*/0.1, cfg );

    EXPECT_TRUE( r.accepted );
    // x(0.1) = e^0.1 ≈ 1.10517091808...
    EXPECT_NEAR( r.x_new( 0 ), std::exp( 0.1 ), 1e-12 );
    // tax::la::eval at τ=0 reproduces x0.
    auto x_at_t0 = tax::la::eval( r.data, 0.0 );
    EXPECT_NEAR( x_at_t0( 0 ), x0( 0 ), 1e-14 );
    // tax::la::eval at τ=h_used reproduces x_new.
    auto x_at_t1 = tax::la::eval( r.data, r.h_used );
    EXPECT_NEAR( x_at_t1( 0 ), r.x_new( 0 ), 1e-14 );
}

TEST( OdeTaylorStepper, HarmonicOneStep )
{
    using State = tax::la::VecNT< 2, double >;
    TaylorStepper< 12, State > stepper;

    State x0;
    x0( 0 ) = 1.0;  // q
    x0( 1 ) = 0.0;  // p
    const auto f = []( const auto& x, const auto& /*t*/ ) {
        using S = std::decay_t< decltype( x ) >;
        S out;
        out( 0 ) = x( 1 );
        out( 1 ) = -x( 0 );
        return out;
    };

    IntegratorConfig< double > cfg;
    cfg.abstol = 1e-12;
    auto r = stepper.step( f, x0, 0.0, 0.05, cfg );

    EXPECT_TRUE( r.accepted );
    EXPECT_NEAR( r.x_new( 0 ), std::cos( 0.05 ), 1e-12 );
    EXPECT_NEAR( r.x_new( 1 ), -std::sin( 0.05 ), 1e-12 );
}

TEST( OdeTaylorStepper, ControllerPIAlsoWorks )
{
    using State = tax::la::VecNT< 1, double >;
    TaylorStepper< 10, State, PI< double > > stepper;

    State x0;
    x0( 0 ) = 1.0;
    const auto f = []( const auto& x, const auto& /*t*/ ) { return x; };

    IntegratorConfig< double > cfg;
    auto r = stepper.step( f, x0, 0.0, 0.05, cfg );
    EXPECT_TRUE( r.accepted );
    EXPECT_NEAR( r.x_new( 0 ), std::exp( 0.05 ), 1e-10 );
}
