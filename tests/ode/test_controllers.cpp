// tests/ode/testControllers.cpp
//
// Covers controller behaviour:
//   - I: stateless, predictable scaling on err < tol and err > tol.
//   - PI: state-evolving, uses previous error.
//   - H211b: state-evolving, smoothed.
//   - JorbaZou: uses last-two-coefficient norms.
// Each test asserts that min_factor / max_factor clamps are applied.

#include <gtest/gtest.h>

#include <cmath>
#include <tax/ode.hpp>

using tax::ode::controllers::H211b;
using tax::ode::controllers::I;
using tax::ode::controllers::JorbaZou;
using tax::ode::controllers::PI;

TEST( OdeControllerI, ScalesDownOnLargeError )
{
    I< double > c;
    const double h_used = 0.1;
    const double err = 10.0;
    const double tol = 1.0;
    const int p_emb = 7;
    const double h_new = c.next_step( h_used, err, tol, p_emb );
    EXPECT_LT( h_new, h_used );
    EXPECT_GE( h_new, h_used * c.min_factor );
}

TEST( OdeControllerI, ScalesUpOnSmallError )
{
    I< double > c;
    const double h_new = c.next_step( /*h_used=*/0.1, /*err=*/0.01,
                                      /*tol=*/1.0, /*p_emb=*/7 );
    EXPECT_GT( h_new, 0.1 );
    EXPECT_LE( h_new, 0.1 * c.max_factor );
}

TEST( OdeControllerPI, RemembersPreviousError )
{
    PI< double > c;
    const double h1 = c.next_step( 0.1, 0.5, 1.0, 7 );
    const double h2 = c.next_step( h1, 0.5, 1.0, 7 );
    // On the second step the proportional term contributes;
    // the result must differ from the I-only equivalent.
    I< double > i;
    const double i_only = i.next_step( h1, 0.5, 1.0, 7 );
    EXPECT_NE( h2, i_only );
}

TEST( OdeControllerH211b, FirstCallBehavesLikeI )
{
    H211b< double > c;
    I< double > i;
    // On its very first call h_prev_ == 0 so the controller falls
    // back to I-step semantics.
    const double h_new_h = c.next_step( 0.1, 0.5, 1.0, 7 );
    const double h_new_i = i.next_step( 0.1, 0.5, 1.0, 7 );
    EXPECT_NEAR( h_new_h, h_new_i, 1e-12 );
}

// The PI proportional/integral exponents must follow Gustafsson: a net current-
// error exponent of -alpha/k (k = p_emb + 1), NOT -(alpha+beta)/k.
TEST( OdeControllerPI, MatchesGustafssonFormula )
{
    PI< double > c;
    const double tol = 1.0;
    const int p = 7;
    const double inv = 1.0 / double( p + 1 );

    const double e0 = 0.5;  // first call -> I-controller, stores err_prev_ = e0
    const double h1 = c.next_step( 0.1, e0, tol, p );

    const double e1 = 2.0;  // second call exercises the PI formula
    const double h2 = c.next_step( h1, e1, tol, p );

    const double raw = std::pow( tol / e1, c.alpha * inv ) * std::pow( tol / e0, -c.beta * inv );
    const double expected = h1 * std::clamp( c.safety * raw, c.min_factor, c.max_factor );
    EXPECT_NEAR( h2, expected, 1e-12 );
}

// The H211b error-term exponent must be order-dependent: 1/(b*k), k = p_emb + 1.
// The previous code used 1/b (independent of order).
TEST( OdeControllerH211b, MatchesFilterFormulaOrderDependent )
{
    H211b< double > c;
    const double tol = 1.0;
    const int p = 7;
    const double inv = 1.0 / double( p + 1 );
    const double b = c.b;

    const double h0 = 0.1, e0 = 2.0;
    const double h1 = c.next_step( h0, e0, tol, p );  // first call -> I

    const double e1 = 3.0;
    const double h2 = c.next_step( h1, e1, tol, p );

    const double expo = inv / b;  // 1 / (b * k)
    const double raw =
        std::pow( tol / e1, expo ) * std::pow( tol / e0, expo ) * std::pow( h1 / h0, -1.0 / b );
    const double expected = h1 * std::clamp( c.safety * raw, c.min_factor, c.max_factor );
    EXPECT_NEAR( h2, expected, 1e-12 );

    // Order-dependence: a higher-order method must yield a different second-call
    // factor for identical errors (the buggy 1/b exponent gave the same value).
    H211b< double > c15;
    const double h1b = c15.next_step( h0, e0, tol, 15 );
    const double h2b = c15.next_step( h1b, e1, tol, 15 );
    EXPECT_NE( h2b / h1b, h2 / h1 );
}

TEST( OdeControllerJorbaZou, ScalesDownOnLargeLeadingCoeff )
{
    JorbaZou< double > c;
    const double h_used = 0.1;
    const double c_N_norm = 1e6;  // very large => shrink
    const double c_Nm1_norm = 1e6;
    const double tol = 1e-12;
    const int N_order = 12;
    const double h_new = c.next_step( h_used, c_N_norm, c_Nm1_norm, tol, N_order );
    EXPECT_LT( h_new, h_used );
    EXPECT_GE( h_new, h_used * c.min_factor );
}
