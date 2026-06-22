// tests/ode/test_mixed_state.cpp
//
// Mixed-order expansions (tax::MTE) as ODE state scalars. Exercises the
// NamedExpansion VectorOps spec + Eigen::NumTraits<MixedTaylorExpansion>
// by integrating Eigen vectors of mixed expansions and checking the
// resulting coefficients against closed-form sensitivities. The second
// test uses two axes with DIFFERENT per-axis orders (genuinely mixed).

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <tax/tax.hpp>

namespace
{

constexpr double kHw = 1e-3;

// --- Harmonic oscillator with a mixed-order IC axis "da" (dim 2, order 2) ---
using DaAxis = tax::named::OrderedAxis< "da", 2, 2 >;
using MEda = tax::named::MixedTaylorExpansion< double, DaAxis >;
using StateH = tax::la::VecNT< 2, MEda >;

StateH make_harmonic_ic()
{
    auto v = tax::mixed::variables< "da", 2, 2 >( std::array< double, 2 >{ 0.0, 0.0 } );
    StateH x0;
    x0( 0 ) = MEda( 1.0 ) + kHw * v[0];  // x(0) = 1
    x0( 1 ) = MEda( 0.0 ) + kHw * v[1];  // v(0) = 0
    return x0;
}

}  // namespace

TEST( MixedOdeState, HarmonicLinearTermMatchesAnalyticalStm )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    auto rhs = []( const StateH& s, double /*t*/ ) {
        StateH out;
        out( 0 ) = s( 1 );
        out( 1 ) = -s( 0 );
        return out;
    };

    tax::ode::Verner89< StateH > integ{ rhs, cfg };
    const double t = M_PI_2;
    auto sol = integ.integrate( make_harmonic_ic(), 0.0, t );

    const StateH& xT = sol.x.back();

    // Analytical STM at t: [[cos t, sin t], [-sin t, cos t]].
    const double c = std::cos( t );
    const double s = std::sin( t );
    const double stm[2][2] = { { c, s }, { -s, c } };

    // Box layout for a single 2-D axis at order 2 is the simplex on those 2
    // vars, so the degree-1 monomial for da_j sits at flat index 1 + j.
    for ( int row = 0; row < 2; ++row )
        for ( int col = 0; col < 2; ++col )
        {
            const double lhs = xT( row ).inner()[std::size_t( 1 + col )] / kHw;
            EXPECT_NEAR( lhs, stm[row][col], 1e-11 ) << "row=" << row << " col=" << col;
        }
}

TEST( MixedOdeState, TwoAxesDifferentOrdersParameterSensitivity )
{
    // x' = a x, with state carrying TWO axes at different orders:
    //   "a"  (parameter)         dim 1, order 4
    //   "x0" (initial condition) dim 1, order 2
    // This is a genuinely mixed (anisotropic-order) expansion.
    using AAxis = tax::named::OrderedAxis< "a", 1, 4 >;
    using X0Axis = tax::named::OrderedAxis< "x0", 1, 2 >;
    using ME = tax::named::MixedTaylorExpansion< double, AAxis, X0Axis >;
    using State = tax::la::VecNT< 1, ME >;

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    const double a0 = 0.7, x0v = 2.0, tEnd = 0.5;

    // Build a-parameter and IC, each promoted into the union axis set {a, x0}.
    auto a1 = tax::mixed::variable< "a", 4 >( a0 );    // MTE over {a@4}
    auto i1 = tax::mixed::variable< "x0", 2 >( x0v );  // MTE over {x0@2}
    ME a_param = a1.template embed< ME >();            // embed {a@4} -> {a@4, x0@2}
    ME x_init = i1.template embed< ME >();             // embed {x0@2} -> {a@4, x0@2}

    auto rhs = [a_param]( const State& s, double /*t*/ ) {
        State out;
        out( 0 ) = a_param * s( 0 );
        return out;
    };

    State x0;
    x0( 0 ) = x_init;

    tax::ode::Verner89< State > integ{ rhs, cfg };
    auto sol = integ.integrate( x0, 0.0, tEnd );
    const ME& xT = sol.x.back()( 0 );

    // x(t) = x0 * exp(a t); value at expansion point:
    const double base = x0v * std::exp( a0 * tEnd );
    EXPECT_NEAR( xT.value(), base, 1e-9 );

    // d x / d a at point = x0 * t * exp(a0 t); read via deriv<"a">().value().
    // deriv<"a">() computes the formal partial derivative w.r.t. the "a" variable,
    // whose constant term equals the linear coefficient in xT.
    const double dxda = x0v * tEnd * std::exp( a0 * tEnd );
    EXPECT_NEAR( xT.deriv< "a" >().value(), dxda, 1e-9 );

    // d x / d x0 at point = exp(a0 t); first-order "x0" coeff.
    EXPECT_NEAR( xT.deriv< "x0" >().value(), std::exp( a0 * tEnd ), 1e-9 );
}
