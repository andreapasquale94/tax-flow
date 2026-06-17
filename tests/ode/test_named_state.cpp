// tests/ode/test_named_state.cpp
//
// Named expansions (tax::named) as ODE state scalars. Exercises the
// VectorOps<NamedTaylorExpansion> specialization + Eigen::NumTraits<NamedTaylorExpansion> by
// integrating Eigen vectors of named expansions and checking the resulting
// Taylor coefficients against closed-form sensitivities.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <tax/tax.hpp>

using namespace tax::named;

namespace
{

constexpr int kP = 2;
constexpr double kHw = 1e-3;

// --- Harmonic oscillator with a named initial-condition axis ---------------

using DaAxis = Axis< "da", 2 >;  // the 2 IC perturbation variables
using NEda = NamedTaylorExpansion< double, kP, DaAxis >;
using StateH = tax::la::VecNT< 2, NEda >;

StateH make_harmonic_ic()
{
    auto v = variables< "da", kP >( std::array< double, 2 >{ 0.0, 0.0 } );
    StateH x0;
    x0( 0 ) = NEda( 1.0 ) + kHw * v[0];  // x(0) = 1
    x0( 1 ) = NEda( 0.0 ) + kHw * v[1];  // v(0) = 0
    return x0;
}

// --- Linear growth x' = a x with a named parameter axis --------------------

using AAxis = Axis< "a", 1 >;
using NEa = NamedTaylorExpansion< double, kP, AAxis >;
using StateP = tax::la::VecNT< 1, NEa >;

}  // namespace

TEST( NamedOdeState, HarmonicLinearTermMatchesAnalyticalStm )
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

    // NEda layout: degree-1 monomial for da_j sits at flat index 1 + j; the
    // half-width chain rule gives coeff(e_j) = kHw * d x_i / d x0_j.
    for ( int row = 0; row < 2; ++row )
    {
        for ( int col = 0; col < 2; ++col )
        {
            const double lhs = xT( row ).inner()[std::size_t( 1 + col )] / kHw;
            EXPECT_NEAR( lhs, stm[row][col], 1e-11 ) << "row=" << row << " col=" << col;
        }
    }
}

TEST( NamedOdeState, ParameterSensitivityMatchesClosedForm )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    const double a0 = 0.7;
    const double x0v = 2.0;
    const double tEnd = 0.5;

    auto av = variables< "a", kP >( std::array< double, 1 >{ 0.0 } );
    NEa a_param = NEa( a0 ) + kHw * av[0];  // a = a0 + kHw * da

    auto rhs = [a_param]( const StateP& s, double /*t*/ ) {
        StateP out;
        out( 0 ) = a_param * s( 0 );  // x' = a x
        return out;
    };

    StateP x0;
    x0( 0 ) = NEa( x0v );  // constant initial condition (no a-dependence yet)

    tax::ode::Verner89< StateP > integ{ rhs, cfg };
    auto sol = integ.integrate( x0, 0.0, tEnd );

    const NEa& xT = sol.x.back()( 0 );

    // x(t) = x0 * exp((a0 + kHw da) t)
    //      = x0 exp(a0 t) [ 1 + (kHw t) da + ... ]
    const double base = x0v * std::exp( a0 * tEnd );
    EXPECT_NEAR( xT.value(), base, 1e-10 );
    // coeff at da^1 (flat index 1) = base * kHw * t
    EXPECT_NEAR( xT.inner()[1], base * kHw * tEnd, 1e-12 );
}
