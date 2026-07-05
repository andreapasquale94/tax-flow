// tests/ode/steppers/test_taylor_model.cpp
//
// Validated Taylor-model integration (methods::Picard / TaylorModelStepper):
//   - the propagated model ENCLOSES the analytic flow across the whole IC box
//     (the defining property — checked on a sample sweep),
//   - the polynomial part reproduces the analytic solution at the box center,
//   - remainders stay tight at integrator tolerances,
//   - non-autonomous RHS receive the time as a Taylor model.
//
// Requires the tax core's tax::model module (tax PR "feat(model)").

#include <gtest/gtest.h>

#if __has_include( <tax/model.hpp>)

#include <array>
#include <cmath>
#include <tax/domain/box.hpp>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>

using tax::ode::IntegratorConfig;

namespace
{
constexpr int P = 7;
constexpr int M = 2;
constexpr int D = 2;
using TM = tax::model::TaylorModel< double, P, M >;
using State = Eigen::Matrix< TM, D, 1 >;

// Harmonic oscillator: y1' = y2, y2' = -y1.
auto rhs()
{
    return []( const auto& x, const auto& ) {
        using S = std::decay_t< decltype( x ) >;
        S out{ x.size() };
        out( 0 ) = x( 1 );
        out( 1 ) = -1.0 * x( 0 );
        return out;
    };
}

State icState()
{
    tax::domain::Box< double, M > box{ { 1.0, 0.5 }, { 0.1, 0.1 } };
    Eigen::Matrix< double, D, 1 > x0;
    x0 << 1.0, 0.5;
    return tax::domain::createModel< P >( box, x0 );
}
}  // namespace

TEST( OdeTaylorModel, EnclosesAnalyticFlowOverTheBox )
{
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-9;

    const double t1 = 1.0;
    auto sol = tax::ode::propagate( tax::ode::methods::Picard{}, rhs(), icState(), 0.0, t1, cfg );

    EXPECT_DOUBLE_EQ( sol.t.back(), t1 );
    const State& xf = sol.x.back();

    // Sweep the factor cube: the model must enclose the analytic flow
    //   y1(t) = y10 cos t + y20 sin t,  y2(t) = -y10 sin t + y20 cos t
    // at every sampled initial condition.
    for ( double xi1 : { -1.0, -0.6, -0.2, 0.0, 0.3, 0.7, 1.0 } )
        for ( double xi2 : { -1.0, -0.5, 0.0, 0.5, 1.0 } )
        {
            const double y10 = 1.0 + 0.1 * xi1;
            const double y20 = 0.5 + 0.1 * xi2;
            const double truth1 = y10 * std::cos( t1 ) + y20 * std::sin( t1 );
            const double truth2 = -y10 * std::sin( t1 ) + y20 * std::cos( t1 );

            const std::array< double, M > xi{ xi1, xi2 };
            const auto e1 = xf( 0 ).eval( xi );
            const auto e2 = xf( 1 ).eval( xi );
            EXPECT_TRUE( e1.contains( truth1 ) )
                << "y1 enclosure [" << e1.lower() << ", " << e1.upper() << "] misses " << truth1;
            EXPECT_TRUE( e2.contains( truth2 ) )
                << "y2 enclosure [" << e2.lower() << ", " << e2.upper() << "] misses " << truth2;
        }

    // The enclosure is tight: remainders at the integrator tolerance scale.
    EXPECT_LT( xf( 0 ).remainder().width(), 1e-6 );
    EXPECT_LT( xf( 1 ).remainder().width(), 1e-6 );

    // Center of the box: the polynomial constant term is the flow of (1, 0.5).
    EXPECT_NEAR( xf( 0 ).value(), std::cos( t1 ) + 0.5 * std::sin( t1 ), 1e-7 );
    EXPECT_NEAR( xf( 1 ).value(), -std::sin( t1 ) + 0.5 * std::cos( t1 ), 1e-7 );
}

TEST( OdeTaylorModel, LinearPartIsTheStateTransitionMatrix )
{
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-9;

    const double t1 = 0.8;
    auto sol = tax::ode::propagate( tax::ode::methods::Picard{}, rhs(), icState(), 0.0, t1, cfg );
    const State& xf = sol.x.back();

    // d(y_i)/d(ξ_j) = STM · diag(halfWidth): halfWidth = 0.1 on both axes.
    tax::MultiIndex< M > e1{}, e2{};
    e1[0] = 1;
    e2[1] = 1;
    EXPECT_NEAR( xf( 0 ).polynomial().coeff( e1 ), 0.1 * std::cos( t1 ), 1e-7 );
    EXPECT_NEAR( xf( 0 ).polynomial().coeff( e2 ), 0.1 * std::sin( t1 ), 1e-7 );
    EXPECT_NEAR( xf( 1 ).polynomial().coeff( e1 ), -0.1 * std::sin( t1 ), 1e-7 );
    EXPECT_NEAR( xf( 1 ).polynomial().coeff( e2 ), 0.1 * std::cos( t1 ), 1e-7 );
}

TEST( OdeTaylorModel, NonAutonomousRhsGetsTimeAsAModel )
{
    // x' = t, x(0) = x0 ⇒ x(t) = x0 + t²/2 — checks the τ-variable plumbing.
    constexpr int M1 = 1;
    using TM1 = tax::model::TaylorModel< double, 4, M1 >;
    using S1 = Eigen::Matrix< TM1, 1, 1 >;

    tax::domain::Box< double, M1 > box;
    box.center( 0 ) = 2.0;
    box.halfWidth( 0 ) = 0.5;
    Eigen::Matrix< double, 1, 1 > x0;
    x0 << 2.0;
    S1 s = tax::domain::createModel< 4 >( box, x0 );

    const auto f = []( const auto& x, const auto& t ) {
        using S = std::decay_t< decltype( x ) >;
        S out{ x.size() };
        out( 0 ) = t;
        return out;
    };

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-10;
    auto sol = tax::ode::propagate( tax::ode::methods::Picard{}, f, s, 0.0, 2.0, cfg );

    const auto& xf = sol.x.back()( 0 );
    for ( double xi : { -1.0, 0.0, 1.0 } )
    {
        const double truth = ( 2.0 + 0.5 * xi ) + 2.0;  // x0 + t²/2 at t = 2
        EXPECT_TRUE( xf.eval( std::array< double, 1 >{ xi } ).contains( truth ) );
    }
    EXPECT_LT( xf.remainder().width(), 1e-8 );
}

TEST( OdeTaylorModel, RemainderGrowsMonotonicallyAcrossSteps )
{
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-9;

    auto sol = tax::ode::propagate( tax::ode::methods::Picard{}, rhs(), icState(), 0.0, 1.5, cfg );
    ASSERT_GE( sol.x.size(), 3u );  // several accepted steps
    double prev = 0.0;
    for ( const auto& x : sol.x )
    {
        const double w = x( 0 ).remainder().width();
        EXPECT_GE( w, prev - 1e-18 );  // the validated enclosure only widens
        prev = w;
    }
}

#else  // tax core without tax::model

#include <gtest/gtest.h>
TEST( OdeTaylorModel, SkippedTaxCoreLacksModelModule ) { GTEST_SKIP(); }

#endif
