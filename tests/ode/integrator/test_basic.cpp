// tests/ode/testIntegratorBasic.cpp
//
// Integrator-level smoke tests for the Taylor method (no events).

#include <gtest/gtest.h>

#include <cmath>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>

using tax::ode::IntegratorConfig;

TEST( OdeIntegrator, ExpEndpointAccurate )
{
    constexpr int N = 16;
    using State = tax::la::VecNT< 1, double >;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const auto f = []( const auto& x, const auto& /*t*/ ) { return x; };

    tax::ode::Taylor< N, State, tax::ode::controllers::JorbaZou< double >, decltype( f ) > integ{
        f, cfg };

    State x0;
    x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, /*t0=*/0.0, /*tmax=*/1.0 );

    EXPECT_GE( sol.size(), 2u );
    EXPECT_DOUBLE_EQ( sol.t.back(), 1.0 );
    EXPECT_NEAR( sol.x.back()( 0 ), std::exp( 1.0 ), 1e-11 );
}

TEST( OdeIntegrator, HarmonicQuarterPeriod )
{
    constexpr int N = 12;
    using State = tax::la::VecNT< 2, double >;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const auto f = []( const auto& x, const auto& /*t*/ ) {
        using S = std::decay_t< decltype( x ) >;
        S out;
        out( 0 ) = x( 1 );
        out( 1 ) = -x( 0 );
        return out;
    };

    tax::ode::Taylor< N, tax::la::VecNT< 2, double >, tax::ode::controllers::JorbaZou< double >,
                      decltype( f ) >
        integ{ f, cfg };

    State x0;
    x0( 0 ) = 1.0;
    x0( 1 ) = 0.0;
    const double T_quarter = M_PI / 2.0;
    auto sol = integ.integrate( x0, 0.0, T_quarter );

    EXPECT_NEAR( sol.x.back()( 0 ), 0.0, 1e-10 );
    EXPECT_NEAR( sol.x.back()( 1 ), -1.0, 1e-10 );
}

TEST( OdeIntegrator, CubicDecayDynamicDim )
{
    constexpr int N = 14;
    using State = Eigen::VectorXd;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const auto f = []( const auto& x, const auto& /*t*/ ) {
        using S = std::decay_t< decltype( x ) >;
        S out{ x.size() };
        out( 0 ) = -x( 0 ) * x( 0 ) * x( 0 );
        return out;
    };

    // Dynamic-D variant.
    tax::ode::Taylor< N, Eigen::VectorXd, tax::ode::controllers::JorbaZou< double >, decltype( f ) >
        integ{ f, cfg };

    State x0( 1 );
    x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    EXPECT_NEAR( sol.x.back()( 0 ), 1.0 / std::sqrt( 3.0 ), 1e-10 );
}

TEST( OdeIntegrator, LotkaVolterraCrossMethod )
{
    using State = tax::la::VecNT< 2, double >;

    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    // dx/dt = x (a - b*y), dy/dt = -y (c - d*x)
    // Use a = 1.1, b = 0.4, c = 0.4, d = 0.1.
    const auto f = []( const auto& z, const auto& /*t*/ ) {
        using S = std::decay_t< decltype( z ) >;
        S out;
        out( 0 ) = z( 0 ) * ( 1.1 - 0.4 * z( 1 ) );
        out( 1 ) = -z( 1 ) * ( 0.4 - 0.1 * z( 0 ) );
        return out;
    };

    State x0;
    x0( 0 ) = 10.0;
    x0( 1 ) = 5.0;
    const double t0 = 0.0, tf = 5.0;

    using S2 = tax::la::VecNT< 2, double >;
    tax::ode::Taylor< 16, S2, tax::ode::controllers::JorbaZou< double >, decltype( f ) > tay{ f,
                                                                                              cfg };
    tax::ode::Verner78< S2 > v78{ f, cfg };
    tax::ode::Verner89< S2 > v89{ f, cfg };
    tax::ode::Fehlberg78< S2 > fhl{ f, cfg };
    tax::ode::Feagin12< S2 > f12{ f, cfg };
    tax::ode::Feagin14< S2 > f14{ f, cfg };

    const auto sol_t = tay.integrate( x0, t0, tf );
    const auto sol_78 = v78.integrate( x0, t0, tf );
    const auto sol_89 = v89.integrate( x0, t0, tf );
    const auto sol_fh = fhl.integrate( x0, t0, tf );
    const auto sol_f12 = f12.integrate( x0, t0, tf );
    const auto sol_f14 = f14.integrate( x0, t0, tf );

    // Cross-method agreement: tolerated to ~1e-9 over moderate horizon.
    EXPECT_NEAR( sol_t.x.back()( 0 ), sol_78.x.back()( 0 ), 1e-9 );
    EXPECT_NEAR( sol_t.x.back()( 1 ), sol_78.x.back()( 1 ), 1e-9 );
    EXPECT_NEAR( sol_t.x.back()( 0 ), sol_89.x.back()( 0 ), 1e-9 );
    EXPECT_NEAR( sol_t.x.back()( 1 ), sol_89.x.back()( 1 ), 1e-9 );
    // Looser tolerance per the Fehlberg coincidence (see spec Risks).
    EXPECT_NEAR( sol_t.x.back()( 0 ), sol_fh.x.back()( 0 ), 1e-8 );
    EXPECT_NEAR( sol_t.x.back()( 1 ), sol_fh.x.back()( 1 ), 1e-8 );
    // Feagin12/14 share the order-matched 1e-10 envelope.
    EXPECT_NEAR( sol_t.x.back()( 0 ), sol_f12.x.back()( 0 ), 1e-10 );
    EXPECT_NEAR( sol_t.x.back()( 1 ), sol_f12.x.back()( 1 ), 1e-10 );
    EXPECT_NEAR( sol_t.x.back()( 0 ), sol_f14.x.back()( 0 ), 1e-10 );
    EXPECT_NEAR( sol_t.x.back()( 1 ), sol_f14.x.back()( 1 ), 1e-10 );
}
