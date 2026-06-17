// tests/ode/testFixedStep.cpp
//
// FixedStep controller: forces every step to use cfg.initial_step
// and to be accepted, regardless of tolerance. One test per stepper.

#include <gtest/gtest.h>

#include <tax/la/types.hpp>

#include <tax/ode.hpp>

using tax::ode::IntegratorConfig;
using tax::ode::controllers::FixedStep;

namespace
{

constexpr double kH = 0.1;

template < class State >
auto identity_rhs()
{
    return []( const State& x, double ) { return x; };
}

template < class Solution >
void check_uniform_grid( const Solution& sol, double h, std::size_t expected_count )
{
    ASSERT_EQ( sol.t.size(), expected_count );
    for ( std::size_t i = 0; i < sol.t.size(); ++i )
        EXPECT_NEAR( sol.t[ i ], h * double( i ), 1e-12 )
            << "step index " << i;
}

}  // namespace

TEST( OdeFixedStep, Verner78AlwaysAcceptedAtTightTol )
{
    using State = tax::la::VecNT< 1, double >;

    IntegratorConfig< double > cfg;
    cfg.initial_step = kH;
    cfg.abstol = cfg.reltol = 1e-30;        // impossibly tight; must still accept

    tax::ode::Verner78< State, FixedStep< double > > integ{ identity_rhs< State >(), cfg };

    State x0; x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    check_uniform_grid( sol, kH, /*expected_count=*/11u );
}

TEST( OdeFixedStep, Verner89AlwaysAcceptedAtTightTol )
{
    using State = tax::la::VecNT< 1, double >;

    IntegratorConfig< double > cfg;
    cfg.initial_step = kH;
    cfg.abstol = cfg.reltol = 1e-30;

    tax::ode::Verner89< State, FixedStep< double > > integ{ identity_rhs< State >(), cfg };

    State x0; x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    check_uniform_grid( sol, kH, /*expected_count=*/11u );
}

TEST( OdeFixedStep, Fehlberg78AlwaysAcceptedAtTightTol )
{
    using State = tax::la::VecNT< 1, double >;

    IntegratorConfig< double > cfg;
    cfg.initial_step = kH;
    cfg.abstol = cfg.reltol = 1e-30;

    tax::ode::Fehlberg78< State, FixedStep< double > > integ{ identity_rhs< State >(), cfg };

    State x0; x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    check_uniform_grid( sol, kH, /*expected_count=*/11u );
}

TEST( OdeFixedStep, Feagin12AlwaysAcceptedAtTightTol )
{
    using State = tax::la::VecNT< 1, double >;

    IntegratorConfig< double > cfg;
    cfg.initial_step = kH;
    cfg.abstol = cfg.reltol = 1e-30;

    tax::ode::Feagin12< State, FixedStep< double > > integ{ identity_rhs< State >(), cfg };

    State x0; x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    check_uniform_grid( sol, kH, /*expected_count=*/11u );
}

TEST( OdeFixedStep, Feagin14AlwaysAcceptedAtTightTol )
{
    using State = tax::la::VecNT< 1, double >;

    IntegratorConfig< double > cfg;
    cfg.initial_step = kH;
    cfg.abstol = cfg.reltol = 1e-30;

    tax::ode::Feagin14< State, FixedStep< double > > integ{ identity_rhs< State >(), cfg };

    State x0; x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    check_uniform_grid( sol, kH, /*expected_count=*/11u );
}

TEST( OdeFixedStep, TaylorAlwaysAcceptedAtTightTol )
{
    using State = tax::la::VecNT< 1, double >;

    IntegratorConfig< double > cfg;
    cfg.initial_step = kH;
    cfg.abstol = cfg.reltol = 1e-30;        // impossibly tight; must still accept

    // TaylorStepper calls f(x_te, t_te) with TE-valued arguments, so the
    // RHS must be a generic lambda (std::function with a fixed signature
    // would not compose with the TE state). The F parameter is spelled
    // explicitly to preserve the zero-indirection path.
    auto rhs = []( const auto& x, const auto& /*t*/ ) { return x; };
    tax::ode::Taylor< 16, State, FixedStep< double >, false, decltype( rhs ) > integ_fs{ rhs, cfg };

    State x0; x0( 0 ) = 1.0;
    auto sol = integ_fs.integrate( x0, 0.0, 1.0 );

    check_uniform_grid( sol, kH, /*expected_count=*/11u );
}
