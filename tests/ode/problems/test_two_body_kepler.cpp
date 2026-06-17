// tests/ode/testTwoBodyKepler.cpp
//
// Planar Kepler with eccentricity e = 0.5, canonical units (GM = 1,
// semi-major axis a = 1). Initial conditions at periapsis:
//   r_p = a(1 - e) = 0.5
//   v_p = sqrt(GM/a * (1+e)/(1-e)) = sqrt(3)
// Period: T = 2π. Propagate 10 periods and verify:
//   - specific energy E = ½‖v‖² - 1/‖r‖ conserved within method-scaled tol
//   - specific angular momentum L = x*vy - y*vx conserved
//   - closure ‖r(10T) - r_periapsis‖ within tol

#include <gtest/gtest.h>

#include <tax/la/types.hpp>
#include <cmath>

#include <tax/ode.hpp>

namespace
{

constexpr double kEcc        = 0.5;
constexpr double kPeriapsis  = 1.0 - kEcc;   // a(1-e), a=1
const     double kVPeriapsis = std::sqrt( 1.0 / 1.0 * ( 1.0 + kEcc ) / ( 1.0 - kEcc ) );
constexpr double kPeriod     = 2.0 * M_PI;

using State = tax::la::VecNT< 4, double >;

inline State make_ic()
{
    State x0;
    x0( 0 ) = kPeriapsis; x0( 1 ) = 0.0;
    x0( 2 ) = 0.0;        x0( 3 ) = kVPeriapsis;
    return x0;
}

inline auto make_rhs()
{
    return []( const auto& s, const auto& /*t*/ )
    {
        using S = std::decay_t< decltype( s ) >;
        S out;
        const auto x  = s( 0 );
        const auto y  = s( 1 );
        const auto r2 = x * x + y * y;
        const auto r3 = r2 * sqrt( r2 );
        out( 0 ) = s( 2 );
        out( 1 ) = s( 3 );
        out( 2 ) = -x / r3;
        out( 3 ) = -y / r3;
        return out;
    };
}

double specific_energy( const State& s )
{
    const double r  = std::hypot( s( 0 ), s( 1 ) );
    const double v2 = s( 2 ) * s( 2 ) + s( 3 ) * s( 3 );
    return 0.5 * v2 - 1.0 / r;
}

double specific_angmom( const State& s )
{
    return s( 0 ) * s( 3 ) - s( 1 ) * s( 2 );
}

template < class Sol >
void check_invariants( const Sol& sol, double tol_E, double tol_L, double tol_close,
                       const char* label )
{
    const State x0     = sol.x.front();
    const State xfinal = sol.x.back();
    EXPECT_NEAR( specific_energy( xfinal ), specific_energy( x0 ), tol_E )
        << "method=" << label << " (energy)";
    EXPECT_NEAR( specific_angmom( xfinal ), specific_angmom( x0 ), tol_L )
        << "method=" << label << " (angular momentum)";
    EXPECT_NEAR( ( xfinal - x0 ).norm(), 0.0, tol_close )
        << "method=" << label << " (closure)";
}

}  // namespace

TEST( OdeTwoBodyKepler, Taylor16 )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    tax::ode::Taylor< 16, State, tax::ode::controllers::JorbaZou< double >, false, decltype( make_rhs() ) > integ{ make_rhs(), cfg };
    auto sol = integ.integrate( make_ic(), 0.0, 10.0 * kPeriod );

    check_invariants( sol, /*E=*/1e-10, /*L=*/1e-10, /*close=*/1e-8, "Taylor16" );
}

TEST( OdeTwoBodyKepler, Verner89 )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    tax::ode::Verner89< State > integ{ make_rhs(), cfg };
    auto sol = integ.integrate( make_ic(), 0.0, 10.0 * kPeriod );

    check_invariants( sol, /*E=*/1e-9, /*L=*/1e-9, /*close=*/1e-7, "Verner89" );
}

TEST( OdeTwoBodyKepler, Feagin14 )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    tax::ode::Feagin14< State > integ{ make_rhs(), cfg };
    auto sol = integ.integrate( make_ic(), 0.0, 10.0 * kPeriod );

    check_invariants( sol, /*E=*/1e-11, /*L=*/1e-11, /*close=*/1e-9, "Feagin14" );
}

TEST( OdeTwoBodyKepler, Verner78 )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    tax::ode::Verner78< State > integ{ make_rhs(), cfg };
    auto sol = integ.integrate( make_ic(), 0.0, 10.0 * kPeriod );

    check_invariants( sol, /*E=*/1e-8, /*L=*/1e-8, /*close=*/1e-6, "Verner78" );
}
