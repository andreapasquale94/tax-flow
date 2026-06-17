// tests/ode/testRKWithDaState.cpp
//
// Propagate vector ODEs with State = Eigen::Matrix<TEn<P,M>, D, 1>
// across the five RK families and verify:
//
//   (a) constant DA term ≈ double-state propagation
//       Reference problem: planar Kepler (e=0.5, GM=a=1), 1 period.
//       The constant-DA recurrence is algebraically identical to the
//       scalar one; differences are pure FP-rounding, so we tighten to
//       a single near-machine bound across every method.
//
//   (b) linear DA term ≈ analytical STM
//       Reference problem: harmonic oscillator (ω=1), state (x,v),
//       integrated for t = π/2. Analytical solution
//           x(t) = x0 cos t + v0 sin t
//           v(t) = -x0 sin t + v0 cos t
//       gives an exact 2×2 STM whose (i,j) entry is the partial
//       derivative ∂(component i at t)/∂(IC component j). The DA
//       linear coefficient at multi-index e_j is exactly
//       halfWidth · STM[i,j]; the residual is integration round-off
//       (no truncation, no finite-difference cancellation).

#include <gtest/gtest.h>

#include <tax/la/types.hpp>
#include <cmath>

#include <tax/ode.hpp>
#include <tax/tax.hpp>

namespace {

constexpr int    P          = 2;
constexpr double kHalfWidth = 1e-3;

// ------ Kepler setup (constant-term parity) ----------------------------

constexpr int    Mk          = 4;        // 4 IC DA variables
constexpr double kEcc        = 0.5;
constexpr double kPeriapsis  = 1.0 - kEcc;
const     double kVPeriapsis = std::sqrt( ( 1.0 + kEcc ) / ( 1.0 - kEcc ) );
constexpr double kPeriod     = 2.0 * M_PI;

using DAk      = tax::TEn< P, Mk >;
using StateKD  = tax::la::VecNT< 4, double >;
using StateKDA = tax::la::VecNT< 4, DAk >;

StateKD make_kepler_ic_double()
{
    StateKD x0;
    x0( 0 ) = kPeriapsis;  x0( 1 ) = 0.0;
    x0( 2 ) = 0.0;         x0( 3 ) = kVPeriapsis;
    return x0;
}

StateKDA make_kepler_ic_da()
{
    StateKD  c = make_kepler_ic_double();
    StateKDA x0;
    for ( int i = 0; i < 4; ++i )
        x0( i ) = DAk( c( i ) ) + DAk( kHalfWidth ) * DAk::variable( 0.0, i );
    return x0;
}

template < class S >
auto make_kepler_rhs()
{
    return []( const S& s, double /*t*/ )
    {
        using std::sqrt;
        using tax::sqrt;
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

template < template < class, class, bool, class > class IntegratorAlias >
void check_constant_term_matches_double( const char* method_name,
                                         double      tol_close )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    using IntegD = IntegratorAlias<
        StateKD, tax::ode::controllers::PI< double >, false,
        typename std::remove_reference_t<
            decltype( make_kepler_rhs< StateKD >() ) > >;
    using IntegDA = IntegratorAlias<
        StateKDA, tax::ode::controllers::PI< double >, false,
        typename std::remove_reference_t<
            decltype( make_kepler_rhs< StateKDA >() ) > >;

    IntegD  integ_d { make_kepler_rhs< StateKD  >(), cfg };
    IntegDA integ_da{ make_kepler_rhs< StateKDA >(), cfg };

    auto sol_d  = integ_d .integrate( make_kepler_ic_double(), 0.0, kPeriod );
    auto sol_da = integ_da.integrate( make_kepler_ic_da(),     0.0, kPeriod );

    const StateKD&  xT_d  = sol_d .x.back();
    const StateKDA& xT_da = sol_da.x.back();
    for ( int i = 0; i < 4; ++i )
    {
        EXPECT_NEAR( xT_da( i )[ 0 ], xT_d( i ), tol_close )
            << "method=" << method_name << " component=" << i;
    }
}

// ------ Harmonic-oscillator setup (analytical-STM check) ---------------

constexpr int    Mh             = 2;            // 2 IC DA variables
constexpr double kHarmIC_x      = 1.0;
constexpr double kHarmIC_v      = 0.0;
const     double kHarmT         = M_PI_2;       // quarter period

using DAh      = tax::TEn< P, Mh >;
using StateHD  = tax::la::VecNT< 2, double >;
using StateHDA = tax::la::VecNT< 2, DAh >;

StateHDA make_harmonic_ic_da()
{
    StateHDA x0;
    x0( 0 ) = DAh( kHarmIC_x ) + DAh( kHalfWidth ) * DAh::variable( 0.0, 0 );
    x0( 1 ) = DAh( kHarmIC_v ) + DAh( kHalfWidth ) * DAh::variable( 0.0, 1 );
    return x0;
}

template < class S >
auto make_harmonic_rhs()
{
    return []( const S& s, double /*t*/ )
    {
        S out;
        out( 0 ) =  s( 1 );
        out( 1 ) = -s( 0 );
        return out;
    };
}

template < template < class, class, bool, class > class IntegratorAlias >
void check_linear_term_matches_analytical_stm( const char* method_name,
                                               double      tol_abs )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    using IntegDA = IntegratorAlias<
        StateHDA, tax::ode::controllers::PI< double >, false,
        typename std::remove_reference_t<
            decltype( make_harmonic_rhs< StateHDA >() ) > >;
    IntegDA integ_da{ make_harmonic_rhs< StateHDA >(), cfg };
    auto    sol_da = integ_da.integrate( make_harmonic_ic_da(), 0.0, kHarmT );

    const StateHDA& xT_da = sol_da.x.back();

    // Analytical STM at time t:
    //   [  cos t   sin t ]
    //   [ -sin t   cos t ]
    const double c = std::cos( kHarmT );
    const double s = std::sin( kHarmT );
    const double stm[ 2 ][ 2 ] = { {  c, s },
                                   { -s, c } };

    // DA layout: TEn<P,2>::flatIndex({e_j}) = 1 + j (degree-1 monomial j).
    // halfWidth chain rule: DA coef at e_j = halfWidth · ∂x_i/∂x0_j.
    for ( int row = 0; row < 2; ++row )
    {
        for ( int col = 0; col < 2; ++col )
        {
            const std::size_t flat = static_cast< std::size_t >( 1 + col );
            const double      lhs  = xT_da( row )[ flat ] / kHalfWidth;
            const double      rhs  = stm[ row ][ col ];
            EXPECT_NEAR( lhs, rhs, tol_abs )
                << "method=" << method_name
                << " row="   << row
                << " col="   << col;
        }
    }
}

}  // namespace

// -------- Constant-term parity (Kepler) --------------------------------

TEST( OdeRKWithDaState, ConstantTermVerner78 )
{
    // Verner78's PI-controlled step pattern can land DA vs scalar runs on
    // slightly different time grids; the resulting FP drift sits around
    // a few e-12 even though both end-states are essentially zero.
    check_constant_term_matches_double< tax::ode::Verner78 >( "Verner78", 1e-11 );
}

TEST( OdeRKWithDaState, ConstantTermVerner89 )
{
    check_constant_term_matches_double< tax::ode::Verner89 >( "Verner89", 1e-12 );
}

TEST( OdeRKWithDaState, ConstantTermFehlberg78 )
{
    check_constant_term_matches_double< tax::ode::Fehlberg78 >( "Fehlberg78", 1e-12 );
}

TEST( OdeRKWithDaState, ConstantTermFeagin12 )
{
    check_constant_term_matches_double< tax::ode::Feagin12 >( "Feagin12", 1e-12 );
}

TEST( OdeRKWithDaState, ConstantTermFeagin14 )
{
    check_constant_term_matches_double< tax::ode::Feagin14 >( "Feagin14", 1e-12 );
}

// -------- Linear term vs analytical STM (harmonic oscillator) ----------

TEST( OdeRKWithDaState, LinearTermVerner78 )
{
    check_linear_term_matches_analytical_stm< tax::ode::Verner78 >( "Verner78", 1e-11 );
}

TEST( OdeRKWithDaState, LinearTermVerner89 )
{
    check_linear_term_matches_analytical_stm< tax::ode::Verner89 >( "Verner89", 1e-11 );
}

TEST( OdeRKWithDaState, LinearTermFehlberg78 )
{
    check_linear_term_matches_analytical_stm< tax::ode::Fehlberg78 >( "Fehlberg78", 1e-11 );
}

TEST( OdeRKWithDaState, LinearTermFeagin12 )
{
    check_linear_term_matches_analytical_stm< tax::ode::Feagin12 >( "Feagin12", 1e-11 );
}

TEST( OdeRKWithDaState, LinearTermFeagin14 )
{
    check_linear_term_matches_analytical_stm< tax::ode::Feagin14 >( "Feagin14", 1e-11 );
}
