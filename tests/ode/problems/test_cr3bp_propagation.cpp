// tests/ode/testCR3BPPropagation.cpp
//
// CR3BP propagation correctness: each method preserves the Jacobi
// constant within a method-scaled tolerance over T_final = 7 units,
// and the endpoint matches a high-precision reference (Feagin14 at
// 1e-14) within similar tolerances.

#include <gtest/gtest.h>

#include <tax/la/types.hpp>
#include <cmath>

#include "cr3bp_problem.hpp"
#include <tax/ode.hpp>

using namespace tax::ode::test;

namespace
{

CR3BPState compute_reference()
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-14;
    tax::ode::Feagin14< CR3BPState > integ{ cr3bp_rhs(), cfg };
    auto sol = integ.integrate( cr3bp_transit_ic(), 0.0, kCR3BPTFinal );
    return sol.x.back();
}

template < class Sol >
void check_jacobi_preserved( const Sol& sol, double tol )
{
    const double C0 = cr3bp_jacobi( sol.x.front() );
    const double C1 = cr3bp_jacobi( sol.x.back() );
    EXPECT_NEAR( C1, C0, tol );
}

}  // namespace

TEST( OdeCR3BPPropagation, ReferenceTrajectoryIsStable )
{
    const CR3BPState ref = compute_reference();
    EXPECT_TRUE( std::isfinite( ref( 0 ) ) );
    EXPECT_TRUE( std::isfinite( ref( 1 ) ) );
}

TEST( OdeCR3BPPropagation, Taylor16 )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;
    tax::ode::Taylor< 16, CR3BPState, tax::ode::controllers::JorbaZou< double >, false, decltype( cr3bp_rhs() ) > integ{ cr3bp_rhs(), cfg };
    auto sol = integ.integrate( cr3bp_transit_ic(), 0.0, kCR3BPTFinal );
    check_jacobi_preserved( sol, 1e-10 );

    const CR3BPState ref = compute_reference();
    EXPECT_NEAR( ( sol.x.back() - ref ).norm(), 0.0, 1e-8 );
}

TEST( OdeCR3BPPropagation, Verner89 )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;
    tax::ode::Verner89< CR3BPState > integ{ cr3bp_rhs(), cfg };
    auto sol = integ.integrate( cr3bp_transit_ic(), 0.0, kCR3BPTFinal );
    check_jacobi_preserved( sol, 1e-9 );

    const CR3BPState ref = compute_reference();
    EXPECT_NEAR( ( sol.x.back() - ref ).norm(), 0.0, 1e-7 );
}

TEST( OdeCR3BPPropagation, Verner78 )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;
    tax::ode::Verner78< CR3BPState > integ{ cr3bp_rhs(), cfg };
    auto sol = integ.integrate( cr3bp_transit_ic(), 0.0, kCR3BPTFinal );
    check_jacobi_preserved( sol, 1e-8 );

    const CR3BPState ref = compute_reference();
    EXPECT_NEAR( ( sol.x.back() - ref ).norm(), 0.0, 1e-6 );
}

TEST( OdeCR3BPPropagation, Feagin12 )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;
    tax::ode::Feagin12< CR3BPState > integ{ cr3bp_rhs(), cfg };
    auto sol = integ.integrate( cr3bp_transit_ic(), 0.0, kCR3BPTFinal );
    check_jacobi_preserved( sol, 1e-10 );

    const CR3BPState ref = compute_reference();
    EXPECT_NEAR( ( sol.x.back() - ref ).norm(), 0.0, 1e-8 );
}
