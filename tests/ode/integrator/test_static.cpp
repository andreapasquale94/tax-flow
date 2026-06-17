// tests/ode/testIntegratorStatic.cpp
//
// Compile-time D == Eigen::Dynamic parity. The same RHS (a generic
// lambda) propagated through static D=2 and dynamic D must produce
// bitwise- or near-bitwise-identical endpoints.

#include <gtest/gtest.h>

#include <tax/la/types.hpp>
#include <cmath>

#include <tax/ode.hpp>

namespace
{

inline auto harmonic_rhs()
{
    return []( const auto& s, const auto& /*t*/ )
    {
        using S = std::decay_t< decltype( s ) >;
        S out;
        if constexpr ( S::RowsAtCompileTime == Eigen::Dynamic )
            out.resize( s.size() );
        out( 0 ) =  s( 1 );
        out( 1 ) = -s( 0 );
        return out;
    };
}

}  // namespace

TEST( OdeIntegratorStatic, Taylor16ParityWithDynamic )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    using SStatic = tax::la::VecNT< 2, double >;
    using SDyn    = Eigen::VectorXd;

    using RhsT = decltype( harmonic_rhs() );
    using Ctrl = tax::ode::controllers::JorbaZou< double >;
    tax::ode::Taylor< 16, SStatic, Ctrl, false, RhsT > integ_s{ harmonic_rhs(), cfg };
    tax::ode::Taylor< 16, SDyn,    Ctrl, false, RhsT > integ_d{ harmonic_rhs(), cfg };

    SStatic x0_s; x0_s( 0 ) = 1.0; x0_s( 1 ) = 0.0;
    SDyn    x0_d( 2 ); x0_d( 0 ) = 1.0; x0_d( 1 ) = 0.0;

    auto sol_s = integ_s.integrate( x0_s, 0.0, M_PI );
    auto sol_d = integ_d.integrate( x0_d, 0.0, M_PI );

    EXPECT_NEAR( sol_s.x.back()( 0 ), sol_d.x.back()( 0 ), 1e-12 );
    EXPECT_NEAR( sol_s.x.back()( 1 ), sol_d.x.back()( 1 ), 1e-12 );
}

TEST( OdeIntegratorStatic, Verner89ParityWithDynamic )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    using SStatic = tax::la::VecNT< 2, double >;
    using SDyn    = Eigen::VectorXd;

    tax::ode::Verner89< SStatic > integ_s{ harmonic_rhs(), cfg };
    tax::ode::Verner89< SDyn >    integ_d{ harmonic_rhs(), cfg };

    SStatic x0_s; x0_s( 0 ) = 1.0; x0_s( 1 ) = 0.0;
    SDyn    x0_d( 2 ); x0_d( 0 ) = 1.0; x0_d( 1 ) = 0.0;

    auto sol_s = integ_s.integrate( x0_s, 0.0, M_PI );
    auto sol_d = integ_d.integrate( x0_d, 0.0, M_PI );

    EXPECT_NEAR( sol_s.x.back()( 0 ), sol_d.x.back()( 0 ), 1e-12 );
    EXPECT_NEAR( sol_s.x.back()( 1 ), sol_d.x.back()( 1 ), 1e-12 );
}
