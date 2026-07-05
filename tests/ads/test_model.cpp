// tests/ads/test_model.cpp
//
// ADS over VALIDATED Taylor-model states (methods::Picard):
//   - end-to-end containment: every sampled true flow lies inside the located
//     leaf's interval enclosure (the property the whole pipeline exists for),
//   - splits fire on the polynomial truncation mass and the children carry
//     the parent's remainder,
//   - split(state, dim) tiles the parent in factor coordinates,
//   - serial and parallel drivers agree.
//
// Requires the tax core's tax::model module (tax PR "feat(model)").

#include <gtest/gtest.h>

#if __has_include( <tax/model.hpp>)

#include <array>
#include <cmath>
#include <tax/ads.hpp>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>

using tax::ode::IntegratorConfig;

namespace
{
constexpr int P = 5;
constexpr int M = 2;
constexpr int D = 2;
using TM = tax::model::TaylorModel< double, P, M >;
using State = Eigen::Matrix< TM, D, 1 >;

// Mildly nonlinear oscillator (same model family as the classic ADS tests):
// y1' = y2, y2' = -y1 - 0.1 y1³ — evaluated in Taylor-model arithmetic.
auto rhs()
{
    return []( const auto& x, const auto& ) {
        using S = std::decay_t< decltype( x ) >;
        S out{ x.size() };
        out( 0 ) = x( 1 );
        out( 1 ) = -1.0 * x( 0 ) - 0.1 * ( x( 0 ) * x( 0 ) * x( 0 ) );
        return out;
    };
}

// Scalar reference propagation (non-validated, tight tolerance).
Eigen::Vector2d scalarReference( const Eigen::Vector2d& x0, double t1 )
{
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    const auto f = []( const auto& x, double ) {
        using S = std::decay_t< decltype( x ) >;
        S out{ x.size() };
        out( 0 ) = x( 1 );
        out( 1 ) = -x( 0 ) - 0.1 * x( 0 ) * x( 0 ) * x( 0 );
        return out;
    };
    tax::ode::Verner89< Eigen::Vector2d > integ{ f, cfg };
    return integ.integrate( x0, 0.0, t1 ).x.back();
}

tax::domain::Box< double, M > icBox()
{
    return { tax::la::VecNT< M, double >{ 1.0, 0.0 }, tax::la::VecNT< M, double >{ 0.4, 0.4 } };
}

Eigen::Matrix< double, D, 1 > icCenter()
{
    Eigen::Matrix< double, D, 1 > c;
    c << 1.0, 0.0;
    return c;
}
}  // namespace

TEST( AdsTaylorModel, LeafEnclosuresContainTheTrueFlow )
{
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-8;
    const double t1 = 2.0;

    auto sol = tax::ads::propagate< P >( tax::ode::methods::Picard{},
                                         tax::ads::TruncationCriterion{ 1e-3, 6 }, rhs(), icBox(),
                                         icCenter(), 0.0, t1, cfg );

    // The wide box over a nonlinear flow must split.
    EXPECT_GE( sol.tree().done().size(), 2u );

    // Rigorous end-to-end containment: for a sweep of physical ICs, the
    // located leaf's interval enclosure must contain the true flow.
    for ( double dx : { -0.38, -0.2, 0.0, 0.17, 0.38 } )
        for ( double dv : { -0.38, -0.11, 0.0, 0.29, 0.38 } )
        {
            const Eigen::Vector2d ic{ 1.0 + dx, 0.0 + dv };
            const auto enc = sol.evaluate( ic );
            ASSERT_TRUE( enc.has_value() )
                << "no leaf claims IC (" << ic( 0 ) << "," << ic( 1 ) << ")";
            const Eigen::Vector2d truth = scalarReference( ic, t1 );
            EXPECT_TRUE( ( *enc )( 0 ).contains( truth( 0 ) ) )
                << "y1 enclosure [" << ( *enc )( 0 ).lower() << ", " << ( *enc )( 0 ).upper()
                << "] misses " << truth( 0 );
            EXPECT_TRUE( ( *enc )( 1 ).contains( truth( 1 ) ) )
                << "y2 enclosure [" << ( *enc )( 1 ).lower() << ", " << ( *enc )( 1 ).upper()
                << "] misses " << truth( 1 );
        }

    // Every done leaf carries a genuine (validated, nonzero) remainder.
    for ( int idx : sol.tree().done() )
    {
        const auto& leaf = sol.tree().leaf( idx );
        EXPECT_GT( leaf.payload( 0 ).remainder().width(), 0.0 );
        EXPECT_LT( leaf.payload( 0 ).remainder().width(), 5e-3 );
    }
}

TEST( AdsTaylorModel, SplitTilesTheParentAndKeepsTheRemainder )
{
    tax::domain::Box< double, M > box{ { 0.0, 0.0 }, { 1.0, 1.0 } };
    Eigen::Matrix< double, D, 1 > x0;
    x0 << 0.0, 0.0;
    State s = tax::domain::createModel< P >( box, x0 );
    // Curvature + a remainder to carry through the split.
    tax::MultiIndex< M > q{};
    q[0] = 2;
    s( 0 ).polynomial()[TM::scheme::flatOf( q )] = 0.5;
    s( 0 ).remainder() += tax::model::Interval< double >{ -0.01, 0.01 };

    const auto [L, R] = tax::ads::split( s, 0 );

    // Child locals map onto the parent halves: L(ξ') == parent(-0.5 + 0.5 ξ').
    for ( double xi : { -1.0, -0.3, 0.0, 0.6, 1.0 } )
    {
        const std::array< double, M > cl{ xi, 0.2 };
        const std::array< double, M > pl{ -0.5 + 0.5 * xi, 0.2 };
        const std::array< double, M > pr{ 0.5 + 0.5 * xi, 0.2 };
        EXPECT_NEAR( L( 0 ).polynomial().eval( cl ), s( 0 ).polynomial().eval( pl ), 1e-14 );
        EXPECT_NEAR( R( 0 ).polynomial().eval( cl ), s( 0 ).polynomial().eval( pr ), 1e-14 );
    }

    // The remainder (and the normalized expansion parameter) carry over.
    EXPECT_EQ( L( 0 ).remainder(), s( 0 ).remainder() );
    EXPECT_EQ( R( 0 ).remainder(), s( 0 ).remainder() );
    EXPECT_EQ( L( 0 ).domain(), s( 0 ).domain() );
}

TEST( AdsTaylorModel, TruncationCriterionReadsThePolynomialParts )
{
    tax::domain::Box< double, M > box{ { 0.0, 0.0 }, { 1.0, 1.0 } };
    Eigen::Matrix< double, D, 1 > x0;
    x0 << 0.0, 0.0;
    State s = tax::domain::createModel< P >( box, x0 );

    tax::ads::TruncationCriterion crit{ /*tol=*/1e-9, /*maxDepth=*/8 };
    EXPECT_FALSE( crit.shouldSplit( s, 0 ) );  // identity: no top-degree mass

    // Add top-degree mass along axis 1 → split fires and picks axis 1.
    tax::MultiIndex< M > top{};
    top[1] = P;
    s( 0 ).polynomial()[TM::scheme::flatOf( top )] = 1e-3;
    EXPECT_TRUE( crit.shouldSplit( s, 0 ) );
    EXPECT_EQ( crit.splitDim( s ), 1 );
    // maxDepth still caps.
    EXPECT_FALSE( crit.shouldSplit( s, 8 ) );
}

TEST( AdsTaylorModel, ParallelMatchesSerial )
{
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-8;
    const double t1 = 1.5;

    auto run = [&]( int nt ) {
        return tax::ads::propagate< P >( tax::ode::methods::Picard{},
                                         tax::ads::TruncationCriterion{ 1e-3, 6 }, rhs(), icBox(),
                                         icCenter(), 0.0, t1, cfg, nt );
    };
    const auto serial = run( 1 );
    const auto parallel = run( 4 );

    auto ds = serial.tree().done();
    auto dp = parallel.tree().done();
    ASSERT_EQ( ds.size(), dp.size() );
    for ( std::size_t i = 0; i < ds.size(); ++i )
    {
        const auto& ls = serial.tree().leaf( ds[i] );
        const auto& lp = parallel.tree().leaf( dp[i] );
        for ( int j = 0; j < M; ++j )
        {
            EXPECT_DOUBLE_EQ( ls.domain.center( j ), lp.domain.center( j ) );
            EXPECT_DOUBLE_EQ( ls.domain.halfWidth( j ), lp.domain.halfWidth( j ) );
        }
        for ( int r = 0; r < D; ++r )
        {
            EXPECT_EQ( ls.payload( r ).remainder(), lp.payload( r ).remainder() );
            for ( std::size_t k = 0; k < TM::nCoefficients; ++k )
                EXPECT_DOUBLE_EQ( ls.payload( r ).polynomial()[k],
                                  lp.payload( r ).polynomial()[k] );
        }
    }
}

#else  // tax core without tax::model

#include <gtest/gtest.h>
TEST( AdsTaylorModel, SkippedTaxCoreLacksModelModule ) { GTEST_SKIP(); }

#endif
