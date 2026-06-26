// tests/ads/test_refine.cpp
//
// The "propagate-then-assess" refinement driver (tax::ads::refine):
//   * accepted leaves reproduce high-accuracy scalar references,
//   * tightening the tolerance yields a finer partition,
//   * the parallel and serial runs agree leaf-for-leaf,
// on a mildly nonlinear oscillator (same model as test_driver.cpp).

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <tax/ads/box.hpp>
#include <tax/ads/refine.hpp>
#include <tax/ads/refine_criteria.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <tax/tax.hpp>
#include <vector>

using tax::ads::Box;
using tax::ads::CoefficientMatchCriterion;
using tax::ads::VolumeRatioCriterion;
using tax::ode::IntegratorConfig;
using tax::ode::methods::Verner89;

namespace
{
constexpr int P = 6;
constexpr int M = 2;
constexpr int D = 2;

using ScState = tax::la::VecNT< D, double >;

// f(x, v) = (v, -x - 0.3 x^3). Generic State so it accepts scalar and DA.
auto rhs()
{
    return []( const auto& x, double ) {
        using S = std::decay_t< decltype( x ) >;
        S out{ x.size() };
        out( 0 ) = x( 1 );
        out( 1 ) = -x( 0 ) - 0.3 * x( 0 ) * x( 0 ) * x( 0 );
        return out;
    };
}

ScState scalarReference( ScState x0, double t1 )
{
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    tax::ode::Verner89< ScState > integ{ rhs(), cfg };
    auto sol = integ.integrate( x0, 0.0, t1 );
    return sol.x.back();
}

template < class Leaf >
ScState evalLeaf( const Leaf& leaf, const ScState& xi )
{
    std::array< double, M > xi_local{};
    for ( int j = 0; j < M; ++j )
        xi_local[static_cast< std::size_t >( j )] =
            ( xi( j ) - leaf.box.center( j ) ) / leaf.box.halfWidth( j );

    ScState out;
    for ( int row = 0; row < D; ++row )
    {
        double acc = 0.0;
        constexpr std::size_t Nc = tax::numMonomials( P, M );
        for ( std::size_t k = 0; k < Nc; ++k )
        {
            const auto alpha = tax::unflatIndex< M >( k );
            double term = leaf.payload( row )[k];
            for ( int j = 0; j < M; ++j )
                for ( int p = 0; p < alpha[static_cast< std::size_t >( j )]; ++p )
                    term *= xi_local[static_cast< std::size_t >( j )];
            acc += term;
        }
        out( row ) = acc;
    }
    return out;
}

Box< double, M > icBox()
{
    return Box< double, M >{ tax::la::VecNT< M, double >{ 1.0, 0.0 },
                             tax::la::VecNT< M, double >{ 0.5, 0.5 } };
}
}  // namespace

TEST( AdsRefine, VolumeRatioLeavesMatchReference )
{
    const double t1 = 2.0 * M_PI;
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    ScState center{ 1.0, 0.0 };

    auto tree =
        tax::ads::refine< P >( Verner89{}, VolumeRatioCriterion{ /*tol=*/1e-6, /*maxDepth=*/10 },
                               rhs(), icBox(), center, 0.0, t1, cfg );

    EXPECT_GE( tree.done().size(), 1u );

    const std::array< ScState, 5 > samples{ {
        { 1.0, 0.0 },
        { 1.3, -0.2 },
        { 0.6, 0.4 },
        { 1.4, 0.45 },
        { 0.55, -0.45 },
    } };

    for ( const auto& xi : samples )
    {
        auto idx = tree.locate( xi );
        ASSERT_TRUE( idx.has_value() );
        const ScState predicted = evalLeaf( tree.leaf( *idx ), xi );
        const ScState reference = scalarReference( xi, t1 );
        EXPECT_NEAR( predicted( 0 ), reference( 0 ), 1e-3 );
        EXPECT_NEAR( predicted( 1 ), reference( 1 ), 1e-3 );
    }
}

TEST( AdsRefine, CoefficientMatchLeavesMatchReference )
{
    const double t1 = 2.0 * M_PI;
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    ScState center{ 1.0, 0.0 };

    auto tree = tax::ads::refine< P >( Verner89{},
                                       CoefficientMatchCriterion{ /*tol=*/1e-4, /*maxDepth=*/8 },
                                       rhs(), icBox(), center, 0.0, t1, cfg );

    EXPECT_GE( tree.done().size(), 1u );

    const std::array< ScState, 4 > samples{ {
        { 1.0, 0.0 },
        { 1.3, -0.2 },
        { 0.6, 0.4 },
        { 0.55, -0.45 },
    } };

    for ( const auto& xi : samples )
    {
        auto idx = tree.locate( xi );
        ASSERT_TRUE( idx.has_value() );
        const ScState predicted = evalLeaf( tree.leaf( *idx ), xi );
        const ScState reference = scalarReference( xi, t1 );
        EXPECT_NEAR( predicted( 0 ), reference( 0 ), 1e-3 );
        EXPECT_NEAR( predicted( 1 ), reference( 1 ), 1e-3 );
    }
}

TEST( AdsRefine, FinerToleranceSplitsMore )
{
    const double t1 = 2.0 * M_PI;
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    ScState center{ 1.0, 0.0 };

    auto coarse = tax::ads::refine< P >( Verner89{}, VolumeRatioCriterion{ 1e-2, 8 }, rhs(),
                                         icBox(), center, 0.0, t1, cfg );
    auto fine = tax::ads::refine< P >( Verner89{}, VolumeRatioCriterion{ 1e-7, 8 }, rhs(), icBox(),
                                       center, 0.0, t1, cfg );

    EXPECT_GE( fine.done().size(), coarse.done().size() );
}

TEST( AdsRefine, ParallelMatchesSerial )
{
    const double t1 = 2.0 * M_PI;
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    ScState center{ 1.0, 0.0 };

    const VolumeRatioCriterion crit{ 1e-6, 8 };

    auto serial = tax::ads::refine< P >( Verner89{}, crit, rhs(), icBox(), center, 0.0, t1, cfg,
                                         /*num_threads=*/1 );
    auto parallel = tax::ads::refine< P >( Verner89{}, crit, rhs(), icBox(), center, 0.0, t1, cfg,
                                           /*num_threads=*/4 );

    ASSERT_EQ( serial.done().size(), parallel.done().size() );

    // canonicalizeDone() sorts leaves by box center, so the two runs line up.
    auto sIdx = serial.done();
    auto pIdx = parallel.done();
    for ( std::size_t i = 0; i < sIdx.size(); ++i )
    {
        const auto& ls = serial.leaf( sIdx[i] );
        const auto& lp = parallel.leaf( pIdx[i] );
        for ( int j = 0; j < M; ++j )
        {
            EXPECT_DOUBLE_EQ( ls.box.center( j ), lp.box.center( j ) );
            EXPECT_DOUBLE_EQ( ls.box.halfWidth( j ), lp.box.halfWidth( j ) );
        }
        constexpr std::size_t Nc = tax::numMonomials( P, M );
        for ( int row = 0; row < D; ++row )
            for ( std::size_t k = 0; k < Nc; ++k )
                EXPECT_DOUBLE_EQ( ls.payload( row )[k], lp.payload( row )[k] );
    }
}

// Aggressive multi-way refinement (split_dirs = 2 → 4 children per node)
// produces a valid partition whose leaves still reproduce the reference.
TEST( AdsRefine, AggressiveSplitMatchesReference )
{
    const double t1 = 2.0 * M_PI;
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    ScState center{ 1.0, 0.0 };

    auto tree = tax::ads::refine< P >( Verner89{}, VolumeRatioCriterion{ 1e-6, 10 }, rhs(), icBox(),
                                       center, 0.0, t1, cfg, /*num_threads=*/1, /*split_dirs=*/2 );

    EXPECT_GE( tree.done().size(), 4u );  // at least one 4-way split

    const std::array< ScState, 4 > samples{ {
        { 1.0, 0.0 },
        { 1.3, -0.2 },
        { 0.6, 0.4 },
        { 0.55, -0.45 },
    } };
    for ( const auto& xi : samples )
    {
        auto idx = tree.locate( xi );
        ASSERT_TRUE( idx.has_value() );
        const ScState predicted = evalLeaf( tree.leaf( *idx ), xi );
        const ScState reference = scalarReference( xi, t1 );
        EXPECT_NEAR( predicted( 0 ), reference( 0 ), 1e-3 );
        EXPECT_NEAR( predicted( 1 ), reference( 1 ), 1e-3 );
    }
}

TEST( AdsRefine, AggressiveParallelMatchesSerial )
{
    const double t1 = 2.0 * M_PI;
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    ScState center{ 1.0, 0.0 };

    const VolumeRatioCriterion crit{ 1e-6, 10 };
    auto serial = tax::ads::refine< P >( Verner89{}, crit, rhs(), icBox(), center, 0.0, t1, cfg,
                                         /*num_threads=*/1, /*split_dirs=*/2 );
    auto parallel = tax::ads::refine< P >( Verner89{}, crit, rhs(), icBox(), center, 0.0, t1, cfg,
                                           /*num_threads=*/4, /*split_dirs=*/2 );

    ASSERT_EQ( serial.done().size(), parallel.done().size() );
    auto sIdx = serial.done();
    auto pIdx = parallel.done();
    for ( std::size_t i = 0; i < sIdx.size(); ++i )
    {
        const auto& ls = serial.leaf( sIdx[i] );
        const auto& lp = parallel.leaf( pIdx[i] );
        for ( int j = 0; j < M; ++j )
        {
            EXPECT_DOUBLE_EQ( ls.box.center( j ), lp.box.center( j ) );
            EXPECT_DOUBLE_EQ( ls.box.halfWidth( j ), lp.box.halfWidth( j ) );
        }
    }
}
