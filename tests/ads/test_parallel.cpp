// tests/ads/test_parallel.cpp
//
// The parallel ADS driver (num_threads > 1) must produce exactly the
// same domain decomposition and flow-map payloads as the serial driver,
// in the same canonical order, and be deterministic run-to-run.

#include <gtest/gtest.h>

#include <cstddef>
#include <tax/ads/box.hpp>
#include <tax/ads/criteria.hpp>
#include <tax/ads/driver.hpp>
#include <tax/ads/tree.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <tax/tax.hpp>
#include <utility>

using tax::ads::AdsDriver;
using tax::ads::Box;
using tax::ads::TruncationCriterion;
using tax::ode::IntegratorConfig;
using tax::ode::Verner89Stepper;

namespace
{
constexpr int P = 6;
constexpr int M = 2;
constexpr int D = 2;

using TE = tax::TE< P, M >;
using DAState = tax::la::VecNT< D, TE >;
using Stepper = Verner89Stepper< DAState >;
using Tree = tax::ads::AdsTree< DAState, M, double >;

auto rhs()
{
    return []( const auto& x, double ) {
        using S = std::decay_t< decltype( x ) >;
        S out{ x.size() };
        out( 0 ) = x( 1 );
        out( 1 ) = -x( 0 ) - 0.1 * x( 0 ) * x( 0 ) * x( 0 );
        return out;
    };
}

Tree runWith( int num_threads )
{
    const double t1 = 2.0 * M_PI;
    Box< double, M > ic_box{ tax::la::VecNT< M, double >{ 1.0, 0.0 },
                             tax::la::VecNT< M, double >{ 1.0, 1.0 } };
    tax::la::VecNT< D, double > center;
    center << 1.0, 0.0;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    AdsDriver< Stepper, TruncationCriterion > driver{
        TruncationCriterion{ /*tol=*/1e-7, /*maxDepth=*/6 }, cfg, {}, num_threads };
    return driver.run( rhs(), ic_box, center, /*t0=*/0.0, t1 );
}

void expectTreesEqual( const Tree& a, const Tree& b )
{
    auto da = a.done();
    auto db = b.done();
    ASSERT_EQ( da.size(), db.size() );
    constexpr std::size_t Nc = tax::numMonomials( P, M );
    for ( std::size_t i = 0; i < da.size(); ++i )
    {
        const auto& la = a.leaf( da[i] );
        const auto& lb = b.leaf( db[i] );
        for ( int j = 0; j < M; ++j )
        {
            EXPECT_NEAR( la.box.center( j ), lb.box.center( j ), 1e-12 );
            EXPECT_NEAR( la.box.halfWidth( j ), lb.box.halfWidth( j ), 1e-12 );
        }
        for ( int r = 0; r < D; ++r )
            for ( std::size_t k = 0; k < Nc; ++k )
                EXPECT_NEAR( la.payload( r )[k], lb.payload( r )[k], 1e-12 )
                    << "leaf " << i << " row " << r << " coeff " << k;
    }
}
}  // namespace

TEST( AdsParallel, MatchesSerialAcrossThreadCounts )
{
    const Tree serial = runWith( 1 );
    EXPECT_GE( serial.done().size(), 2u );  // box is wide / tol tight -> several splits

    for ( int nt : { 2, 4, 8 } )
    {
        const Tree par = runWith( nt );
        expectTreesEqual( serial, par );
    }
}

TEST( AdsParallel, DeterministicRunToRun )
{
    const Tree a = runWith( 8 );
    const Tree b = runWith( 8 );
    expectTreesEqual( a, b );
}
