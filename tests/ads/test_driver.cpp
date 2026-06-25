// tests/ads/test_driver.cpp
//
// End-to-end ADS propagation on a mildly nonlinear oscillator,
// verified against high-accuracy scalar reference propagations sampled
// from the initial-condition box.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <tax/ads/box.hpp>
#include <tax/ads/criteria.hpp>
#include <tax/ads/driver.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <tax/ode/event.hpp>
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
using ScState = tax::la::VecNT< D, double >;
using Stepper = Verner89Stepper< DAState >;

// f(x, v) = (v, -x - 0.1 x^3). Generic State so it accepts scalar and DA.
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

ScState scalarReference( ScState x0, double t1 )
{
    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;
    tax::ode::Verner89< ScState > integ{ rhs(), cfg };
    auto sol = integ.integrate( x0, 0.0, t1 );
    return sol.x.back();
}
}  // namespace

TEST( AdsDriver, MildlyNonlinearOscillatorMatchesReference )
{
    const double t1 = 2.0 * M_PI;

    Box< double, M > ic_box{ tax::la::VecNT< M, double >{ 1.0, 0.0 },
                             tax::la::VecNT< M, double >{ 0.5, 0.5 } };
    tax::la::VecNT< D, double > center;
    center << 1.0, 0.0;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    AdsDriver< Stepper, TruncationCriterion > driver{
        TruncationCriterion{ /*tol=*/1e-4, /*maxDepth=*/8 }, cfg };

    auto sol = driver.run( rhs(), ic_box, center, /*t0=*/0.0, t1 );
    const auto& tree = sol.tree();

    EXPECT_GE( tree.done().size(), 1u );

    using V2 = tax::la::VecNT< M, double >;
    const std::array< V2, 5 > samples{ {
        V2{ 1.0, 0.0 },
        V2{ 1.3, -0.2 },
        V2{ 0.6, 0.4 },
        V2{ 1.5, 0.5 },
        V2{ 0.5, -0.5 },
    } };

    for ( const auto& xi : samples )
    {
        auto idx = tree.leaf( xi );
        ASSERT_TRUE( idx.has_value() );
        const auto& leaf = tree.leaf( *idx );

        std::array< double, M > xi_local{};
        for ( int j = 0; j < M; ++j )
            xi_local[static_cast< std::size_t >( j )] =
                ( xi( j ) - leaf.box.center( j ) ) / leaf.box.halfWidth( j );

        ScState x_predicted;
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
            x_predicted( row ) = acc;
        }

        ScState ic;
        ic( 0 ) = xi[0];
        ic( 1 ) = xi[1];
        const ScState x_ref = scalarReference( ic, t1 );

        EXPECT_NEAR( x_predicted( 0 ), x_ref( 0 ), 1e-3 )
            << "row 0 mismatch at xi = (" << xi[0] << ", " << xi[1] << ")";
        EXPECT_NEAR( x_predicted( 1 ), x_ref( 1 ), 1e-3 )
            << "row 1 mismatch at xi = (" << xi[0] << ", " << xi[1] << ")";
    }
}

TEST( AdsDriver, ExtraUserEventIsForwarded )
{
    const double t1 = 0.5;
    Box< double, M > ic_box{ tax::la::VecNT< M, double >{ 1.0, 0.0 },
                             tax::la::VecNT< M, double >{ 0.05, 0.05 } };
    tax::la::VecNT< D, double > center;
    center << 1.0, 0.0;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-10;

    // Use a shared counter so that the cloned instance (per-leaf) still
    // increments the same tally: the driver deep-clones extras_, so the
    // BaseEvent copy constructor must carry the shared_ptr.
    auto counter = std::make_shared< int >( 0 );

    struct CountingEvent : tax::ode::BaseEvent< CountingEvent, DAState, double >
    {
        std::shared_ptr< int > n;
        explicit CountingEvent( std::shared_ptr< int > n_ ) : n( std::move( n_ ) ) {}
        [[nodiscard]] std::string name() const override { return "count"; }
        Action onStep( tax::ode::Recorder< DAState, double >& ) override
        {
            ++( *n );
            return Action::Continue;
        }
    };

    using ExtraEvt = AdsDriver< Stepper, TruncationCriterion >::ExtraEvt;
    ExtraEvt extras;
    extras.push_back( std::make_shared< CountingEvent >( counter ) );

    AdsDriver< Stepper, TruncationCriterion > driver{
        TruncationCriterion{ /*tol=*/1.0, /*maxDepth=*/0 },  // never split
        cfg, std::move( extras ) };

    auto sol = driver.run( rhs(), ic_box, center, 0.0, t1 );
    const auto& tree = sol.tree();
    EXPECT_EQ( tree.done().size(), 1u );
    EXPECT_GT( *counter, 0 );
}
