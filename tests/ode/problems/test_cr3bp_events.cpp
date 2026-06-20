// tests/ode/testCR3BPEvents.cpp
//
// CR3BP propagation with ZeroCrossing events:
//   - L1 crossing  (x crosses x_L1 going +x)
//   - lunar periapsis  ((x - 1 + μ)·vx + y·vy crosses zero going +)
//   - L2 crossing  (x crosses x_L2 going +x)
// Assertions: at least one L1 event before the lunar loop, at least
// one moon-periapsis event within distance 0.1 of the Moon, and at
// least one L2 event before T_final. Jacobi constant is verified to
// be preserved through the event machinery.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <vector>

#include "cr3bp_problem.hpp"

using namespace tax::ode::test;
using tax::ode::Direction;
using tax::ode::Event;
using tax::ode::IntegratorConfig;
using tax::ode::Record;
using tax::ode::TaylorStepper;
using tax::ode::ZeroCrossing;

TEST( OdeCR3BPEvents, TaylorRecordsL1MoonL2 )
{
    constexpr int N = 16;
    using State = CR3BPState;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-13;

    using Stepper = TaylorStepper< N, State >;
    std::vector< Event< Stepper > > events;

    events.emplace_back(
        ZeroCrossing( []( const auto& s, const auto& /*t*/ ) { return s( 0 ) - kCR3BPL1; },
                      Direction::Increasing ),
        Record( "L1" ) );

    const double mu = kCR3BPMu;
    events.emplace_back( ZeroCrossing(
                             [mu]( const auto& s, const auto& /*t*/ ) {
                                 return ( s( 0 ) - 1.0 + mu ) * s( 2 ) + s( 1 ) * s( 3 );
                             },
                             Direction::Increasing ),
                         Record( "moon_periapsis" ) );

    events.emplace_back(
        ZeroCrossing( []( const auto& s, const auto& /*t*/ ) { return s( 0 ) - kCR3BPL2; },
                      Direction::Increasing ),
        Record( "L2" ) );

    tax::ode::Taylor< N, State, tax::ode::controllers::JorbaZou< double >, decltype( cr3bp_rhs() ) >
        integ{ cr3bp_rhs(), cfg, std::move( events ) };
    auto sol = integ.integrate( cr3bp_transit_ic(), 0.0, kCR3BPTFinal );

    const auto countLabel = [&]( const char* lbl ) {
        return std::count_if( sol.events.begin(), sol.events.end(),
                              [lbl]( const auto& e ) { return e.label == lbl; } );
    };
    EXPECT_GE( countLabel( "L1" ), 1 );
    EXPECT_GE( countLabel( "moon_periapsis" ), 1 );
    EXPECT_GE( countLabel( "L2" ), 1 );

    // At least one moon-periapsis event has r2 < 0.1.
    bool close_moon = false;
    for ( const auto& e : sol.events )
    {
        if ( e.label != "moon_periapsis" ) continue;
        const double r2 = std::hypot( e.x( 0 ) - 1.0 + mu, e.x( 1 ) );
        if ( r2 < 0.1 )
        {
            close_moon = true;
            break;
        }
    }
    EXPECT_TRUE( close_moon );

    // Jacobi constant preserved through event machinery.
    const double C0 = cr3bp_jacobi( sol.x.front() );
    const double C1 = cr3bp_jacobi( sol.x.back() );
    EXPECT_NEAR( C1, C0, 1e-10 );
}
