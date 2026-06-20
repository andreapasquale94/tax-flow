// GridEvent: the integrator must land exactly on each requested grid time and
// record the boundary state there.
#include <gtest/gtest.h>

#include <cmath>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <vector>

using tax::ode::IntegratorConfig;

TEST( OdeEventsGrid, StopsAtRequestedTimes )
{
    constexpr int N = 12;
    using State = tax::la::VecNT< 1, double >;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    // x' = x, x(0) = 1 ⇒ x(t) = e^t.
    const auto f = []( const auto& x, const auto& ) { return x; };

    tax::ode::Taylor< N, State, tax::ode::controllers::JorbaZou< double >, decltype( f ) > integ{
        f, cfg };
    integ.addGridEvent( { 0.25, 0.5, 0.75 }, "grid" );

    State x0;
    x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    ASSERT_EQ( sol.events.size(), 3u );
    const double want_t[3] = { 0.25, 0.5, 0.75 };
    for ( int i = 0; i < 3; ++i )
    {
        EXPECT_EQ( sol.events[i].label, "grid" );
        EXPECT_NEAR( sol.events[i].t, want_t[i], 1e-12 );
        EXPECT_NEAR( sol.events[i].x( 0 ), std::exp( want_t[i] ), 1e-10 );
    }

    // The grid times appear as genuine boundaries in the step grid.
    auto on_grid = []( const std::vector< double >& ts, double v ) {
        for ( double t : ts )
            if ( std::abs( t - v ) < 1e-12 ) return true;
        return false;
    };
    EXPECT_TRUE( on_grid( sol.t, 0.25 ) );
    EXPECT_TRUE( on_grid( sol.t, 0.5 ) );
    EXPECT_TRUE( on_grid( sol.t, 0.75 ) );
}
