// tests/ads/test_split_event.cpp
//
// SplitTrigger / SplitAction round-trip: fabricate a TriggerContext with
// a known DA state, route through the trigger and action, verify the
// SplitRequest is populated and ControlFlow::Terminate is returned.

#include <gtest/gtest.h>

#include <tax/ads/criteria.hpp>
#include <tax/ads/split_event.hpp>
#include <tax/la/types.hpp>
#include <tax/ode/actions.hpp>
#include <tax/ode/event.hpp>
#include <tax/tax.hpp>

using tax::ads::SplitAction;
using tax::ads::SplitRequest;
using tax::ads::SplitTrigger;
using tax::ads::TruncationCriterion;
using tax::ode::ControlFlow;
using tax::ode::EventStorage;
using tax::ode::TriggerContext;

namespace
{
using TE        = tax::TE< 3, 2 >;
using State     = tax::la::VecNT< 2, TE >;
using DenseData = State;   // any matrix-of-TE works as fake DenseData

State makeQuadraticState()
{
    Eigen::Vector2d p{ 0.0, 0.0 };
    auto vars = tax::la::variables< TE >( p );
    auto x = vars[ 0 ];
    auto y = vars[ 1 ];
    State F;
    // x + 0.5*x^2 + x^2*y: the cubic term x^2*y gives nonzero top-degree (N=3)
    // mass > 1e-3, so TruncationCriterion fires for the trigger test.
    F( 0 ) = x + 0.5 * x * x + x * x * y;
    F( 1 ) = y;
    return F;
}
}  // namespace

TEST( AdsSplitEvent, TriggerFiresAtBoundaryWhenCriterionAgrees )
{
    auto    state  = makeQuadraticState();
    auto    state2 = state;
    DenseData dense = state;
    TriggerContext< State, double, DenseData > ctx{ state, 0.0, state2, 0.1, dense };

    TruncationCriterion crit{ /*tol=*/1e-3 };
    auto                trig = SplitTrigger( crit, /*depth=*/0 );
    auto                tau  = trig( ctx );
    ASSERT_TRUE( tau.has_value() );
    EXPECT_DOUBLE_EQ( *tau, 0.1 );
}

TEST( AdsSplitEvent, TriggerSilentBelowTol )
{
    auto      state = makeQuadraticState();
    auto      s2    = state;
    DenseData dense = state;
    TriggerContext< State, double, DenseData > ctx{ state, 0.0, s2, 0.1, dense };

    TruncationCriterion crit{ /*tol=*/1.0 };   // nothing exceeds it
    auto                trig = SplitTrigger( crit, /*depth=*/0 );
    auto                tau  = trig( ctx );
    EXPECT_FALSE( tau.has_value() );
}

TEST( AdsSplitEvent, ActionRecordsRequestAndTerminates )
{
    auto      state = makeQuadraticState();
    auto      s2    = state;
    DenseData dense = state;
    TriggerContext< State, double, DenseData > ctx{ state, 0.0, s2, 0.1, dense };

    TruncationCriterion crit{ /*tol=*/1e-3 };
    SplitRequest< double > req{};
    auto                  act = SplitAction( crit, &req );
    EventStorage< State, double > storage{ /*events=*/nullptr };
    auto cf = act( ctx, /*tau=*/0.1, storage );
    EXPECT_EQ( cf, ControlFlow::Terminate );
    EXPECT_TRUE( req.fired );
    EXPECT_EQ( req.dim, 0 );
    EXPECT_DOUBLE_EQ( req.t, 0.1 );
}
