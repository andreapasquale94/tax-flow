// tests/ads/test_split_event.cpp
//
// SplitEvent: fabricate a StepEvaluator with a known DA boundary state, route
// it through SplitEvent::onStep, verify the SplitRequest is populated and
// Action::Terminate is returned (and that a below-tolerance state continues).
#include <gtest/gtest.h>

#include <memory>
#include <tax/ads/split_criteria.hpp>
#include <tax/ads/split_event.hpp>
#include <tax/la/types.hpp>
#include <tax/ode/event.hpp>
#include <tax/ode/step_evaluator.hpp>
#include <tax/tax.hpp>

using tax::ads::SplitEvent;
using tax::ads::SplitRequest;
using tax::ads::TruncationCriterion;

namespace
{
using TE = tax::TE< 3, 2 >;
using State = tax::la::VecNT< 2, TE >;
using Action = tax::ode::Event< State, double >::Action;

State makeQuadraticState()
{
    Eigen::Vector2d p{ 0.0, 0.0 };
    auto vars = tax::la::variables< TE >( p );
    auto x = vars[0];
    auto y = vars[1];
    State F;
    F( 0 ) = x + 0.5 * x * x + x * x * y;  // cubic top-degree term ⇒ exceeds 1e-3
    F( 1 ) = y;
    return F;
}

// Boundary-only evaluator: eval is unused by SplitEvent.
class BoundaryEval : public tax::ode::StepEvaluator< State, double >
{
   public:
    BoundaryEval( State x_old, double t_old, State x_new, double h )
        : x_old_( std::move( x_old ) ), x_new_( std::move( x_new ) ), t_old_( t_old ), h_( h )
    {
    }
    const State& xOld() const noexcept override { return x_old_; }
    const State& xNew() const noexcept override { return x_new_; }
    double tOld() const noexcept override { return t_old_; }
    double hUsed() const noexcept override { return h_; }
    State eval( double ) const override { return x_new_; }

   private:
    State x_old_, x_new_;
    double t_old_, h_;
};

std::shared_ptr< BoundaryEval > makeEval()
{
    auto s = makeQuadraticState();
    return std::make_shared< BoundaryEval >( s, 0.0, s, 0.1 );
}
}  // namespace

TEST( AdsSplitEvent, FiresAndTerminatesWhenCriterionAgrees )
{
    SplitRequest< double > req{};
    SplitEvent< State, double, TruncationCriterion > ev{ TruncationCriterion{ 1e-3 }, 0, &req };
    ev.setEvaluator( makeEval() );

    std::vector< tax::ode::EventRecord< State, double > > sink;
    tax::ode::Recorder< State, double > rec{ &sink };

    EXPECT_EQ( ev.onStep( rec ), Action::Terminate );
    EXPECT_TRUE( req.fired );
    EXPECT_EQ( req.dim, 0 );
    EXPECT_DOUBLE_EQ( req.t, 0.1 );
}

TEST( AdsSplitEvent, ContinuesBelowTolerance )
{
    SplitRequest< double > req{};
    SplitEvent< State, double, TruncationCriterion > ev{ TruncationCriterion{ 1.0 }, 0, &req };
    ev.setEvaluator( makeEval() );

    std::vector< tax::ode::EventRecord< State, double > > sink;
    tax::ode::Recorder< State, double > rec{ &sink };

    EXPECT_EQ( ev.onStep( rec ), Action::Continue );
    EXPECT_FALSE( req.fired );
}
