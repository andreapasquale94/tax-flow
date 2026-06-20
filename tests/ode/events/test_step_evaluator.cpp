// Standalone StepEvaluator::findRoot test using a hand-rolled evaluator whose
// eval is an analytic function — no integrator needed.
#include <gtest/gtest.h>

#include <cmath>
#include <tax/la/types.hpp>
#include <tax/ode/step_evaluator.hpp>

using tax::ode::Direction;

namespace
{
using State = tax::la::VecNT< 1, double >;

// eval(τ) = cos(t0 + τ); boundary over [t0, t0+h].
class CosEval : public tax::ode::StepEvaluator< State, double >
{
   public:
    CosEval( double t0, double h ) : t0_( t0 ), h_( h )
    {
        x_old_( 0 ) = std::cos( t0_ );
        x_new_( 0 ) = std::cos( t0_ + h_ );
    }
    const State& xOld() const noexcept override { return x_old_; }
    const State& xNew() const noexcept override { return x_new_; }
    double tOld() const noexcept override { return t0_; }
    double hUsed() const noexcept override { return h_; }
    State eval( double tau ) const override
    {
        State s;
        s( 0 ) = std::cos( t0_ + tau );
        return s;
    }

   private:
    double t0_, h_;
    State x_old_, x_new_;
};
}  // namespace

TEST( StepEvaluator, FindRootLocatesCosineZero )
{
    // cos crosses zero (decreasing) at t = π/2. Bracket [1.0, 2.0].
    CosEval eval{ 1.0, 1.0 };
    auto g = []( const State& x, double ) { return x( 0 ); };
    auto tau = eval.findRoot( g, Direction::Decreasing );
    ASSERT_TRUE( tau.has_value() );
    EXPECT_NEAR( eval.tOld() + *tau, M_PI / 2.0, 1e-9 );
}

TEST( StepEvaluator, FindRootReturnsNulloptWhenNoCrossing )
{
    CosEval eval{ 0.0, 0.5 };  // cos stays positive on [0, 0.5]
    auto g = []( const State& x, double ) { return x( 0 ); };
    EXPECT_FALSE( eval.findRoot( g, Direction::Any ).has_value() );
}
