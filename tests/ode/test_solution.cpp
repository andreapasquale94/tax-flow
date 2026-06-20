// tests/ode/test_solution.cpp
//
// Verifies that Solution<Stepper, State> has the expected member layout,
// that appending steps/events works, and that size() reports correctly.

#include <gtest/gtest.h>

#include <functional>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>

namespace
{

using State = tax::la::VecNT< 2, double >;

struct FakeStepper
{
    using State = ::State;
    using T = double;
    using Config = tax::ode::IntegratorConfig< T >;
    using Rhs = std::function< State( const State&, T ) >;
    using StepData = State;

    static constexpr bool has_step_expansion = true;

    tax::ode::StepResult< State, FakeStepper > step( const Rhs&, const State& x, T, T h,
                                                     const Config& ) const
    {
        tax::ode::StepResult< State, FakeStepper > r;
        r.x_new = x;
        r.h_used = h;
        r.data = x;
        return r;
    }
};

}  // namespace

TEST( OdeSolution, DiscreteStorageWorks )
{
    using Sol = tax::ode::Solution< FakeStepper, State >;
    Sol s;
    s.t.push_back( 0.0 );
    s.x.push_back( State{ 1.0, 2.0 } );
    s.events.push_back( { "tag", 0.5, State{ 3.0, 4.0 } } );

    EXPECT_EQ( s.t.size(), 1u );
    EXPECT_EQ( s.x.size(), 1u );
    EXPECT_EQ( s.events.size(), 1u );
    EXPECT_EQ( s.size(), 1u );
}

TEST( OdeSolution, MultipleStepsAppend )
{
    using Sol = tax::ode::Solution< FakeStepper, State >;
    Sol s;
    s.t.push_back( 0.0 );
    s.x.push_back( State{ 1.0, 0.0 } );
    s.t.push_back( 0.5 );
    s.x.push_back( State{ 2.0, 0.0 } );
    s.t.push_back( 1.0 );
    s.x.push_back( State{ 3.0, 0.0 } );

    EXPECT_EQ( s.size(), 3u );
    EXPECT_DOUBLE_EQ( s.t.front(), 0.0 );
    EXPECT_DOUBLE_EQ( s.t.back(), 1.0 );
}
