// tests/ode/testSolution.cpp
//
// Verifies that both Solution specialisations (Dense=false and
// Dense=true) exist with the expected member layout, that appending
// steps/events works, and that operator()(t_query) on the dense form
// throws on empty/out-of-range inputs. FakeStepper is defined locally
// and is compatible with concepts::Stepper.

#include <gtest/gtest.h>

#include <tax/la/types.hpp>
#include <functional>

#include <tax/ode.hpp>

namespace
{

using State = tax::la::VecNT< 2, double >;

struct FakeStepper
{
    using State     = ::State;
    using T         = double;
    using Config    = tax::ode::IntegratorConfig< T >;
    using Rhs       = std::function< State( const State&, T ) >;
    using DenseData = State;

    static constexpr bool has_dense_output = true;

    tax::ode::StepResult< State, FakeStepper >
    step( const Rhs&, const State& x, T, T h, const Config& ) const
    {
        tax::ode::StepResult< State, FakeStepper > r;
        r.x_new = x; r.h_used = h; r.dense = x;
        return r;
    }
    static State eval_dense( const DenseData& d, const T&, const T& )
    {
        return d;
    }
};

}  // namespace

// Helper trait — must be a template so the requires expression is
// dependent; GCC 15 hard-errors on non-dependent ill-formed member access
// inside requires even though the standard says it should just be false.
template < class T >
constexpr bool has_dense_member = requires( const T& t ) { t.dense; };

TEST( OdeSolution, DiscreteHasNoDenseStorage )
{
    using Sol = tax::ode::Solution< FakeStepper, State, /*Dense=*/false >;
    Sol s;
    s.t.push_back( 0.0 );
    s.x.push_back( State{ 1.0, 2.0 } );
    s.events.push_back( { "tag", 0.5, State{ 3.0, 4.0 } } );

    EXPECT_EQ( s.t.size(), 1u );
    EXPECT_EQ( s.x.size(), 1u );
    EXPECT_EQ( s.events.size(), 1u );
    EXPECT_EQ( s.size(), 1u );
    // The discrete specialisation must not expose a `dense` vector.
    static_assert( !has_dense_member< Sol >,
                   "Discrete Solution must not have a dense field" );
}

TEST( OdeSolution, DenseExposesDenseAndOperatorCall )
{
    using Sol = tax::ode::Solution< FakeStepper, State, /*Dense=*/true >;
    Sol s;
    s.t = { 0.0, 1.0 };
    s.x = { State{ 1.0, 2.0 }, State{ 3.0, 4.0 } };
    s.dense.push_back( State{ 1.0, 2.0 } );

    // operator()(t_query) is defined and calls eval_dense.
    State at_t0 = s( 0.0 );
    EXPECT_DOUBLE_EQ( at_t0( 0 ), 1.0 );
    EXPECT_DOUBLE_EQ( at_t0( 1 ), 2.0 );
}

TEST( OdeSolution, DenseOperatorThrowsOnEmpty )
{
    using Sol = tax::ode::Solution< FakeStepper, State, /*Dense=*/true >;
    Sol s;
    EXPECT_THROW( s( 0.0 ), std::runtime_error );
}

TEST( OdeSolution, DenseOperatorThrowsOnOutOfRange )
{
    using Sol = tax::ode::Solution< FakeStepper, State, /*Dense=*/true >;
    Sol s;
    s.t = { 0.0, 1.0 };
    s.x = { State{ 1.0, 2.0 }, State{ 3.0, 4.0 } };
    s.dense.push_back( State{ 1.0, 2.0 } );

    EXPECT_THROW( s( -0.1 ), std::out_of_range );
    EXPECT_THROW( s(  1.1 ), std::out_of_range );
}
