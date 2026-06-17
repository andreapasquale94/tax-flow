// tests/ode/testConcepts.cpp
//
// Compile-time validation of the Stepper concept hierarchy:
//   1. FakeStepper satisfies concepts::Stepper.
//   2. FakeAdaptiveStepper (which declares `static constexpr bool
//      is_adaptive = true`) additionally satisfies AdaptiveStepper.
//   3. FakeStepper does NOT satisfy AdaptiveStepper (lacks the marker).

#include <gtest/gtest.h>

#include <tax/la/types.hpp>
#include <functional>

#include <tax/ode.hpp>

namespace
{

using State = tax::la::VecNT< 1, double >;

// FakeStepper satisfies concepts::Stepper but does NOT declare the
// `static constexpr bool is_adaptive` marker, so it must not satisfy
// AdaptiveStepper.
struct FakeStepper
{
    using T         = double;
    using State     = ::State;
    using Config    = tax::ode::IntegratorConfig< T >;
    using Rhs       = std::function< State( const State&, T ) >;
    using DenseData = State;

    tax::ode::StepResult< State, FakeStepper >
    step( const Rhs& /*f*/, const State& x, T /*t*/, T h, const Config& /*cfg*/ ) const
    {
        tax::ode::StepResult< State, FakeStepper > r;
        r.x_new = x;
        r.h_used = h;
        r.dense = x;
        return r;
    }

    static State eval_dense( const DenseData& d, const T& /*t0*/, const T& /*tq*/ )
    {
        return d;
    }
};

static_assert( tax::ode::concepts::Stepper< FakeStepper >,
               "FakeStepper should satisfy concepts::Stepper" );
static_assert( !tax::ode::concepts::AdaptiveStepper< FakeStepper >,
               "FakeStepper should NOT satisfy concepts::AdaptiveStepper" );

// FakeAdaptiveStepper opts in to the adaptive concept via the
// `is_adaptive` marker, so AdaptiveStepper<FakeAdaptiveStepper> holds.
struct FakeAdaptiveStepper
{
    using State     = ::State;
    using T         = double;
    using Config    = tax::ode::IntegratorConfig< T >;
    using Rhs       = std::function< State( const State&, T ) >;
    using DenseData = State;

    static constexpr bool is_adaptive = true;

    tax::ode::StepResult< State, FakeAdaptiveStepper >
    step( const Rhs&, const State& x, T, T h, const Config& ) const
    {
        tax::ode::StepResult< State, FakeAdaptiveStepper > r;
        r.x_new   = x;
        r.h_used  = h;
        r.dense   = x;
        r.h_next  = h;
        r.err_norm = T{ 0 };
        r.accepted = true;
        return r;
    }

    static State eval_dense( const DenseData& d, const T&, const T& )
    {
        return d;
    }
};

static_assert( tax::ode::concepts::Stepper< FakeAdaptiveStepper >,
               "FakeAdaptiveStepper should satisfy concepts::Stepper" );
static_assert( tax::ode::concepts::AdaptiveStepper< FakeAdaptiveStepper >,
               "FakeAdaptiveStepper should satisfy concepts::AdaptiveStepper" );

TEST( OdeConcepts, FakeStepperCompiles )
{
    FakeStepper s;
    FakeStepper::Rhs f = []( const State& x, double ) { return x; };
    State x = State::Zero();
    auto r = s.step( f, x, 1.0, 0.1, tax::ode::IntegratorConfig< double >{} );
    EXPECT_DOUBLE_EQ( r.h_used, 0.1 );
}

}  // namespace
