// include/tax/ode/event.hpp
//
// Direction / ControlFlow enums, TriggerContext, StepperCtx, and the
// type-erased Event<Stepper> class. Trigger and Action factories live
// in triggers.hpp / actions.hpp; the EventStorage they use is defined in
// actions.hpp and forward-declared at the bottom of this file.

#pragma once

#include <functional>
#include <optional>
#include <utility>

#include <tax/ode/step_result.hpp>

namespace tax::ode
{

enum class Direction   { Increasing, Decreasing, Any };
enum class ControlFlow { Continue, Terminate };

// Plain TriggerContext — Stepper-agnostic view of the step boundary.
// The Integrator passes a StepperCtx (below) which adds the Stepper
// typedef for triggers that need it (e.g. the Brent fallback in
// ZeroCrossing routes through Stepper::eval_dense).
template < class State_, class T_, class DenseData_ >
struct TriggerContext
{
    using State_type     = State_;
    using T_type         = T_;
    using DenseData_type = DenseData_;

    const State_&     x_old;
    T_                t_old;
    const State_&     x_new;
    T_                h_used;
    const DenseData_& dense;
};

// StepperCtx — Stepper-aware view of TriggerContext used inside the
// integrator so triggers can route through Stepper::eval_dense.
template < class Stepper, class State_, class T_, class DenseData_ >
struct StepperCtx : TriggerContext< State_, T_, DenseData_ >
{
    using Stepper_type = Stepper;

    explicit StepperCtx( const State_& xo, T_ to, const State_& xn, T_ hu,
                         const DenseData_& d )
        : TriggerContext< State_, T_, DenseData_ >{ xo, to, xn, hu, d }
    {
    }
};

// Forward declaration of EventStorage (definition in actions.hpp).
template < class State, class T > struct EventStorage;

// Event<Stepper>
//
// Type-erased pair of {Trigger, Action}. Triggers and Actions are
// stored as std::function-erased callables. Any state captured by
// reference inside a Trigger or Action lambda (e.g. counters, output
// buffers) MUST outlive the call to Integrator::integrate(); references
// dangling at trigger-evaluation time produce UB.
//
// Stepper is fixed at the Event's instantiation point so the Trigger /
// Action callables can statically rely on Stepper::eval_dense,
// Stepper::DenseData, etc.
template < class Stepper >
class Event
{
public:
    using T         = typename Stepper::T;
    using State     = typename Stepper::State;
    using DenseData = typename Stepper::DenseData;
    using Ctx       = StepperCtx< Stepper, State, T, DenseData >;
    using Storage      = EventStorage< State, T >;
    using TriggerFn = std::function< std::optional< T >( const Ctx& ) >;
    using ActionFn  = std::function< ControlFlow( const Ctx&, T, Storage& ) >;

    template < class Trig, class Act >
    Event( Trig trig, Act act )
        : trigger_( std::move( trig ) ), action_( std::move( act ) )
    {
    }

    [[nodiscard]] std::optional< T > test( const Ctx& ctx ) const
    {
        return trigger_( ctx );
    }

    ControlFlow run( const Ctx& ctx, T tau_fired, Storage& storage ) const
    {
        return action_( ctx, tau_fired, storage );
    }

private:
    TriggerFn trigger_;
    ActionFn  action_;
};

}  // namespace tax::ode
