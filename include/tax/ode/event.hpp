// include/tax/ode/event.hpp
//
// Direction / ControlFlow enums, FlowRef, TriggerContext, StepperCtx, and the
// type-erased Event<Stepper> class. Trigger and Action factories live
// in triggers.hpp / actions.hpp; the EventStorage they use is defined in
// actions.hpp and forward-declared at the bottom of this file.

#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <tax/ode/step_result.hpp>
#include <type_traits>
#include <utility>

namespace tax::ode
{

enum class Direction
{
    Increasing,
    Decreasing,
    Any
};
enum class ControlFlow
{
    Continue,
    Terminate
};

// Non-owning type-erased callable: State(T tau). No per-step heap allocation.
template < class State, class T >
class FlowRef
{
   public:
    FlowRef() = default;
    template < class Fn >
        requires( !std::is_same_v< std::remove_cvref_t< Fn >, FlowRef > )
    FlowRef( Fn& fn ) noexcept
        : obj_( static_cast< void* >( std::addressof( fn ) ) ),
          call_( []( void* o, T tau ) -> State { return ( *static_cast< Fn* >( o ) )( tau ); } )
    {
    }
    State operator()( T tau ) const { return call_( obj_, tau ); }
    [[nodiscard]] explicit operator bool() const noexcept { return call_; }

   private:
    void* obj_ = nullptr;
    State ( *call_ )( void*, T ) = nullptr;
};

// Plain TriggerContext — Stepper-agnostic view of the step boundary.
// The Integrator passes a StepperCtx (below) which adds the Stepper
// typedef for triggers that need it.
template < class State_, class T_ >
struct TriggerContext
{
    using State_type = State_;
    using T_type = T_;

    const State_& x_old;
    T_ t_old;
    const State_& x_new;
    T_ h_used;
    FlowRef< State_, T_ > flow;  // state at t_old + τ
};

// StepperCtx — Stepper-aware view of TriggerContext used inside the
// integrator.
template < class Stepper, class State_, class T_ >
struct StepperCtx : TriggerContext< State_, T_ >
{
    using Stepper_type = Stepper;

    explicit StepperCtx( const State_& xo, T_ to, const State_& xn, T_ hu,
                         FlowRef< State_, T_ > fl )
        : TriggerContext< State_, T_ >{ xo, to, xn, hu, fl }
    {
    }
};

// Forward declaration of EventStorage (definition in actions.hpp).
template < class State, class T >
struct EventStorage;

// Event<Stepper>
//
// Type-erased pair of {Trigger, Action}. Triggers and Actions are
// stored as std::function-erased callables. Any state captured by
// reference inside a Trigger or Action lambda (e.g. counters, output
// buffers) MUST outlive the call to Integrator::integrate(); references
// dangling at trigger-evaluation time produce UB.
//
// Stepper is fixed at the Event's instantiation point so the Trigger /
// Action callables can statically rely on Stepper types.
template < class Stepper >
class Event
{
   public:
    using T = typename Stepper::T;
    using State = typename Stepper::State;
    using Ctx = StepperCtx< Stepper, State, T >;
    using Storage = EventStorage< State, T >;
    using TriggerFn = std::function< std::optional< T >( const Ctx& ) >;
    using ActionFn = std::function< ControlFlow( const Ctx&, T, Storage& ) >;

    template < class Trig, class Act >
    Event( Trig trig, Act act ) : trigger_( std::move( trig ) ), action_( std::move( act ) )
    {
    }

    [[nodiscard]] std::optional< T > test( const Ctx& ctx ) const { return trigger_( ctx ); }

    ControlFlow run( const Ctx& ctx, T tau_fired, Storage& storage ) const
    {
        return action_( ctx, tau_fired, storage );
    }

   private:
    TriggerFn trigger_;
    ActionFn action_;
};

}  // namespace tax::ode
