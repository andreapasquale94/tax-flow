// include/tax/ode/actions.hpp
//
// Action factories. Each factory returns a callable with the signature
//   ControlFlow(const StepperCtx<…>&, T tau_fired, EventStorage<State, T>&).
//
// The action is invoked by the Integrator after a Trigger has fired
// with the τ at which the firing took place. Actions decide whether
// the integration continues (ControlFlow::Continue) or terminates
// (ControlFlow::Terminate) and may push EventRecord entries into the
// Solution via the provided storage.

#pragma once

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <tax/ode/event.hpp>
#include <tax/ode/solution.hpp>

namespace tax::ode
{

// Storage that the Integrator hands to each Action so it can push
// EventRecords into the Solution. Defined here (one definition) and
// forward declared in event.hpp.
template < class State, class T >
struct EventStorage
{
    std::vector< EventRecord< State, T > >* events;

    void push( EventRecord< State, T > rec )
    {
        if ( events ) events->push_back( std::move( rec ) );
    }
};

// Continue — no-op action that always asks the Integrator to keep going.
inline auto Continue()
{
    return []< class Ctx, class T, class Storage >(
               const Ctx&, T, Storage& ) -> ControlFlow
    {
        return ControlFlow::Continue;
    };
}

// Terminate — always asks the Integrator to halt at the event.
inline auto Terminate()
{
    return []< class Ctx, class T, class Storage >(
               const Ctx&, T, Storage& ) -> ControlFlow
    {
        return ControlFlow::Terminate;
    };
}

// Record(label) — push an EventRecord with the Stepper's continuous
// extension evaluated at τ into the storage and continue. Using
// eval_dense gives machine-precision x_event when τ was located by
// polynomial-Newton on g_poly.
inline auto Record( std::string label )
{
    return [ lbl = std::move( label ) ]<
               class Ctx, class T, class Storage >(
               const Ctx& ctx, T tau, Storage& storage ) -> ControlFlow
    {
        using StorageState = std::remove_cvref_t< decltype( ctx.x_old ) >;
        // Use the Stepper's continuous extension to evaluate state at
        // the event time — gives machine-precision x_event when τ was
        // located by polynomial-Newton on g_poly.
        StorageState x_event = Ctx::Stepper_type::eval_dense(
            ctx.dense,
            ctx.t_old,
            ctx.t_old + tau );
        storage.push( { lbl, ctx.t_old + tau, std::move( x_event ) } );
        return ControlFlow::Continue;
    };
}

// Custom(fn) — wrap a user callable as an Action. The wrapped callable
// must accept (Ctx, T, Storage) and return ControlFlow.
template < class Fn >
auto Custom( Fn fn )
{
    return [ fn = std::move( fn ) ]<
               class Ctx, class T, class Storage >(
               const Ctx& ctx, T tau, Storage& storage ) -> ControlFlow
    {
        return fn( ctx, tau, storage );
    };
}

}  // namespace tax::ode
