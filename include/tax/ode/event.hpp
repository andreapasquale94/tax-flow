// include/tax/ode/event.hpp
//
// Event<State,T>: a first-class, typed event. Each event IS its own trigger
// (onStep decides if/where it fires and returns an Action) and may constrain
// the next step (nextStop). Events record results into the Solution through a
// Recorder. The Integrator binds each event to its shared StepEvaluator via
// setEvaluator. Built-in events live in tax/ode/events/*.hpp.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <tax/ode/solution.hpp>        // EventRecord
#include <tax/ode/step_evaluator.hpp>  // StepEvaluator, Direction
#include <utility>
#include <vector>

namespace tax::ode
{

// Thin handle over the Solution's event list. All event output funnels here.
template < class State, class T >
class Recorder
{
   public:
    explicit Recorder( std::vector< EventRecord< State, T > >* events ) : events_( events ) {}

    void record( std::string name, T t, State x )
    {
        events_->push_back( { std::move( name ), t, std::move( x ) } );
    }

    [[nodiscard]] std::size_t count() const noexcept { return events_->size(); }

   private:
    std::vector< EventRecord< State, T > >* events_;
};

template < class State, class T >
class Event
{
   public:
    enum class Action
    {
        Continue,
        Terminate
    };

    virtual ~Event() = default;

    // Identifier used as the label of every record this event emits.
    [[nodiscard]] virtual std::string name() const = 0;

    // Pre-step hook: "do not let the next step advance past this absolute
    // time." nullopt = no constraint. Default: no constraint.
    [[nodiscard]] virtual std::optional< T > nextStop( T /*t*/ ) const { return std::nullopt; }

    // Post-step hook: called once per accepted step (reads eval_, writes rec).
    virtual Action onStep( Recorder< State, T >& rec ) = 0;

    // Deep copy (per-leaf instances for the parallel ADS driver).
    [[nodiscard]] virtual std::shared_ptr< Event > clone() const = 0;

    void setEvaluator( const std::shared_ptr< const StepEvaluator< State, T > >& e ) { eval_ = e; }

   protected:
    std::shared_ptr< const StepEvaluator< State, T > > eval_;
};

// CRTP helper: gives Derived a clone() via its copy constructor.
template < class Derived, class State, class T >
class BaseEvent : public Event< State, T >
{
   public:
    [[nodiscard]] std::shared_ptr< Event< State, T > > clone() const override
    {
        return std::make_shared< Derived >( static_cast< const Derived& >( *this ) );
    }
};

}  // namespace tax::ode
