// include/tax/ode/events/root_finding_event.hpp
//
// RootFindingEvent: locate a root of g(x(τ), t_old+τ) within the step via the
// integrator's StepEvaluator::findRoot. On a hit it records the located state
// and either continues (terminal=false) or terminates (terminal=true).

#pragma once

#include <string>
#include <tax/ode/event.hpp>
#include <tax/ode/step_evaluator.hpp>
#include <utility>

namespace tax::ode
{

template < class State, class T, class G >
class RootFindingEvent final : public BaseEvent< RootFindingEvent< State, T, G >, State, T >
{
   public:
    using Action = typename Event< State, T >::Action;

    RootFindingEvent( G g, Direction dir, std::string name, bool terminal = false )
        : g_( std::move( g ) ), dir_( dir ), name_( std::move( name ) ), terminal_( terminal )
    {
    }

    [[nodiscard]] std::string name() const override { return name_; }

    Action onStep( Recorder< State, T >& rec ) override
    {
        const auto tau = this->eval_->findRoot( g_, dir_ );
        if ( !tau ) return Action::Continue;
        rec.record( name_, this->eval_->tOld() + *tau, this->eval_->eval( *tau ) );
        return terminal_ ? Action::Terminate : Action::Continue;
    }

   private:
    G g_;
    Direction dir_;
    std::string name_;
    bool terminal_;
};

}  // namespace tax::ode
