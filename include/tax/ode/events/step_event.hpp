// include/tax/ode/events/step_event.hpp
//
// StepEvent: fires at every accepted step, recording the boundary state under
// its name. (The raw grid still lives in Solution::t/x via save_steps; this is
// the named event-stream view.)

#pragma once

#include <string>
#include <tax/ode/event.hpp>
#include <utility>

namespace tax::ode
{

template < class State, class T >
class StepEvent final : public BaseEvent< StepEvent< State, T >, State, T >
{
   public:
    using Action = typename Event< State, T >::Action;

    explicit StepEvent( std::string name ) : name_( std::move( name ) ) {}

    [[nodiscard]] std::string name() const override { return name_; }

    Action onStep( Recorder< State, T >& rec ) override
    {
        rec.record( name_, this->eval_->tOld() + this->eval_->hUsed(), this->eval_->xNew() );
        return Action::Continue;
    }

   private:
    std::string name_;
};

}  // namespace tax::ode
