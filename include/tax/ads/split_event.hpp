// include/tax/ads/split_event.hpp
//
// Interop with tax::ode::Event. SplitEvent fires at the step boundary iff the
// criterion demands a split; it writes {fired, dim, t} into the driver-owned
// SplitRequest and returns Action::Terminate so the integrator truncates the
// solution at the boundary. The driver then consumes req to decide split-vs-done.

#pragma once

#include <string>
#include <tax/ode/event.hpp>
#include <utility>

namespace tax::ads
{

template < class T >
struct SplitRequest
{
    bool fired = false;
    int dim = -1;
    T t = T{ 0 };
};

template < class State, class T, class Criterion >
class SplitEvent final : public tax::ode::BaseEvent< SplitEvent< State, T, Criterion >, State, T >
{
   public:
    using Action = typename tax::ode::Event< State, T >::Action;

    SplitEvent( Criterion crit, int depth, SplitRequest< T >* out )
        : crit_( std::move( crit ) ), depth_( depth ), out_( out )
    {
    }

    [[nodiscard]] std::string name() const override { return "ads:split"; }

    Action onStep( tax::ode::Recorder< State, T >& /*rec*/ ) override
    {
        const State& x_new = this->eval_->xNew();
        if ( !crit_.shouldSplit( x_new, depth_ ) ) return Action::Continue;
        out_->fired = true;
        out_->dim = crit_.splitDim( x_new );
        out_->t = this->eval_->tOld() + this->eval_->hUsed();
        return Action::Terminate;
    }

   private:
    Criterion crit_;
    int depth_;
    SplitRequest< T >* out_;
};

}  // namespace tax::ads
