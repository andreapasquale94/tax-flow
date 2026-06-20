// include/tax/ode/events/grid_event.hpp
//
// GridEvent: stop the propagator at each of a sorted list of times. nextStop
// clamps the next step to land on the next grid time; onStep records the
// boundary state once the step has reached it. Because the step LANDS on the
// grid time, the recorded state is a genuine integrated boundary.

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <tax/ode/event.hpp>
#include <utility>
#include <vector>

namespace tax::ode
{

template < class State, class T >
class GridEvent final : public BaseEvent< GridEvent< State, T >, State, T >
{
   public:
    using Action = typename Event< State, T >::Action;

    GridEvent( std::vector< T > times, std::string name )
        : times_( std::move( times ) ), name_( std::move( name ) )
    {
    }

    [[nodiscard]] std::string name() const override { return name_; }

    [[nodiscard]] std::optional< T > nextStop( T t ) const override
    {
        for ( std::size_t i = cursor_; i < times_.size(); ++i )
            if ( times_[i] > t ) return times_[i];
        return std::nullopt;
    }

    Action onStep( Recorder< State, T >& rec ) override
    {
        const T t_new = this->eval_->tOld() + this->eval_->hUsed();
        // Tolerance relative to the step: the clamp targets an exact landing,
        // so the boundary coincides with the grid time up to rounding.
        const T tol = ( std::abs( t_new ) + this->eval_->hUsed() ) *
                          std::numeric_limits< T >::epsilon() * T{ 16 } +
                      std::numeric_limits< T >::min();
        while ( cursor_ < times_.size() && times_[cursor_] <= t_new + tol )
        {
            if ( std::abs( times_[cursor_] - t_new ) <= tol )
                rec.record( name_, t_new, this->eval_->xNew() );
            ++cursor_;
        }
        return Action::Continue;
    }

   private:
    std::vector< T > times_;
    mutable std::size_t cursor_ = 0;
    std::string name_;
};

}  // namespace tax::ode
