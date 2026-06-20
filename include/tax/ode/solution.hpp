// include/tax/ode/solution.hpp
//
// Solution<Stepper, State> — step-boundary states + events.
// Dense interpolation and the Dense template parameter have been removed.
// Use IntegratorConfig::save_steps = true (the default) to record the
// full accepted-step grid; save_steps = false keeps only the initial and
// final state.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace tax::ode
{

template < class State, class T >
struct EventRecord
{
    std::string label;  // "" if anonymous
    T t;
    State x;
};

template < class Stepper, class State >
class Solution
{
   public:
    using T = typename Stepper::T;
    std::vector< T > t;
    std::vector< State > x;
    std::vector< EventRecord< State, T > > events;

    [[nodiscard]] std::size_t size() const noexcept { return t.size(); }
};

}  // namespace tax::ode
