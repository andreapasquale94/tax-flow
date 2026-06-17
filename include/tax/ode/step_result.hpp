// include/tax/ode/step_result.hpp
//
// Result type returned by every Stepper's step() method.
// Adaptive fields (h_next, err_norm, accepted) are always present
// for layout simplicity; fixed-step Steppers (future) leave them at
// their defaults.

#pragma once

namespace tax::ode
{

template < class State, class Stepper >
struct StepResult
{
    State                          x_new{};
    typename Stepper::T            h_used{};
    typename Stepper::DenseData    dense{};
    typename Stepper::T            h_next{};
    typename Stepper::T            err_norm{};
    bool                           accepted = true;
};

}  // namespace tax::ode
