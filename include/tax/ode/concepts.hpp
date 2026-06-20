// include/tax/ode/concepts.hpp
//
// Stepper concept hierarchy.
//   - Stepper:         minimum — take one step at the supplied h and
//                      expose a StepData typedef.
//   - AdaptiveStepper: refinement — embedded error estimator + retry
//                      loop, keyed off `static constexpr bool is_adaptive`.
//
// Steppers additionally declare `static constexpr bool has_step_expansion`
// (true for Taylor, false for RK) and the matching eval (via tax::la::eval)
// or step() member, exercised via `if constexpr` in the integrator's flow
// closure. The concept stays minimal; a missing member yields a clear
// compile error at the flow-closure site.

#pragma once

#include <concepts>
#include <tax/ode/step_result.hpp>
#include <utility>

namespace tax::ode::concepts
{

template < class S >
concept Stepper = requires( S s, typename S::Rhs f, typename S::State x, typename S::T t,
                            typename S::T h, const typename S::Config& cfg ) {
    typename S::State;
    typename S::T;
    typename S::Config;
    typename S::Rhs;
    typename S::StepData;

    { s.step( f, x, t, h, cfg ) } -> std::same_as< StepResult< typename S::State, S > >;
};

template < class S >
concept AdaptiveStepper = Stepper< S > && requires {
    { S::is_adaptive } -> std::convertible_to< bool >;
} && S::is_adaptive;

}  // namespace tax::ode::concepts
