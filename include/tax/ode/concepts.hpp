// include/tax/ode/concepts.hpp
//
// Stepper concept hierarchy.
//   - Stepper:         minimum — take one step at the supplied h and
//                      expose a dense-output payload + eval_dense.
//   - AdaptiveStepper: refinement — embedded error estimator + retry
//                      loop, keyed off `static constexpr bool is_adaptive`.
//
// Steppers may additionally declare `static constexpr bool
// has_dense_output` to indicate whether eval_dense reproduces the
// method's full order (true) or only a cubic-Hermite fallback (false).
// The flag is informational; no concept gates on it.

#pragma once

#include <concepts>
#include <utility>

#include <tax/ode/step_result.hpp>

namespace tax::ode::concepts
{

template < class S >
concept Stepper = requires(
    S s,
    typename S::Rhs f,
    typename S::State x,
    typename S::T t,
    typename S::T h,
    const typename S::Config& cfg )
{
    typename S::State;
    typename S::T;
    typename S::Config;
    typename S::Rhs;
    typename S::DenseData;

    { s.step( f, x, t, h, cfg ) }
        -> std::same_as< StepResult< typename S::State, S > >;

    { S::eval_dense( std::declval< typename S::DenseData >(),
                     std::declval< typename S::T >(),
                     std::declval< typename S::T >() ) }
        -> std::same_as< typename S::State >;
};

template < class S >
concept AdaptiveStepper = Stepper< S >
    && requires { { S::is_adaptive } -> std::convertible_to< bool >; }
    && S::is_adaptive;

}  // namespace tax::ode::concepts
