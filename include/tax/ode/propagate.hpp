// include/tax/ode/propagate.hpp
//
// Function-form propagate. Takes a method tag, an RHS, an initial
// state, a time interval, an optional config, and optional events;
// returns the resulting Solution.
//
//   auto sol = tax::ode::propagate(methods::Verner89{},
//                                  rhs, x0, t0, t1, cfg);
//
// The returned Solution holds all accepted step boundaries
// (cfg.save_steps == true, the default) or only the initial and final
// state (cfg.save_steps == false). Events are always recorded.

#pragma once

#include <tax/ode/event.hpp>
#include <tax/ode/integrator.hpp>
#include <tax/ode/steppers/feagin12.hpp>
#include <tax/ode/steppers/feagin14.hpp>
#include <tax/ode/steppers/fehlberg78.hpp>
#include <tax/ode/steppers/taylor.hpp>
#include <tax/ode/steppers/verner78.hpp>
#include <tax/ode/steppers/verner89.hpp>
#include <type_traits>
#include <utility>
#include <vector>

namespace tax::ode
{

namespace methods
{

template < int N >
struct Taylor
{
};

struct Verner78
{
};

struct Verner89
{
};

struct Fehlberg78
{
};

struct Feagin12
{
};

struct Feagin14
{
};

}  // namespace methods

namespace detail
{

// Map a method tag + State to the corresponding Stepper type. Uses the
// stepper's default Controller.
template < class Method, class State >
struct StepperFor;

template < int N, class State >
struct StepperFor< methods::Taylor< N >, State >
{
    using type = TaylorStepper< N, State >;
};

template < class State >
struct StepperFor< methods::Verner78, State >
{
    using type = Verner78Stepper< State >;
};

template < class State >
struct StepperFor< methods::Verner89, State >
{
    using type = Verner89Stepper< State >;
};

template < class State >
struct StepperFor< methods::Fehlberg78, State >
{
    using type = Fehlberg78Stepper< State >;
};

template < class State >
struct StepperFor< methods::Feagin12, State >
{
    using type = Feagin12Stepper< State >;
};

template < class State >
struct StepperFor< methods::Feagin14, State >
{
    using type = Feagin14Stepper< State >;
};

template < class Method, class State >
using StepperT = typename StepperFor< Method, State >::type;

}  // namespace detail

// Propagate an ODE. Returns a Solution holding accepted step boundaries
// (save_steps=true by default) plus any recorded events.
template < class Method, class F, class State, class T >
[[nodiscard]] auto propagate(
    Method, F&& rhs, const State& x0, const T& t0, const T& t1, IntegratorConfig< T > cfg = {},
    std::vector< Event< detail::StepperT< Method, State > > > events = {} )
{
    using Stepper = detail::StepperT< Method, State >;
    Integrator< Stepper, std::decay_t< F > > integ{ std::forward< F >( rhs ), std::move( cfg ),
                                                    std::move( events ) };
    return integ.integrate( x0, t0, t1 );
}

}  // namespace tax::ode
