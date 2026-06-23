// include/tax/ode/aliases.hpp
//
// Per-method Integrator type aliases (Verner78, Verner89, Fehlberg78,
// Feagin12, Feagin14, Taylor). Split out of integrator.hpp so the
// generic driver can be included without pulling in every stepper and
// tableau header; the <tax/ode.hpp> umbrella includes both.
//
// Defaulted F parameter resolves to Stepper::Rhs = std::function<…>, so
// the simple form `Verner78<State>{f, cfg}` works out of the box (one
// vtable indirection per RHS call). Users who care about that overhead
// can spell F explicitly:
//
//   Verner78<State, controllers::PI<double>,
//            decltype(my_lambda)> integ{ my_lambda, cfg };

#pragma once

#include <tax/ode/controllers.hpp>
#include <tax/ode/integrator.hpp>
#include <tax/ode/steppers/dormand_prince45.hpp>
#include <tax/ode/steppers/feagin12.hpp>
#include <tax/ode/steppers/feagin14.hpp>
#include <tax/ode/steppers/fehlberg78.hpp>
#include <tax/ode/steppers/taylor.hpp>
#include <tax/ode/steppers/verner67.hpp>
#include <tax/ode/steppers/verner78.hpp>
#include <tax/ode/steppers/verner89.hpp>

namespace tax::ode
{

template < class State, class Controller = controllers::PI< double >,
           class F = typename Verner67Stepper< State, Controller >::Rhs >
using Verner67 = Integrator< Verner67Stepper< State, Controller >, F >;

template < class State, class Controller = controllers::PI< double >,
           class F = typename Verner78Stepper< State, Controller >::Rhs >
using Verner78 = Integrator< Verner78Stepper< State, Controller >, F >;

template < class State, class Controller = controllers::PI< double >,
           class F = typename Verner89Stepper< State, Controller >::Rhs >
using Verner89 = Integrator< Verner89Stepper< State, Controller >, F >;

template < class State, class Controller = controllers::PI< double >,
           class F = typename Fehlberg78Stepper< State, Controller >::Rhs >
using Fehlberg78 = Integrator< Fehlberg78Stepper< State, Controller >, F >;

template < class State, class Controller = controllers::PI< double >,
           class F = typename DormandPrince45Stepper< State, Controller >::Rhs >
using DormandPrince45 = Integrator< DormandPrince45Stepper< State, Controller >, F >;

template < class State, class Controller = controllers::PI< double >,
           class F = typename Feagin12Stepper< State, Controller >::Rhs >
using Feagin12 = Integrator< Feagin12Stepper< State, Controller >, F >;

template < class State, class Controller = controllers::PI< double >,
           class F = typename Feagin14Stepper< State, Controller >::Rhs >
using Feagin14 = Integrator< Feagin14Stepper< State, Controller >, F >;

template < int N, class State, class Controller = controllers::JorbaZou< double >,
           class F = typename TaylorStepper< N, State, Controller >::Rhs >
using Taylor = Integrator< TaylorStepper< N, State, Controller >, F >;

}  // namespace tax::ode
