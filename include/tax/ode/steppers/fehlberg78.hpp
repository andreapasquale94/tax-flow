// include/tax/ode/steppers/fehlberg78.hpp
//
// Classical Fehlberg 1968 RK 7(8) pair. 13 stages, propagates at order 7,
// uses an order-8 embedded estimator for adaptive step-size control. Known
// to suffer from the "Fehlberg coincidence" (embedded estimator zero on
// certain steps); the shared stepper floors the controller error at
// machine eps * tol to keep the step size growing in that case.
// Implementation: detail::EmbeddedRKStepper over the Fehlberg78 tableau.

#pragma once

#include <tax/ode/controllers.hpp>
#include <tax/ode/detail/embedded_rk_stepper.hpp>
#include <tax/ode/detail/fehlberg_tableaus.hpp>

namespace tax::ode
{

template < class StateT, class Controller = controllers::PI< double > >
using Fehlberg78Stepper = detail::EmbeddedRKStepper< detail::Fehlberg78Tab, StateT, Controller >;

}  // namespace tax::ode
