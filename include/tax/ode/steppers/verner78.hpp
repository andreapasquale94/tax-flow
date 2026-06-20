// include/tax/ode/steppers/verner78.hpp
//
// Verner 8(7) RK stepper. 13 stages, propagates at order 8, uses an
// order-7 embedded estimator for adaptive step-size control.
// Implementation: detail::EmbeddedRKStepper over the Verner78 tableau.

#pragma once

#include <tax/ode/controllers.hpp>
#include <tax/ode/detail/embedded_rk_stepper.hpp>
#include <tax/ode/detail/verner_tableaus.hpp>

namespace tax::ode
{

template < class StateT, class Controller = controllers::PI< double > >
using Verner78Stepper = detail::EmbeddedRKStepper< detail::Verner78Tab, StateT, Controller >;

}  // namespace tax::ode
