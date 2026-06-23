// include/tax/ode/steppers/verner67.hpp
//
// Verner "most efficient" 7(6) RK stepper. 10 stages, propagates at
// order 7, uses an order-6 embedded estimator for adaptive step-size
// control. A lighter-weight companion to Verner 8(7)/9(8) for moderate
// tolerances.
// Implementation: detail::EmbeddedRKStepper over the Verner67 tableau.

#pragma once

#include <tax/ode/controllers.hpp>
#include <tax/ode/detail/embedded_rk_stepper.hpp>
#include <tax/ode/detail/verner_tableaus.hpp>

namespace tax::ode
{

template < class StateT, class Controller = controllers::PI< double > >
using Verner67Stepper = detail::EmbeddedRKStepper< detail::Verner67Tab, StateT, Controller >;

}  // namespace tax::ode
