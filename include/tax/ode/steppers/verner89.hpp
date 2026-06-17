// include/tax/ode/steppers/verner89.hpp
//
// Verner 9(8) RK stepper. 16 stages, propagates at order 9, uses an
// order-8 embedded estimator for adaptive step-size control.
// Implementation: detail::EmbeddedRKStepper over the Verner89 tableau.

#pragma once

#include <tax/ode/controllers.hpp>
#include <tax/ode/detail/embedded_rk_stepper.hpp>
#include <tax/ode/detail/verner_tableaus.hpp>

namespace tax::ode
{

template < class StateT,
           class Controller = controllers::PI< double > >
using Verner89Stepper =
    detail::EmbeddedRKStepper< detail::Verner89Tab, StateT, Controller >;

}  // namespace tax::ode
