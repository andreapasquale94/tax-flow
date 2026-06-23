// include/tax/ode/steppers/dormand_prince45.hpp
//
// Dormand–Prince 5(4) RK stepper — the classical "RK45" (MATLAB `ode45`,
// SciPy `RK45`). 7 stages, propagates at order 5, uses an order-4
// embedded estimator for adaptive step-size control. A low-order,
// inexpensive default for non-stiff problems at modest tolerances.
// Implementation: detail::EmbeddedRKStepper over the DormandPrince45 tableau.

#pragma once

#include <tax/ode/controllers.hpp>
#include <tax/ode/detail/dormand_prince_tableaus.hpp>
#include <tax/ode/detail/embedded_rk_stepper.hpp>

namespace tax::ode
{

template < class StateT, class Controller = controllers::PI< double > >
using DormandPrince45Stepper =
    detail::EmbeddedRKStepper< detail::DormandPrince45Tab, StateT, Controller >;

}  // namespace tax::ode
