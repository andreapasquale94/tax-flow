// include/tax/ode/steppers/feagin14.hpp
//
// Feagin 14(12) RK stepper. 35 stages, propagates at order 14, uses an
// order-12 embedded estimator for adaptive step-size control. Feagin's
// `(k_2 - k_{n-1})` error indicator can underflow to exactly zero on
// benign integrands at small h; the shared stepper floors the
// controller error at machine eps * tol in that case.
// Implementation: detail::EmbeddedRKStepper over the Feagin14 tableau.

#pragma once

#include <tax/ode/controllers.hpp>
#include <tax/ode/detail/embedded_rk_stepper.hpp>
#include <tax/ode/detail/feagin_tableaus.hpp>

namespace tax::ode
{

template < class StateT,
           class Controller = controllers::PI< double > >
using Feagin14Stepper =
    detail::EmbeddedRKStepper< detail::Feagin14Tab, StateT, Controller >;

}  // namespace tax::ode
