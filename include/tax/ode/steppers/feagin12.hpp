// include/tax/ode/steppers/feagin12.hpp
//
// Feagin 12(10) RK stepper. 25 stages, propagates at order 12, uses an
// order-10 embedded estimator for adaptive step-size control. Feagin's
// `(k_2 - k_{n-1})` error indicator can underflow to exactly zero on
// benign integrands at small h; the shared stepper floors the
// controller error at machine eps * tol in that case.
// Implementation: detail::EmbeddedRKStepper over the Feagin12 tableau.

#pragma once

#include <tax/ode/controllers.hpp>
#include <tax/ode/detail/embedded_rk_stepper.hpp>
#include <tax/ode/detail/feagin_tableaus.hpp>

namespace tax::ode
{

template < class StateT,
           class Controller = controllers::PI< double > >
using Feagin12Stepper =
    detail::EmbeddedRKStepper< detail::Feagin12Tab, StateT, Controller >;

}  // namespace tax::ode
