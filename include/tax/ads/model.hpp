// include/tax/ads/model.hpp
//
// ADS over Taylor-model states (requires the tax core's tax::model module).
//
// Wires tax::ode::methods::Picard into the ADS convenience layer: propagate
// with the Picard tag runs the whole Automatic Domain Splitting pipeline on
// VALIDATED Taylor-model payloads — every leaf's flow map carries a rigorous
// remainder enclosure, splits act on the polynomial parts (the remainder
// stays valid on each child's sub-domain), and AdsSolution::evaluate returns
// interval enclosures.
//
//   auto sol = tax::ads::propagate<P>(
//       tax::ode::methods::Picard{},
//       tax::ads::TruncationCriterion{1e-4, 8},
//       rhs, ic_box, ic_center, t0, t1, cfg);
//
// Not supported for Taylor-model payloads (compile-time refusals with the
// rationale at the call site): merge() (a child's remainder bound does not
// extend to the parent domain) and refine() (TE-specific).

#pragma once

#include <tax/ads/detail/state_traits.hpp>
#include <tax/domain/model.hpp>
#include <tax/model.hpp>
#include <tax/ode/steppers/taylor_model.hpp>

namespace tax::ads::detail
{

// The (model, state, stepper) triplet for the Picard method tag: the DA state
// is an Eigen vector of order-P, M-variate Taylor models (interface parity:
// TaylorModel exposes order_v / vars_v exactly like a TaylorExpansion, which
// is all AdsDriver reads from `TE`).
template < int P, class T, int M, int D >
struct AdsStateTraits< P, tax::ode::methods::Picard, T, M, D >
{
    using TE = tax::model::TaylorModel< T, P, M >;
    using DAState = Eigen::Matrix< TE, D, 1 >;
    using Stepper = tax::ode::TaylorModelStepper< DAState >;
};

}  // namespace tax::ads::detail
