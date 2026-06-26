#pragma once
#include <tax/core/taylor_expansion.hpp>
#include <tax/la/types.hpp>
#include <tax/ode/propagate.hpp>
namespace tax::ads::detail
{
// The (TE, DA-state, Stepper) triplet derived from order P, dims M/D and a
// method tag — shared by propagate()'s overloads. Storage is Dense.
template < int P, class Method, class T, int M, int D >
struct AdsStateTraits
{
    using TE = tax::TaylorExpansion< T, tax::IsotropicScheme< P, M >, tax::storage::Dense >;
    using DAState = Eigen::Matrix< TE, D, 1 >;
    using Stepper = tax::ode::StepperType< Method, DAState >;
};
}  // namespace tax::ads::detail
