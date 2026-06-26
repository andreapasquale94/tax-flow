// include/tax/ads/propagate.hpp
//
// Function-form propagate for ADS / LOADS. Constructs the DA-state
// type from the explicit truncation order P, the box dimension M
// (deduced from the Box) and the state dimension D (deduced from the
// IC center vector), then runs an AdsDriver under the hood.
//
//   auto sol = tax::ads::propagate<6>(
//       tax::ode::methods::Verner89{},
//       tax::ads::TruncationCriterion{1e-4, 8},
//       rhs, ic_box, ic_center, t0, t1, cfg);
//   const auto& tree = sol.tree();
//
// P is required (the DA truncation order). The method tag selects the
// stepper. The criterion drives splits. cfg is optional. num_threads
// (trailing, default 1) opts into the parallel ADS driver; >1 integrates
// independent boxes concurrently. User events are not forwarded by this
// convenience wrapper — instantiate AdsDriver directly if you need them.

#pragma once

#include <memory>
#include <string>
#include <tax/ads/domains/box.hpp>
#include <tax/ads/driver.hpp>
#include <tax/ads/solution.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/la/types.hpp>
#include <tax/ode/events/grid_event.hpp>
#include <tax/ode/propagate.hpp>
#include <utility>
#include <vector>

namespace tax::ads
{

// (1) No snapshot grid. AdsSolution still holds each leaf's Solution;
//     snapshots() brackets to {t0, t1}.
template < int P, class Method, class Criterion, class F, class T, int M, int D >
[[nodiscard]] auto propagate( Method, Criterion crit, F&& rhs, const Box< T, M >& ic_box,
                              const Eigen::Matrix< T, D, 1 >& ic_center, const T& t0, const T& t1,
                              tax::ode::IntegratorConfig< T > cfg = {}, int num_threads = 1 )
{
    using TE = tax::TaylorExpansion< T, tax::IsotropicScheme< P, M >, tax::storage::Dense >;
    using DAState = Eigen::Matrix< TE, D, 1 >;
    using Stepper = tax::ode::StepperType< Method, DAState >;

    AdsDriver< Stepper, Criterion > driver{ std::move( crit ), std::move( cfg ), {}, num_threads };
    return driver.run( std::forward< F >( rhs ), ic_box, ic_center, t0, t1 );
}

// (2) With snapshot grid: a GridEvent(kSnapshotLabel) — passed through the
//     driver's generic extras list — lands a step on every grid time so each
//     leaf records the partition there. snapshots() reconstructs them.
template < int P, class Method, class Criterion, class F, class T, int M, int D >
[[nodiscard]] auto propagate( Method, Criterion crit, F&& rhs, const Box< T, M >& ic_box,
                              const Eigen::Matrix< T, D, 1 >& ic_center, const T& t0, const T& t1,
                              std::vector< T > grid_times, tax::ode::IntegratorConfig< T > cfg = {},
                              int num_threads = 1 )
{
    using TE = tax::TaylorExpansion< T, tax::IsotropicScheme< P, M >, tax::storage::Dense >;
    using DAState = Eigen::Matrix< TE, D, 1 >;
    using Stepper = tax::ode::StepperType< Method, DAState >;

    typename AdsDriver< Stepper, Criterion >::ExtraEvt extras;
    extras.push_back( std::make_shared< tax::ode::GridEvent< DAState, T > >(
        std::move( grid_times ), std::string{ kSnapshotLabel } ) );

    AdsDriver< Stepper, Criterion > driver{ std::move( crit ), std::move( cfg ),
                                            std::move( extras ), num_threads };
    return driver.run( std::forward< F >( rhs ), ic_box, ic_center, t0, t1 );
}

}  // namespace tax::ads
