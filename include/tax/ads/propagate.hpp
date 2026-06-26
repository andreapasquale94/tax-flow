// include/tax/ads/propagate.hpp
//
// Function-form propagate for ADS / LOADS. Generic over the IC domain type
// (Box / Zonotope / PolynomialZonotope): domain_traits recovers the scalar T
// and factor dimension M, the state dimension D is deduced from the IC center
// vector, and detail::AdsStateTraits assembles the (TE, DA-state, Stepper)
// triplet from the explicit truncation order P. An AdsDriver runs underneath.
//
//   // Axis-aligned Box IC:
//   auto sol = tax::ads::propagate<6>(
//       tax::ode::methods::Verner89{},
//       tax::ads::TruncationCriterion{1e-4, 8},
//       rhs, ic_box, ic_center, t0, t1, cfg);
//   const auto& tree = sol.tree();
//
//   // Oriented Zonotope IC (same call shape — only the domain type changes):
//   tax::ads::Zonotope<double, 2> z = ...;
//   auto zsol = tax::ads::propagate<6>(
//       tax::ode::methods::Verner89{},
//       tax::ads::TruncationCriterion{1e-4, 8},
//       rhs, z, ic_center, t0, t1, cfg);
//
// P is required (the DA truncation order). The method tag selects the
// stepper. The criterion drives splits. cfg is optional. num_threads
// (trailing, default 1) opts into the parallel ADS driver; >1 integrates
// independent leaves concurrently. User events are not forwarded by this
// convenience wrapper — instantiate AdsDriver directly if you need them.

#pragma once

#include <memory>
#include <string>
#include <tax/ads/detail/state_traits.hpp>
#include <tax/ads/domains/domain.hpp>
#include <tax/ads/driver.hpp>
#include <tax/ads/solution.hpp>
#include <tax/ode/events/grid_event.hpp>
#include <utility>
#include <vector>

namespace tax::ads
{

// (1) No snapshot grid. Generic over the IC domain (Box / Zonotope /
//     PolynomialZonotope). AdsSolution still holds each leaf's Solution;
//     snapshots() brackets to {t0, t1}.
template < int P, class Method, class Criterion, class F, class DomainArg, int D >
[[nodiscard]] auto propagate( Method, Criterion crit, F&& rhs, const DomainArg& ic_domain,
                              const Eigen::Matrix< domain_scalar_t< DomainArg >, D, 1 >& ic_center,
                              const domain_scalar_t< DomainArg >& t0,
                              const domain_scalar_t< DomainArg >& t1,
                              tax::ode::IntegratorConfig< domain_scalar_t< DomainArg > > cfg = {},
                              int num_threads = 1 )
{
    using T = domain_scalar_t< DomainArg >;
    constexpr int M = domain_dim_v< DomainArg >;
    using Tr = detail::AdsStateTraits< P, Method, T, M, D >;
    AdsDriver< typename Tr::Stepper, Criterion, DomainArg > driver{
        std::move( crit ), std::move( cfg ), {}, num_threads };
    return driver.run( std::forward< F >( rhs ), ic_domain, ic_center, t0, t1 );
}

// (2) With snapshot grid: a GridEvent(kSnapshotLabel) — passed through the
//     driver's generic extras list — lands a step on every grid time so each
//     leaf records the partition there. AdsSolution::snapshots() reconstructs
//     them.
template < int P, class Method, class Criterion, class F, class DomainArg, int D >
[[nodiscard]] auto propagate( Method, Criterion crit, F&& rhs, const DomainArg& ic_domain,
                              const Eigen::Matrix< domain_scalar_t< DomainArg >, D, 1 >& ic_center,
                              const domain_scalar_t< DomainArg >& t0,
                              const domain_scalar_t< DomainArg >& t1,
                              std::vector< domain_scalar_t< DomainArg > > grid_times,
                              tax::ode::IntegratorConfig< domain_scalar_t< DomainArg > > cfg = {},
                              int num_threads = 1 )
{
    using T = domain_scalar_t< DomainArg >;
    constexpr int M = domain_dim_v< DomainArg >;
    using Tr = detail::AdsStateTraits< P, Method, T, M, D >;
    using DAState = typename Tr::DAState;
    using Driver = AdsDriver< typename Tr::Stepper, Criterion, DomainArg >;

    typename Driver::ExtraEvt extras;
    extras.push_back( std::make_shared< tax::ode::GridEvent< DAState, T > >(
        std::move( grid_times ), std::string{ kSnapshotLabel } ) );

    Driver driver{ std::move( crit ), std::move( cfg ), std::move( extras ), num_threads };
    return driver.run( std::forward< F >( rhs ), ic_domain, ic_center, t0, t1 );
}

}  // namespace tax::ads
