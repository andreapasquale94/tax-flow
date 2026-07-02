// include/tax/domain/domain.hpp
//
// Domain concepts for set-valued initial conditions. A "domain" is a geometric
// primitive describing a set as the image of the normalized factor cube
// ξ ∈ [-1,1]^M; consumers (e.g. the ADS tree in tax::ads) stay domain-agnostic
// because every payload substitution acts in ξ. Two tiers:
//   * Domain          — every primitive: center / split / denormalize.
//   * LocatableDomain — primitives with an EXACT inverse of denormalize:
//                       localize / contains / splitOrdinate. Required for
//                       exact point location (AdsTree::locate/locateFactors)
//                       and for merge's sibling ordering.
//                       PolynomialZonotope models Domain but NOT
//                       LocatableDomain (a polynomial image has no closed-form
//                       inverse; its contains() is only a conservative hull
//                       test and is deliberately not part of either tier).
#pragma once
#include <concepts>
#include <tax/la/types.hpp>
#include <utility>

namespace tax::domain
{
// Maps a domain type to its scalar (T) and ambient/factor dimension (M).
// Specialized by each primitive header (box/zonotope/polynomial_zonotope).
template < class D >
struct domain_traits;

template < class D >
using domain_scalar_t = typename domain_traits< D >::scalar;

template < class D >
inline constexpr int domain_dim_v = domain_traits< D >::dim;

template < class D >
using domain_vec_t = tax::la::VecNT< domain_traits< D >::dim, typename domain_traits< D >::scalar >;

template < class D >
concept Domain = requires( const D d, int dim, const domain_vec_t< D >& xi ) {
    typename domain_traits< D >::scalar;
    { domain_traits< D >::dim } -> std::convertible_to< int >;
    // center(i): per-axis anchor used for canonical leaf ordering.
    { d.center( 0 ) } -> std::convertible_to< domain_scalar_t< D > >;
    // split(dim): bisect along factor `dim` into two child domains.
    { d.split( dim ) } -> std::same_as< std::pair< D, D > >;
    // denormalize(ξ): map ξ ∈ [-1,1]^M to physical coordinates.
    { d.denormalize( xi ) } -> std::convertible_to< domain_vec_t< D > >;
};

template < class D >
concept LocatableDomain =
    Domain< D > && requires( const D d, int dim, const domain_vec_t< D >& pt ) {
        // localize(pt): exact inverse of denormalize (factor coordinates).
        { d.localize( pt ) } -> std::convertible_to< domain_vec_t< D > >;
        // contains(pt): exact membership.
        { d.contains( pt ) } -> std::convertible_to< bool >;
        // splitOrdinate(dim): scalar sibling ordering along the split factor.
        { d.splitOrdinate( dim ) } -> std::convertible_to< domain_scalar_t< D > >;
    };
}  // namespace tax::domain
