// include/tax/ads/domains/domain.hpp
//
// Domain concepts for the ADS tree. A "domain" is the geometric primitive a
// leaf owns: the subdomain of initial conditions its payload is valid for. All
// domains live in normalized factor coordinates ξ ∈ [-1,1]^M; the DA payload
// math is domain-agnostic. Two tiers:
//   * Domain          — every primitive: split / denormalize / center / create.
//   * LocatableDomain — linear primitives that also support EXACT point
//                       location (contains) and sibling ordering (splitOrdinate).
//                       PolynomialZonotope models Domain but NOT LocatableDomain.
#pragma once
#include <concepts>
#include <tax/la/types.hpp>
#include <utility>

namespace tax::ads
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
concept Domain = requires( const D d, int dim ) {
    typename domain_traits< D >::scalar;
    { domain_traits< D >::dim } -> std::convertible_to< int >;
    // center(i): per-axis anchor used for canonical leaf ordering.
    { d.center( 0 ) };
    // split(dim): bisect along factor `dim` into two child domains.
    { d.split( dim ) } -> std::same_as< std::pair< D, D > >;
};

template < class D >
concept LocatableDomain = Domain< D > && requires( const D d, int dim ) {
    { d.splitOrdinate( dim ) } -> std::convertible_to< domain_scalar_t< D > >;
};
}  // namespace tax::ads
