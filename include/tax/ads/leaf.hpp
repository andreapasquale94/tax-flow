// include/tax/ads/leaf.hpp
//
// Leaf<Payload, Domain> — single arena entry in AdsTree. Each leaf owns
// its subdomain (any tax::domain primitive) and a Payload, plus parent /
// sibling indices and the dim that separated it from its sibling. A
// retired leaf is the parent of an active or done sibling pair; it stays
// in the arena so the merger can revive it via AdsTree::merge.

#pragma once

#include <tax/domain/domain.hpp>

namespace tax::ads
{

template < class Payload, tax::domain::Domain Domain >
struct Leaf
{
    using T = tax::domain::domain_scalar_t< Domain >;

    Domain domain{};
    Payload payload{};
    int depth = 0;
    bool done = false;
    bool retired = false;
    int parentIdx = -1;
    int siblingIdx = -1;
    int splitDim = -1;
    T tEntry = T{ 0 };
};

}  // namespace tax::ads
