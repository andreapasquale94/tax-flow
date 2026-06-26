// include/tax/ads/leaf.hpp
//
// Leaf<Payload, M, T> — single arena entry in AdsTree. Each leaf owns
// its Box subdomain and a Payload, plus parent / sibling indices and
// the dim/value that separated it from its sibling. A retired leaf is
// the parent of an active or done sibling pair; it stays in the arena
// so the merger can revive it via AdsTree::merge.

#pragma once

#include <tax/ads/domains/box.hpp>

namespace tax::ads
{

template < class Payload, int M, class T = double >
struct Leaf
{
    Box< T, M > box{};
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
