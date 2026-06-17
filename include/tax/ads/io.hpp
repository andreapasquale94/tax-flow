// include/tax/ads/io.hpp
//
// CSV helpers for AdsTree. Opt-in: not pulled in by the umbrella
// <tax/ads.hpp>.
//
// Each `done` leaf in the tree has a known [tEntry, t_final] span;
// each retired ancestor has [tEntry, child.tEntry]. The helpers below
// walk this lifecycle and write either the full per-leaf table or a
// time-series of the number of boxes alive at each snapshot time.

#pragma once

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <span>
#include <unordered_set>
#include <vector>

#include <tax/ads/tree.hpp>
#include <tax/la/types.hpp>

namespace tax::ads
{

namespace detail
{

template < int M, class T >
struct LeafSpan
{
    int                    idx;
    int                    parent;
    int                    sibling;
    int                    depth;
    bool                   retired;
    bool                   done;
    T                      tEntry;
    T                      tExit;
    tax::la::VecNT< M, T > center;
    tax::la::VecNT< M, T > halfWidth;
};

template < class Payload, int M, class T >
[[nodiscard]] std::vector< LeafSpan< M, T > > collectLeafSpans(
    const AdsTree< Payload, M, T >& tree, T t_final )
{
    std::vector< LeafSpan< M, T > > out;
    std::unordered_set< int >       seen;
    auto                            push = [ & ]( int idx, T tEntry, T tExit )
    {
        const auto&           l = tree.leaf( idx );
        LeafSpan< M, T > s{};
        s.idx       = idx;
        s.parent    = l.parentIdx;
        s.sibling   = l.siblingIdx;
        s.depth     = l.depth;
        s.retired   = l.retired;
        s.done      = l.done;
        s.tEntry    = tEntry;
        s.tExit     = tExit;
        s.center    = l.box.center;
        s.halfWidth = l.box.halfWidth;
        out.push_back( s );
        seen.insert( idx );
    };
    for ( int idx : tree.done() ) push( idx, tree.leaf( idx ).tEntry, t_final );
    for ( int idx : tree.done() )
    {
        int cur       = tree.leaf( idx ).parentIdx;
        T   exit_time = tree.leaf( idx ).tEntry;
        while ( cur >= 0 && !seen.count( cur ) )
        {
            const auto& p = tree.leaf( cur );
            push( cur, p.tEntry, exit_time );
            exit_time = p.tEntry;
            cur       = p.parentIdx;
        }
    }
    return out;
}

}  // namespace detail

template < class Payload, int M, class T >
void writeTreeCsv( const AdsTree< Payload, M, T >& tree, T t_final,
                   const std::filesystem::path& path )
{
    std::ofstream out( path );
    out << "idx,parent,sibling,depth,retired,done,t_entry,t_exit";
    for ( int j = 0; j < M; ++j ) out << ",cx" << j;
    for ( int j = 0; j < M; ++j ) out << ",hw" << j;
    out << '\n';
    const auto spans = detail::collectLeafSpans( tree, t_final );
    for ( const auto& s : spans )
    {
        out << s.idx << ',' << s.parent << ',' << s.sibling << ',' << s.depth << ','
            << ( s.retired ? 1 : 0 ) << ',' << ( s.done ? 1 : 0 ) << ',' << s.tEntry << ','
            << s.tExit;
        for ( int j = 0; j < M; ++j ) out << ',' << s.center( j );
        for ( int j = 0; j < M; ++j ) out << ',' << s.halfWidth( j );
        out << '\n';
    }
}

template < class Payload, int M, class T >
void writeBoxCountCsv( const AdsTree< Payload, M, T >& tree, T t_final,
                       std::span< const double > times, const std::filesystem::path& path )
{
    const auto    spans = detail::collectLeafSpans( tree, t_final );
    std::ofstream out( path );
    out << "t,n_alive\n";
    for ( double t : times )
    {
        int n = 0;
        for ( const auto& s : spans )
            if ( s.tEntry <= t && t <= s.tExit ) ++n;
        out << t << ',' << n << '\n';
    }
}

}  // namespace tax::ads
