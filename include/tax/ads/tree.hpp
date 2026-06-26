// include/tax/ads/tree.hpp
//
// AdsTree<Payload, M, T> — arena-backed leaf-only binary tree used by
// the ADS driver. Splits append two children and retire the parent;
// merges restore the parent and retire the pair. No internal "split"
// nodes: every record in the arena is a Leaf, and the tree shape is
// reconstructed from parentIdx / siblingIdx links.
//
// Work queue: std::deque<int> driven in BFS order via popFront. The
// driver pops a leaf, integrates, and either splits or finalize-s it.
//
// Point lookup: locate(pt) does a linear scan over active+done (skipping
// retired). At ADS-typical sizes (10..1000 leaves) this is faster than
// a tree walk in practice and avoids the variant-node bookkeeping.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <deque>
#include <optional>
#include <span>
#include <tax/ads/box.hpp>
#include <tax/ads/leaf.hpp>
#include <tax/la/types.hpp>
#include <utility>
#include <vector>

namespace tax::ads
{

template < class Payload, int M, class T = double >
class AdsTree
{
   public:
    using LeafT = Leaf< Payload, M, T >;
    using BoxT = Box< T, M >;

    [[nodiscard]] int init( BoxT box, Payload payload, T tEntry = T{ 0 } )
    {
        LeafT l{};
        l.box = std::move( box );
        l.payload = std::move( payload );
        l.tEntry = tEntry;
        const int idx = static_cast< int >( leaves_.size() );
        leaves_.push_back( std::move( l ) );
        listPos_.push_back( -1 );
        roots_.push_back( idx );
        pushActive( idx );
        workQueue_.push_back( idx );
        return idx;
    }

    [[nodiscard]] bool empty() const noexcept { return workQueue_.empty(); }

    [[nodiscard]] int front() const noexcept
    {
        assert( !workQueue_.empty() );
        return workQueue_.front();
    }

    [[nodiscard]] int popFront() noexcept
    {
        assert( !workQueue_.empty() );
        const int idx = workQueue_.front();
        workQueue_.pop_front();
        return idx;
    }

    [[nodiscard]] std::pair< int, int > split( int idx, int dim, Payload leftPayload,
                                               Payload rightPayload, T tEntry )
    {
        assert( idx >= 0 && idx < static_cast< int >( leaves_.size() ) );
        assert( !leaves_[idx].done && !leaves_[idx].retired );

        // Retire parent and remove from active list.
        leaves_[idx].retired = true;
        removeFromActive( idx );

        // Halve the parent's box.
        auto pr = leaves_[idx].box.split( dim );
        auto& boxL = pr.first;
        auto& boxR = pr.second;

        const int parentDepth = leaves_[idx].depth;

        LeafT L{};
        L.box = std::move( boxL );
        L.payload = std::move( leftPayload );
        L.depth = parentDepth + 1;
        L.parentIdx = idx;
        L.splitDim = dim;
        L.tEntry = tEntry;

        LeafT R{};
        R.box = std::move( boxR );
        R.payload = std::move( rightPayload );
        R.depth = parentDepth + 1;
        R.parentIdx = idx;
        R.splitDim = dim;
        R.tEntry = tEntry;

        const int lIdx = static_cast< int >( leaves_.size() );
        leaves_.push_back( std::move( L ) );
        listPos_.push_back( -1 );
        const int rIdx = static_cast< int >( leaves_.size() );
        leaves_.push_back( std::move( R ) );
        listPos_.push_back( -1 );

        // Wire sibling links.
        leaves_[lIdx].siblingIdx = rIdx;
        leaves_[rIdx].siblingIdx = lIdx;

        pushActive( lIdx );
        pushActive( rIdx );
        workQueue_.push_back( lIdx );
        workQueue_.push_back( rIdx );

        return { lIdx, rIdx };
    }

    void finalize( int idx )
    {
        assert( idx >= 0 && idx < static_cast< int >( leaves_.size() ) );
        assert( !leaves_[idx].done && !leaves_[idx].retired );
        leaves_[idx].done = true;
        removeFromActive( idx );
        pushDone( idx );
    }

    void merge( int leftIdx, int rightIdx, Payload mergedPayload )
    {
        assert( leftIdx >= 0 && leftIdx < static_cast< int >( leaves_.size() ) );
        assert( rightIdx >= 0 && rightIdx < static_cast< int >( leaves_.size() ) );
        assert( leaves_[leftIdx].parentIdx == leaves_[rightIdx].parentIdx );
        assert( leaves_[leftIdx].siblingIdx == rightIdx );
        assert( leaves_[rightIdx].siblingIdx == leftIdx );
        const int parent = leaves_[leftIdx].parentIdx;
        assert( parent >= 0 );
        assert( leaves_[parent].retired );

        // Both children become retired; the parent revives as done.
        // Preconditions above guarantee the children are in doneList_, not
        // activeList_, so only removeFromDone is needed.
        leaves_[leftIdx].done = false;
        leaves_[rightIdx].done = false;
        leaves_[leftIdx].retired = true;
        leaves_[rightIdx].retired = true;
        removeFromDone( leftIdx );
        removeFromDone( rightIdx );

        leaves_[parent].retired = false;
        leaves_[parent].done = true;
        leaves_[parent].payload = std::move( mergedPayload );
        pushDone( parent );
    }

    [[nodiscard]] const LeafT& leaf( int idx ) const noexcept
    {
        return leaves_[static_cast< std::size_t >( idx )];
    }
    [[nodiscard]] LeafT& leaf( int idx ) noexcept
    {
        return leaves_[static_cast< std::size_t >( idx )];
    }

    [[nodiscard]] std::span< const int > active() const noexcept
    {
        return { activeList_.data(), activeList_.size() };
    }
    [[nodiscard]] std::span< const int > done() const noexcept
    {
        return { doneList_.data(), doneList_.size() };
    }

    // Reorder the done-leaf index list into a canonical, deterministic
    // order: ascending box center, lexicographic over the M coordinates.
    // Disjoint boxes have distinct centers, so this is a stable total
    // order independent of insertion order — used to make parallel and
    // serial propagation agree and to make output reproducible.
    void canonicalizeDone()
    {
        std::sort( doneList_.begin(), doneList_.end(), [this]( int a, int b ) {
            const auto& ca = leaves_[static_cast< std::size_t >( a )].box.center;
            const auto& cb = leaves_[static_cast< std::size_t >( b )].box.center;
            for ( int i = 0; i < M; ++i )
            {
                if ( ca( i ) < cb( i ) ) return true;
                if ( cb( i ) < ca( i ) ) return false;
            }
            return false;
        } );
        for ( std::size_t i = 0; i < doneList_.size(); ++i )
            listPos_[static_cast< std::size_t >( doneList_[i] )] = static_cast< int >( i );
    }

    [[nodiscard]] std::span< const int > roots() const noexcept
    {
        return { roots_.data(), roots_.size() };
    }

    template < class Derived >
    [[nodiscard]] std::optional< int > locate( const Eigen::MatrixBase< Derived >& pt ) const
    {
        for ( int idx : activeList_ )
            if ( leaves_[idx].box.contains( pt ) ) return idx;
        for ( int idx : doneList_ )
            if ( leaves_[idx].box.contains( pt ) ) return idx;
        return std::nullopt;
    }

   private:
    // A leaf is in at most one of activeList_/doneList_ at a time;
    // listPos_[idx] is its position there (-1 while in neither), so
    // removal is an O(1) swap-with-back instead of a linear scan.
    void pushActive( int idx )
    {
        listPos_[static_cast< std::size_t >( idx )] = static_cast< int >( activeList_.size() );
        activeList_.push_back( idx );
    }
    void pushDone( int idx )
    {
        listPos_[static_cast< std::size_t >( idx )] = static_cast< int >( doneList_.size() );
        doneList_.push_back( idx );
    }
    void removeFrom( std::vector< int >& list, int idx ) noexcept
    {
        const int pos = listPos_[static_cast< std::size_t >( idx )];
        assert( pos >= 0 && list[static_cast< std::size_t >( pos )] == idx );
        const int last = list.back();
        list[static_cast< std::size_t >( pos )] = last;
        listPos_[static_cast< std::size_t >( last )] = pos;
        list.pop_back();
        listPos_[static_cast< std::size_t >( idx )] = -1;
    }
    void removeFromActive( int idx ) noexcept { removeFrom( activeList_, idx ); }
    void removeFromDone( int idx ) noexcept { removeFrom( doneList_, idx ); }

    std::vector< LeafT > leaves_;
    std::vector< int > activeList_;
    std::vector< int > doneList_;
    std::vector< int > listPos_;
    std::vector< int > roots_;
    std::deque< int > workQueue_;
};

}  // namespace tax::ads
