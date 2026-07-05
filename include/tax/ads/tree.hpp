// include/tax/ads/tree.hpp
//
// AdsTree<Payload, Domain> — arena-backed leaf-only binary tree used by
// the ADS driver. Splits append two children and retire the parent;
// merges restore the parent and retire the pair. No internal "split"
// nodes: every record in the arena is a Leaf, and the tree shape is
// reconstructed from parentIdx / siblingIdx links. The scalar type and
// factor dimension are recovered from the Domain via domain_traits.
//
// Work queue: std::deque<int> driven in BFS order via popFront. The
// driver pops a leaf, integrates, and either splits or finalize-s it.
//
// Point lookup: locate/locateFactors do a linear scan over active+done
// (skipping retired). At ADS-typical sizes (10..1000 leaves) this is
// faster than a tree walk in practice and avoids the variant-node
// bookkeeping. Both require a LocatableDomain — a PolynomialZonotope
// tree refuses point location at compile time (its contains() is only a
// conservative hull test).

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <deque>
#include <optional>
#include <span>
#include <tax/ads/leaf.hpp>
#include <tax/domain/domain.hpp>
#include <tax/la/types.hpp>
#include <utility>
#include <vector>

namespace tax::ads
{

template < class Payload, tax::domain::Domain Domain >
class AdsTree
{
   public:
    using T = tax::domain::domain_scalar_t< Domain >;
    static constexpr int M = tax::domain::domain_dim_v< Domain >;
    using LeafT = Leaf< Payload, Domain >;
    using DomainT = Domain;

    // A located point: the owning leaf and its exact factor coordinates.
    struct Location
    {
        int idx;
        tax::la::VecNT< M, T > xi;
    };

    [[nodiscard]] int init( Domain domain, Payload payload, T tEntry = T{ 0 } )
    {
        LeafT l{};
        l.domain = std::move( domain );
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

    // Discard any pending work-queue entries. Used by drivers (e.g. the refine
    // driver) that schedule from their own queue instead of this one, so the
    // returned tree reports empty()==true like the classic driver's.
    void clearWorkQueue() noexcept { workQueue_.clear(); }

    [[nodiscard]] std::pair< int, int > split( int idx, int dim, Payload leftPayload,
                                               Payload rightPayload, T tEntry )
    {
        assert( idx >= 0 && idx < static_cast< int >( leaves_.size() ) );
        assert( !leaves_[idx].done && !leaves_[idx].retired );

        // Retire parent and remove from active list.
        leaves_[idx].retired = true;
        removeFromActive( idx );

        // Halve the parent's domain.
        auto pr = leaves_[idx].domain.split( dim );
        auto& domL = pr.first;
        auto& domR = pr.second;

        const int parentDepth = leaves_[idx].depth;

        LeafT L{};
        L.domain = std::move( domL );
        L.payload = std::move( leftPayload );
        L.depth = parentDepth + 1;
        L.parentIdx = idx;
        L.splitDim = dim;
        L.tEntry = tEntry;

        LeafT R{};
        R.domain = std::move( domR );
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

    // Split a leaf into two already-DONE children — no re-integration. Used when
    // the split criterion fires exactly at the final time t1: the parent's flow
    // map is complete, so tax::ads::split of it IS the pair of final child flow
    // maps. The children are wired like split()'s (parent/sibling links) but go
    // straight into doneList_, never onto the active list or work queue.
    [[nodiscard]] std::pair< int, int > splitDone( int idx, int dim, Payload leftPayload,
                                                   Payload rightPayload, T tEntry )
    {
        assert( idx >= 0 && idx < static_cast< int >( leaves_.size() ) );
        assert( !leaves_[idx].done && !leaves_[idx].retired );

        leaves_[idx].retired = true;
        removeFromActive( idx );

        auto pr = leaves_[idx].domain.split( dim );
        const int parentDepth = leaves_[idx].depth;

        LeafT L{};
        L.domain = std::move( pr.first );
        L.payload = std::move( leftPayload );
        L.depth = parentDepth + 1;
        L.parentIdx = idx;
        L.splitDim = dim;
        L.tEntry = tEntry;
        L.done = true;

        LeafT R{};
        R.domain = std::move( pr.second );
        R.payload = std::move( rightPayload );
        R.depth = parentDepth + 1;
        R.parentIdx = idx;
        R.splitDim = dim;
        R.tEntry = tEntry;
        R.done = true;

        const int lIdx = static_cast< int >( leaves_.size() );
        leaves_.push_back( std::move( L ) );
        listPos_.push_back( -1 );
        const int rIdx = static_cast< int >( leaves_.size() );
        leaves_.push_back( std::move( R ) );
        listPos_.push_back( -1 );

        leaves_[lIdx].siblingIdx = rIdx;
        leaves_[rIdx].siblingIdx = lIdx;

        pushDone( lIdx );
        pushDone( rIdx );

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
        // Children MUST be done (in doneList_): removeFromDone below indexes via
        // listPos_, which for an active leaf points into activeList_ and would
        // silently corrupt both lists in a release build.
        assert( leaves_[leftIdx].done && leaves_[rightIdx].done );

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

    // Reorder the done-leaf index list into a canonical, deterministic order via
    // tax::domain::domainLess (a strict total order over disjoint domains, found
    // by ADL — center-lexicographic by default, coefficient-lexicographic for
    // PolynomialZonotope). Independent of insertion order, so parallel and serial
    // propagation agree and output is reproducible.
    void canonicalizeDone()
    {
        std::sort( doneList_.begin(), doneList_.end(), [this]( int a, int b ) {
            return domainLess( leaves_[static_cast< std::size_t >( a )].domain,
                               leaves_[static_cast< std::size_t >( b )].domain );
        } );
        for ( std::size_t i = 0; i < doneList_.size(); ++i )
            listPos_[static_cast< std::size_t >( doneList_[i] )] = static_cast< int >( i );
    }

    [[nodiscard]] std::span< const int > roots() const noexcept
    {
        return { roots_.data(), roots_.size() };
    }

    // First leaf (active first, then done) whose domain contains pt.
    template < class Derived >
    [[nodiscard]] std::optional< int > locate( const Eigen::MatrixBase< Derived >& pt ) const
        requires tax::domain::LocatableDomain< Domain >
    {
        for ( int idx : activeList_ )
            if ( leaves_[idx].domain.contains( pt ) ) return idx;
        for ( int idx : doneList_ )
            if ( leaves_[idx].domain.contains( pt ) ) return idx;
        return std::nullopt;
    }

    // Exact point location with factor coordinates: among the non-retired
    // leaves whose domain reconstructs pt, the one with the smallest ‖ξ‖∞
    // (robust on shared split boundaries, where siblings both report ξ = ±1).
    // nullopt if no leaf claims the point.
    template < class Derived >
    [[nodiscard]] std::optional< Location > locateFactors( const Eigen::MatrixBase< Derived >& pt,
                                                           T tol = T{ 1e-9 } ) const
        requires tax::domain::LocatableDomain< Domain >
    {
        std::optional< Location > best;
        T bestInf = T{ 1 } + tol;
        auto consider = [&]( int idx ) {
            const auto& d = leaves_[static_cast< std::size_t >( idx )].domain;
            const tax::la::VecNT< M, T > xi = d.localize( pt );
            const T inf = xi.template lpNorm< Eigen::Infinity >();
            if ( inf > bestInf ) return;
            // Reject points off the domain (e.g. off the span of a
            // rank-deficient generator matrix): denormalize must round-trip.
            const tax::la::VecNT< M, T > rec = d.denormalize( xi );
            const T scale = T{ 1 } + pt.template lpNorm< Eigen::Infinity >();
            if ( ( rec - pt ).template lpNorm< Eigen::Infinity >() > tol * scale ) return;
            bestInf = inf;
            best = Location{ idx, xi };
        };
        for ( int idx : activeList_ ) consider( idx );
        for ( int idx : doneList_ ) consider( idx );
        return best;
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
