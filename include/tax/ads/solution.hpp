// include/tax/ads/solution.hpp
//
// AdsSolution<Stepper, M> — owning, time-resolved result of an ADS run.
// Wraps the AdsTree (topology + final flow maps in leaf.payload, unchanged)
// and, beside it, each arena leaf's ode::Solution (events + steps). Snapshots
// at the grid-event times are reconstructed by grouping the reserved-label
// ("ads:grid") event records across leaves; each group is the active partition
// (a tiling of the IC box) at that time. There is no arbitrary-t query: the
// stored nodes are exactly the propagator's step/event boundaries.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <tax/ads/box.hpp>
#include <tax/ads/tree.hpp>
#include <tax/ode/solution.hpp>
#include <utility>
#include <vector>

namespace tax::ads
{

// Reserved event label for synchronized snapshot times. propagate<P>'s grid
// overload tags its GridEvent with this; snapshots() groups only records that
// carry it, so a user's own (unsynchronized) events never enter a snapshot.
inline constexpr const char* kSnapshotLabel = "ads:grid";

template < class Stepper, int M >
class AdsSolution
{
   public:
    using State = typename Stepper::State;
    using T = typename Stepper::T;
    using LeafSol = tax::ode::Solution< Stepper, State >;
    using Tree = AdsTree< State, M, T >;
    using BoxT = Box< T, M >;

    // One leaf within a partition: its box, its DA flow map at the snapshot
    // time, its depth, and a deterministic id (index in canonical order).
    struct LeafView
    {
        const BoxT& box;
        const State& flowMap;
        int depth;
        int id;
    };

    // The active partition at one time — a tiling of the IC box.
    class Partition
    {
       public:
        explicit Partition( T t ) : t_( t ) {}
        [[nodiscard]] T time() const noexcept { return t_; }
        [[nodiscard]] std::size_t size() const noexcept { return leaves_.size(); }
        [[nodiscard]] auto begin() const noexcept { return leaves_.begin(); }
        [[nodiscard]] auto end() const noexcept { return leaves_.end(); }
        [[nodiscard]] const LeafView& operator[]( std::size_t i ) const { return leaves_[i]; }

       private:
        friend AdsSolution;

        void reserve( std::size_t n ) { leaves_.reserve( n ); }
        void add( const BoxT& box, const State& x, int depth, int id )
        {
            leaves_.push_back( LeafView{ box, x, depth, id } );
        }

        T t_;
        std::vector< LeafView > leaves_;
    };

    // `leafSol` MUST be indexed parallel to the tree arena: leafSol[idx] is the
    // ode::Solution for tree.leaf(idx), one entry per arena leaf (retired,
    // active, and done alike). The AdsDriver upholds this. snapshots()/final()
    // index leafSol_ by arena index, so a size mismatch is a caller error.
    AdsSolution( Tree tree, std::vector< LeafSol > leafSol, T t0, T t1 )
        : tree_( std::move( tree ) ), leafSol_( std::move( leafSol ) ), t0_( t0 ), t1_( t1 )
    {
    }

    [[nodiscard]] const Tree& tree() const noexcept { return tree_; }
    [[nodiscard]] const LeafSol& leaf( int idx ) const noexcept
    {
        return leafSol_[static_cast< std::size_t >( idx )];
    }
    [[nodiscard]] T t0() const noexcept { return t0_; }
    [[nodiscard]] T t1() const noexcept { return t1_; }

    // The accepted partition at t1: the done leaves with their final flow maps.
    [[nodiscard]] Partition final() const
    {
        std::vector< Cand > cands;
        for ( int idx : tree_.done() )
        {
            const auto& lf = tree_.leaf( idx );
            cands.push_back( Cand{ &lf.box, &lf.payload, lf.depth } );
        }
        return makePartition( t1_, std::move( cands ) );
    }

    // One partition per synchronized snapshot time, bracketed by t0 and t1.
    [[nodiscard]] std::vector< Partition > snapshots() const
    {
        // Gather all reserved-label event records across every arena leaf.
        std::vector< Rec > recs;
        // Iterate ALL arena leaves, including retired parents: a retired parent
        // holds the grid records for times before its split, where its box is the
        // active partition member. Excluding them would leave gaps before splits.
        for ( std::size_t idx = 0; idx < leafSol_.size(); ++idx )
        {
            const auto& ls = leafSol_[idx];
            const auto& lf = tree_.leaf( static_cast< int >( idx ) );
            for ( const auto& e : ls.events )
                if ( e.label == kSnapshotLabel )
                    recs.push_back( Rec{ e.t, &lf.box, &e.x, lf.depth } );
        }
        std::sort( recs.begin(), recs.end(),
                   []( const Rec& a, const Rec& b ) { return a.t < b.t; } );

        std::vector< Partition > out;
        out.push_back( rootPartition( t0_ ) );  // IC partition at t0

        std::size_t i = 0;
        while ( i < recs.size() )
        {
            const T tref = recs[i].t;
            const T tol = clusterTol( tref );
            std::vector< Cand > cands;
            std::size_t j = i;
            for ( ; j < recs.size() && recs[j].t <= tref + tol; ++j )
                cands.push_back( Cand{ recs[j].box, recs[j].x, recs[j].depth } );
            i = j;
            // Skip clusters coincident with the t0/t1 brackets (avoid doubling).
            if ( std::abs( tref - t0_ ) <= tol || std::abs( tref - t1_ ) <= tol ) continue;
            out.push_back( makePartition( tref, std::move( cands ) ) );
        }

        out.push_back( final() );  // accepted partition at t1
        return out;
    }

   private:
    struct Cand
    {
        const BoxT* box;
        const State* x;
        int depth;
    };
    struct Rec
    {
        T t;
        const BoxT* box;
        const State* x;
        int depth;
    };

    // Two recorded times belong to the same snapshot if they differ by less
    // than this. Grid landings agree to a few epsilon across leaves; assumes
    // the snapshot grid is spaced by >> 1e-9*|t|.
    [[nodiscard]] static T clusterTol( T t ) noexcept
    {
        return ( std::abs( t ) + T{ 1 } ) * T{ 1e-9 };
    }

    // Sort candidates by canonical box-center order, assign ids, build a Partition.
    [[nodiscard]] Partition makePartition( T t, std::vector< Cand > cands ) const
    {
        std::sort( cands.begin(), cands.end(), []( const Cand& a, const Cand& b ) {
            for ( int k = 0; k < M; ++k )
            {
                if ( a.box->center( k ) < b.box->center( k ) ) return true;
                if ( b.box->center( k ) < a.box->center( k ) ) return false;
            }
            return false;
        } );
        Partition p{ t };
        p.reserve( cands.size() );
        for ( std::size_t id = 0; id < cands.size(); ++id )
            p.add( *cands[id].box, *cands[id].x, cands[id].depth, static_cast< int >( id ) );
        return p;
    }

    // The t0 partition: the root(s) with their initial DA states (x.front()).
    [[nodiscard]] Partition rootPartition( T t ) const
    {
        std::vector< Cand > cands;
        for ( int idx : tree_.roots() )
        {
            const auto& lf = tree_.leaf( idx );
            const auto& ls = leafSol_[static_cast< std::size_t >( idx )];
            cands.push_back( Cand{ &lf.box, &ls.x.front(), lf.depth } );
        }
        return makePartition( t, std::move( cands ) );
    }

    Tree tree_;
    std::vector< LeafSol > leafSol_;
    T t0_;
    T t1_;
};

}  // namespace tax::ads
