// include/tax/ads/solution.hpp
//
// AdsSolution<Stepper, Domain> — owning, time-resolved result of an ADS run.
// Wraps the AdsTree (topology + final flow maps in leaf.payload, unchanged)
// and, beside it, each arena leaf's ode::Solution (events + steps). Snapshots
// at the grid-event times are reconstructed by grouping the reserved-label
// ("ads:grid") event records across leaves; each group is the active partition
// (a tiling of the IC domain) at that time. There is no arbitrary-t query: the
// stored nodes are exactly the propagator's step/event boundaries.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <tax/ads/tree.hpp>
#include <tax/domain/domain.hpp>
#include <tax/la/types.hpp>
#include <tax/ode/solution.hpp>
#include <utility>
#include <vector>

namespace tax::ads
{

// Reserved event label for synchronized snapshot times. propagate<P>'s grid
// overload tags its GridEvent with this; snapshots() groups only records that
// carry it, so a user's own (unsynchronized) events never enter a snapshot.
inline constexpr const char* kSnapshotLabel = "ads:grid";

template < class Stepper, tax::domain::Domain Domain >
class AdsSolution
{
   public:
    using State = typename Stepper::State;
    using T = typename Stepper::T;
    static constexpr int M = tax::domain::domain_dim_v< Domain >;
    using LeafSol = tax::ode::Solution< Stepper, State >;
    using Tree = AdsTree< State, Domain >;
    using DomainT = Domain;

    // One leaf within a partition: its domain, its DA flow map at the snapshot
    // time, its depth, and a deterministic id (index in canonical order).
    struct LeafView
    {
        const Domain& domain;
        const State& flowMap;
        int depth;
        int id;
    };

    // The active partition at one time — a tiling of the IC domain.
    class Partition
    {
       public:
        explicit Partition( T t ) : t_( t ) {}
        [[nodiscard]] T time() const noexcept { return t_; }
        [[nodiscard]] std::size_t size() const noexcept { return leaves_.size(); }
        [[nodiscard]] auto begin() const noexcept { return leaves_.begin(); }
        [[nodiscard]] auto end() const noexcept { return leaves_.end(); }
        [[nodiscard]] const LeafView& operator[]( std::size_t i ) const { return leaves_[i]; }

        // Evaluate the partition's piecewise-polynomial map at a physical IC
        // point: exact factor location (smallest ‖ξ‖∞ among claiming leaves),
        // then the owning leaf's flow map at ξ. nullopt if no leaf claims pt.
        template < class Derived >
        [[nodiscard]] std::optional< Eigen::Matrix< T, State::RowsAtCompileTime, 1 > > evaluate(
            const Eigen::MatrixBase< Derived >& pt, T tol = T{ 1e-9 } ) const
            requires tax::domain::LocatableDomain< Domain >
        {
            const LeafView* best = nullptr;
            tax::la::VecNT< M, T > bestXi;
            T bestInf = T{ 1 } + tol;
            for ( const LeafView& lv : leaves_ )
            {
                const tax::la::VecNT< M, T > xi = lv.domain.localize( pt );
                const T inf = xi.template lpNorm< Eigen::Infinity >();
                if ( inf > bestInf ) continue;
                const tax::la::VecNT< M, T > rec = lv.domain.denormalize( xi );
                const T scale = T{ 1 } + pt.template lpNorm< Eigen::Infinity >();
                if ( ( rec - pt ).template lpNorm< Eigen::Infinity >() > tol * scale ) continue;
                bestInf = inf;
                bestXi = xi;
                best = &lv;
            }
            if ( best == nullptr ) return std::nullopt;
            Eigen::Matrix< T, State::RowsAtCompileTime, 1 > out{ best->flowMap.size() };
            for ( Eigen::Index i = 0; i < best->flowMap.size(); ++i )
                out( i ) = best->flowMap( i ).eval( bestXi );
            return out;
        }

       private:
        friend AdsSolution;

        void reserve( std::size_t n ) { leaves_.reserve( n ); }
        void add( const Domain& domain, const State& x, int depth, int id )
        {
            leaves_.push_back( LeafView{ domain, x, depth, id } );
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
    [[nodiscard]] Tree& tree() noexcept
    {
        return tree_;
    }  // mutable access, e.g. for merge(sol.tree(), crit)
    [[nodiscard]] const LeafSol& leaf( int idx ) const noexcept
    {
        return leafSol_[static_cast< std::size_t >( idx )];
    }
    [[nodiscard]] T t0() const noexcept { return t0_; }
    [[nodiscard]] T t1() const noexcept { return t1_; }

    // Evaluate the FINAL piecewise-polynomial flow map at a physical IC point:
    // locate the done leaf owning pt (exact factor recovery) and evaluate its
    // payload there. nullopt if no leaf claims the point.
    template < class Derived >
    [[nodiscard]] std::optional< Eigen::Matrix< T, State::RowsAtCompileTime, 1 > > evaluate(
        const Eigen::MatrixBase< Derived >& pt, T tol = T{ 1e-9 } ) const
        requires tax::domain::LocatableDomain< Domain >
    {
        const auto loc = tree_.locateFactors( pt, tol );
        if ( !loc ) return std::nullopt;
        const State& payload = tree_.leaf( loc->idx ).payload;
        Eigen::Matrix< T, State::RowsAtCompileTime, 1 > out{ payload.size() };
        for ( Eigen::Index i = 0; i < payload.size(); ++i ) out( i ) = payload( i ).eval( loc->xi );
        return out;
    }

    // The accepted partition at t1: the done leaves with their final flow maps.
    [[nodiscard]] Partition final() const
    {
        std::vector< Cand > cands;
        for ( int idx : tree_.done() )
        {
            const auto& lf = tree_.leaf( idx );
            cands.push_back( Cand{ &lf.domain, &lf.payload, lf.depth } );
        }
        return makePartition( t1_, std::move( cands ) );
    }

    // One partition per synchronized snapshot time, bracketed by t0 and t1.
    [[nodiscard]] std::vector< Partition > snapshots() const
    {
        // Gather all reserved-label event records across every arena leaf.
        std::vector< Rec > recs;
        // Iterate ALL arena leaves, including retired parents: a retired parent
        // holds the grid records for times before its split, where its domain is
        // the active partition member. Excluding them would leave gaps before
        // splits.
        for ( std::size_t idx = 0; idx < leafSol_.size(); ++idx )
        {
            const auto& ls = leafSol_[idx];
            const auto& lf = tree_.leaf( static_cast< int >( idx ) );
            for ( const auto& e : ls.events )
                if ( e.label == kSnapshotLabel )
                    recs.push_back( Rec{ e.t, &lf.domain, &e.x, lf.depth } );
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
                cands.push_back( Cand{ recs[j].domain, recs[j].x, recs[j].depth } );
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
        const Domain* domain;
        const State* x;
        int depth;
    };
    struct Rec
    {
        T t;
        const Domain* domain;
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

    // Sort candidates into the canonical domain order (tax::domain::domainLess,
    // via ADL — the same total order AdsTree::canonicalizeDone uses, so ids are
    // reproducible and match across serial/parallel runs and PZ trees).
    [[nodiscard]] Partition makePartition( T t, std::vector< Cand > cands ) const
    {
        std::sort( cands.begin(), cands.end(), []( const Cand& a, const Cand& b ) {
            return domainLess( *a.domain, *b.domain );
        } );
        Partition p{ t };
        p.reserve( cands.size() );
        for ( std::size_t id = 0; id < cands.size(); ++id )
            p.add( *cands[id].domain, *cands[id].x, cands[id].depth, static_cast< int >( id ) );
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
            cands.push_back( Cand{ &lf.domain, &ls.x.front(), lf.depth } );
        }
        return makePartition( t, std::move( cands ) );
    }

    Tree tree_;
    std::vector< LeafSol > leafSol_;
    T t0_;
    T t1_;
};

}  // namespace tax::ads
