// include/tax/ads/driver.hpp
//
// AdsDriver<Stepper, Criterion> — BFS driver around tax::ode::Integrator.
// For each leaf in the work queue, the driver runs the integrator with
// a SplitEvent appended to per-leaf clones of any user-supplied events.
// If the split event fires, the leaf is replaced by two children with
// the parent's DA state re-identified on each half via tax::ads::split.
// Otherwise the leaf is marked done with the propagated DA flow map
// stored as its payload.

#pragma once

#include <memory>
#include <tax/ads/da_state.hpp>
#include <tax/ads/detail/work_pool.hpp>
#include <tax/ads/domains/box.hpp>
#include <tax/ads/solution.hpp>
#include <tax/ads/split_event.hpp>
#include <tax/ads/tree.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/la/types.hpp>
#include <tax/ode/event.hpp>
#include <tax/ode/integrator.hpp>
#include <type_traits>
#include <utility>
#include <vector>

namespace tax::ads
{

// Domain defaults to void → resolved below to Box<T,M> (T,M derive from Stepper
// and can't be named in the template default). Pass e.g. Zonotope<T,M> for
// oriented leaves.
template < class Stepper, class Criterion, class Domain = void >
class AdsDriver
{
   public:
    using State = typename Stepper::State;
    using T = typename Stepper::T;
    using Cfg = typename Stepper::Config;
    using ExtraEvt = std::vector< std::shared_ptr< tax::ode::Event< State, T > > >;

    using TE = typename State::Scalar;
    static constexpr int N = TE::order_v;
    static constexpr int M = TE::vars_v;
    static constexpr int D = State::RowsAtCompileTime;

    using DomainT = std::conditional_t< std::is_void_v< Domain >, Box< T, M >, Domain >;
    using Tree = AdsTree< State, M, T, DomainT >;
    using BoxT = DomainT;
    using LeafSol = tax::ode::Solution< Stepper, State >;

    // The driver uses Stepper::T as the (real) time/scalar type. Embedded-RK
    // steppers expose T == double; TaylorStepper exposes T == State::Scalar (a
    // TaylorExpansion), which would make t0/t1/cfg expansion-valued. Reject that
    // here with a clear message rather than failing deep in instantiation.
    static_assert( std::is_floating_point_v< T >,
                   "tax::ads requires a stepper whose ::T is a real scalar (e.g. an "
                   "embedded-RK method); methods::Taylor<N> is not supported." );

    AdsDriver( Criterion crit, Cfg cfg, ExtraEvt extras = {}, int num_threads = 1 )
        : crit_( std::move( crit ) ),
          cfg_( std::move( cfg ) ),
          extras_( std::move( extras ) ),
          num_threads_( num_threads < 1 ? 1 : num_threads )
    {
    }

    template < class F >
    [[nodiscard]] AdsSolution< Stepper, M, DomainT > run( F&& rhs, const BoxT& ic_box,
                                                          const Eigen::Matrix< T, D, 1 >& ic_center,
                                                          T t0, T t1 )
    {
        Tree tree;
        State root_state = tax::ads::create< N, M >( ic_box, ic_center );
        (void)tree.init( ic_box, std::move( root_state ), t0 );

        std::vector< LeafSol > leafSol;
        if ( num_threads_ > 1 )
            driveParallel( rhs, tree, leafSol, t1 );
        else
            driveSerial( rhs, tree, leafSol, t1 );

        tree.canonicalizeDone();
        return AdsSolution< Stepper, M, DomainT >{ std::move( tree ), std::move( leafSol ), t0,
                                                   t1 };
    }

   protected:
    // Outcome of integrating one leaf: either split into two children or
    // finalize with a flow-map payload. Computed lock-free in stepLeaf.
    struct LeafVerdict
    {
        bool split = false;
        int dim = -1;
        T splitTime{};
        State left{};
        State right{};
        LeafSol leafSol{};
    };

    // Pure, lock-free: integrate one leaf from tEntry to t1 with the
    // split event appended, and decide split-vs-finalize. Reads only
    // crit_, cfg_, extras_ (all const) and the passed-in rhs / inputs.
    template < class F >
    [[nodiscard]] LeafVerdict stepLeaf( const F& rhs, const State& payload, T tEntry, int depth,
                                        const BoxT& box, T t1 ) const
    {
        SplitRequest< T > req;
        tax::ode::Integrator< Stepper, std::decay_t< F > > integ{ rhs, cfg_ };
        // Per-leaf independent instances of any user extras (deep clone).
        for ( const auto& e : extras_ ) integ.addEvent( e->clone() );
        integ.addEvent(
            std::make_shared< SplitEvent< State, T, Criterion > >( crit_, depth, &req ) );
        auto sol = integ.integrate( payload, tEntry, t1 );

        // Guard against a split fired at (or beyond) the final time —
        // splitting would queue two children with tEntry == t1, and the
        // integrator rejects empty intervals.
        const bool atFinal = req.fired && !( req.t < t1 );

        LeafVerdict v;
        if ( req.fired && !atFinal )
        {
            v.split = true;
            v.dim = req.dim;
            v.splitTime = req.t;
            auto pr = tax::ads::split( sol.x.back(), req.dim );
            v.left = std::move( pr.first );
            v.right = std::move( pr.second );
        } else
        {
            v.split = false;
        }
        v.leafSol = std::move( sol );  // capture events + steps (read sol.x.back() above first)
        return v;
    }

    template < class F >
    void driveSerial( const F& rhs, Tree& tree, std::vector< LeafSol >& leafSol, T t1 )
    {
        while ( !tree.empty() )
        {
            const int idx = tree.popFront();
            const auto& l = tree.leaf( idx );

            LeafVerdict v = stepLeaf( rhs, l.payload, l.tEntry, l.depth, l.box, t1 );

            if ( static_cast< int >( leafSol.size() ) <= idx )
                leafSol.resize( static_cast< std::size_t >( idx ) + 1 );

            if ( v.split )
            {
                (void)tree.split( idx, v.dim, std::move( v.left ), std::move( v.right ),
                                  v.splitTime );
            } else
            {
                tree.leaf( idx ).payload = v.leafSol.x.back();
                tree.finalize( idx );
            }
            leafSol[static_cast< std::size_t >( idx )] = std::move( v.leafSol );
        }
    }

    // Parallel scheduler: num_threads_ workers pull ready leaves from the
    // tree's work queue (which IS the AdsTree). The expensive integration
    // (stepLeaf) runs lock-free on copied-out inputs; the shared work-pool
    // (detail::parallelDrive) guards only the cheap queue access and tree
    // mutation plus the in_flight counter. Termination, exception capture,
    // and the notify policy are owned by the pool.
    template < class F >
    void driveParallel( const F& rhs, Tree& tree, std::vector< LeafSol >& leafSol, T t1 )
    {
        // One leaf taken out of the tree for lock-free integration. tree.split
        // appends to the arena vector and may reallocate, so we copy/move the
        // inputs out rather than hold references across the lock-free work:
        // indices stay valid across reallocation; references do not.
        struct Item
        {
            int idx = -1;
            State payload{};
            T tEntry{};
            int depth = 0;
            BoxT box{};
        };
        // stepLeaf's verdict, tagged with the originating leaf index so apply()
        // can commit it without the item.
        struct Result
        {
            int idx = -1;
            LeafVerdict v{};
        };

        detail::parallelDrive(
            num_threads_,
            // empty(): under lock.
            [&] { return tree.empty(); },
            // pop(): under lock — take the front leaf's inputs out.
            [&]() -> Item {
                const int idx = tree.popFront();
                Item it;
                it.idx = idx;
                it.payload = std::move( tree.leaf( idx ).payload );
                it.tEntry = tree.leaf( idx ).tEntry;
                it.depth = tree.leaf( idx ).depth;
                it.box = tree.leaf( idx ).box;
                return it;
            },
            // process(): lock-free integration on the copied-out item.
            [&]( Item it ) -> Result {
                return Result{ it.idx,
                               stepLeaf( rhs, it.payload, it.tEntry, it.depth, it.box, t1 ) };
            },
            // apply(): under lock — split or finalize, capture the per-leaf
            // Solution, and report whether work was enqueued (split → 2 leaves).
            [&]( Result r ) -> bool {
                const int idx = r.idx;
                LeafVerdict& v = r.v;
                if ( static_cast< int >( leafSol.size() ) <= idx )
                    leafSol.resize( static_cast< std::size_t >( idx ) + 1 );
                bool produced;
                if ( v.split )
                {
                    (void)tree.split( idx, v.dim, std::move( v.left ), std::move( v.right ),
                                      v.splitTime );
                    produced = true;  // two new leaves are now available to claim
                } else
                {
                    tree.leaf( idx ).payload = v.leafSol.x.back();
                    tree.finalize( idx );
                    produced = false;
                }
                leafSol[static_cast< std::size_t >( idx )] = std::move( v.leafSol );
                return produced;
            } );
    }

    Criterion crit_;
    Cfg cfg_;
    ExtraEvt extras_;
    int num_threads_ = 1;
};

}  // namespace tax::ads
