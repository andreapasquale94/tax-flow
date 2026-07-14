// include/tax/ads/detail/work_pool.hpp
//
// parallelDrive — the shared parallel work-queue scaffold used by both the
// classic AdsDriver (driveParallel) and the AdsRefineDriver (drive). It owns
// the mutex / condition_variable / in-flight counter / stopping flag /
// first-exception machinery and the worker pool spawn-join + final rethrow,
// so each driver only supplies the four behaviours that differ:
//
//   - empty()        : bool, called UNDER the pool lock — is the queue empty?
//   - pop()          : returns the next work item, called UNDER the pool lock
//                      (the pool increments in_flight right after).
//   - process(item)  : the expensive work, called LOCK-FREE on the popped item;
//                      must touch only the popped item, never shared state.
//                      Returns a verdict consumed by apply().
//   - apply(verdict) : bool, called UNDER the pool lock — commit the result
//                      (mutate the tree / queue / side-channels) and RETURN
//                      whether it enqueued new work.
//
// Notify policy (single, unified): after applying a result the pool decrements
// in_flight and notifies iff (producedWork || (in_flight == 0 && empty())),
// OUTSIDE the lock so woken workers don't immediately contend on the mutex.
// Producing work makes leaves claimable; reaching quiescence lets idle workers
// observe termination. Both drivers adopt this policy — for the refine driver
// it replaces an unconditional notify-inside-the-lock, a benign wakeup-timing
// change that leaves the computed results identical.
//
// Termination: each worker blocks until the queue is non-empty, the pool is
// stopping, or nothing is in flight; it returns once the queue is drained with
// no task in flight. The first worker exception wins, sets stopping, and is
// rethrown on the calling thread after the pool joins.

#pragma once

#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace tax::ads::detail
{

// Runs `num_threads` workers over a work queue until it drains with no task in
// flight. The item and verdict types are deduced from the callbacks, so the
// pool never names them. See the file header for the callback contract.
template < class EmptyFn, class PopFn, class ProcessFn, class ApplyFn >
void parallelDrive( int num_threads, EmptyFn empty, PopFn pop, ProcessFn process, ApplyFn apply )
{
    using Item = decltype( pop() );
    using Verdict = decltype( process( std::declval< Item >() ) );

    std::mutex mtx;
    std::condition_variable cv;
    int in_flight = 0;
    bool stopping = false;
    std::exception_ptr first_err = nullptr;

    auto worker = [&]() {
        for ( ;; )
        {
            std::unique_lock< std::mutex > lk( mtx );
            cv.wait( lk, [&] { return stopping || !empty() || in_flight == 0; } );

            if ( stopping ) return;
            if ( empty() )
            {
                if ( in_flight == 0 )
                {
                    cv.notify_all();  // wake the others to terminate
                    return;
                }
                continue;  // tasks still running may yet enqueue work
            }

            // pop() runs under the lock and can throw (it may move/allocate a
            // work item holding dense DA states). Guard it: a throw here must
            // stop the pool, not std::terminate.
            Item item{};
            try
            {
                item = pop();
            } catch ( ... )
            {
                if ( !first_err ) first_err = std::current_exception();
                stopping = true;
                cv.notify_all();
                return;  // lk releases on scope exit
            }
            ++in_flight;
            lk.unlock();

            // Expensive, lock-free work on copied-out inputs, followed by the
            // locked commit. BOTH are guarded: apply() allocates (arena
            // push_backs, queue growth), so a std::bad_alloc there must be
            // captured and rethrown on the caller, never escape the worker.
            bool produced = false;
            bool failed = false;
            try
            {
                Verdict verdict = process( std::move( item ) );
                lk.lock();
                produced = apply( std::move( verdict ) );
            } catch ( ... )
            {
                if ( !lk.owns_lock() ) lk.lock();
                if ( !first_err ) first_err = std::current_exception();
                stopping = true;
                failed = true;
            }
            // The lock is held here (from apply's lk.lock() or the catch).
            --in_flight;
            const bool do_notify = failed || produced || ( in_flight == 0 && empty() );
            lk.unlock();
            // Notify outside the lock so woken workers don't immediately contend
            // on the mutex we still hold.
            if ( do_notify ) cv.notify_all();
            if ( failed ) return;
        }
    };

    const int n = num_threads < 1 ? 1 : num_threads;
    // std::jthread joins on destruction, so a throw from emplace_back mid-spawn
    // (thread-resource exhaustion) unwinds cleanly instead of terminating on a
    // still-joinable std::thread.
    std::vector< std::jthread > pool;
    pool.reserve( static_cast< std::size_t >( n ) );
    for ( int i = 0; i < n; ++i ) pool.emplace_back( worker );
    for ( auto& th : pool ) th.join();

    if ( first_err ) std::rethrow_exception( first_err );
}

}  // namespace tax::ads::detail
