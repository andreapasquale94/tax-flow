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

            Item item = pop();
            ++in_flight;
            lk.unlock();

            // Expensive, lock-free work on copied-out inputs.
            Verdict verdict{};
            try
            {
                verdict = process( std::move( item ) );
            } catch ( ... )
            {
                lk.lock();
                if ( !first_err ) first_err = std::current_exception();
                stopping = true;
                --in_flight;
                cv.notify_all();
                return;
            }

            lk.lock();
            const bool produced = apply( std::move( verdict ) );
            --in_flight;
            const bool do_notify = produced || ( in_flight == 0 && empty() );
            lk.unlock();
            // Notify outside the lock so woken workers don't immediately contend
            // on the mutex we still hold.
            if ( do_notify ) cv.notify_all();
        }
    };

    const int n = num_threads < 1 ? 1 : num_threads;
    std::vector< std::thread > pool;
    pool.reserve( static_cast< std::size_t >( n ) );
    for ( int i = 0; i < n; ++i ) pool.emplace_back( worker );
    for ( auto& th : pool ) th.join();

    if ( first_err ) std::rethrow_exception( first_err );
}

}  // namespace tax::ads::detail
