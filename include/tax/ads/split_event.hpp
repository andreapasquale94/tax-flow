// include/tax/ads/split_event.hpp
//
// Interop with tax::ode::Event. A SplitRequest is the side-channel from
// the driver-owned split event back to the BFS driver: the trigger asks
// the criterion whether the current DA state demands a split, and the
// action records {fired, dim, t} into the request and returns
// ControlFlow::Terminate. The integrator then truncates the solution at
// the step boundary; the driver consumes req to decide split-vs-done.
//
// Note: tax::ode is not modified. Trigger and Action satisfy ode's
// std::function-erased signatures.

#pragma once

#include <optional>
#include <tax/ode/event.hpp>

namespace tax::ads
{

template < class T >
struct SplitRequest
{
    bool fired = false;
    int dim = -1;
    T t = T{ 0 };
};

// Trigger: fires at the step boundary iff criterion.shouldSplit is true.
// `depth` is captured by value at construction (the driver builds a new
// event per BFS leaf, so the leaf's depth is known).
template < class Criterion >
[[nodiscard]] auto SplitTrigger( Criterion crit, int depth )
{
    return [crit = std::move( crit ),
            depth]< class Ctx >( const Ctx& ctx ) -> std::optional< typename Ctx::T_type > {
        if ( crit.shouldSplit( ctx.x_new, depth ) ) return ctx.h_used;
        return std::nullopt;
    };
}

// Action: write {fired, dim, t} into *out and Terminate. The caller
// keeps `out` alive for the duration of Integrator::integrate().
template < class Criterion, class T >
[[nodiscard]] auto SplitAction( Criterion crit, SplitRequest< T >* out )
{
    return [crit = std::move( crit ), out]< class Ctx, class TT, class Storage >(
               const Ctx& ctx, TT tau, Storage& ) -> tax::ode::ControlFlow {
        out->fired = true;
        out->dim = crit.splitDim( ctx.x_new );
        out->t = ctx.t_old + static_cast< T >( tau );
        return tax::ode::ControlFlow::Terminate;
    };
}

}  // namespace tax::ads
