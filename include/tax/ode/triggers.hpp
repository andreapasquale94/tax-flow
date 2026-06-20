// include/tax/ode/triggers.hpp
//
// Trigger factories. A Trigger has signature
//   std::optional<T>(const StepperCtx<Stepper, State, T>&)
// returning the τ in [0, h_used] at which the event fires, or
// std::nullopt to indicate "did not fire on this step".
//
// EveryStep        — fires at the step boundary (τ = h_used).
// ZeroCrossing(g)  — locates a root of g(x(τ), t_old + τ) via endpoint
//                    sign-check followed by Brent on the integrator-provided
//                    flow(τ) callable (Taylor: expansion eval; RK: full-order
//                    re-step).

#pragma once

#include <optional>
#include <tax/ode/detail/brent_root.hpp>
#include <tax/ode/event.hpp>
#include <utility>

namespace tax::ode
{

// Direction filter: returns true if (s0, s1) is a crossing of the
// requested type.
template < class T >
[[nodiscard]] inline bool dir_match( T s0, T s1, Direction d ) noexcept
{
    // Half-open crossing detection: s0 must be strictly on one side of zero and
    // s1 must reach or pass zero. Including s1 == 0 catches a root that lands
    // exactly on a step boundary; excluding s0 == 0 prevents the next step (whose
    // s0 is that same zero) from firing it a second time. (A root exactly at the
    // initial point t0 is therefore not reported — the integrator starts on it.)
    const bool up = ( s0 < T{ 0 } && s1 >= T{ 0 } );
    const bool down = ( s0 > T{ 0 } && s1 <= T{ 0 } );
    switch ( d )
    {
        case Direction::Any:
            return up || down;
        case Direction::Increasing:
            return up;
        case Direction::Decreasing:
            return down;
    }
    return up || down;
}

// EveryStep — fires at the boundary, τ = h_used.
inline auto EveryStep()
{
    return []< class Ctx >( const Ctx& ctx ) -> std::optional< typename Ctx::T_type > {
        return ctx.h_used;
    };
}

// ZeroCrossing(g, dir) — locate a root of g(x(τ), t_old + τ) in
// (0, h_used) using Brent on scalar samples drawn from the integrator-
// provided flow(τ) callable (accurate to the method order).
template < class GFn >
auto ZeroCrossing( GFn g, Direction dir = Direction::Any )
{
    return [g = std::move( g ),
            dir]< class Ctx >( const Ctx& ctx ) -> std::optional< typename Ctx::T_type > {
        using T = typename Ctx::T_type;

        const T s0 = T( g( ctx.x_old, ctx.t_old ) );
        const T s1 = T( g( ctx.x_new, ctx.t_old + ctx.h_used ) );
        if ( !dir_match( s0, s1, dir ) ) return std::nullopt;

        auto sample = [&]( T tau ) -> T { return T( g( ctx.flow( tau ), ctx.t_old + tau ) ); };
        return detail::brent_root< T >( sample, T{ 0 }, ctx.h_used, s0, s1 );
    };
}

}  // namespace tax::ode
