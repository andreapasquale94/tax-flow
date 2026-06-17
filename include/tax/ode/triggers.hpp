// include/tax/ode/triggers.hpp
//
// Trigger factories. A Trigger has signature
//   std::optional<T>(const StepperCtx<Stepper, State, T, DenseData>&)
// returning the τ in [0, h_used] at which the event fires, or
// std::nullopt to indicate "did not fire on this step".
//
// EveryStep        — fires at the step boundary (τ = h_used).
// ZeroCrossing(g)  — locates a root of g(x(τ), t_old + τ).
//
// For TaylorStepper's DenseData (Eigen vector of tax::TE<N, 1>), if g
// is invocable with TE-valued state we compose g with the per-step
// expansion to obtain a univariate TE g_poly and run safeguarded
// polynomial-Newton. Otherwise we fall back to Brent on scalar samples
// via Stepper::eval_dense.

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <tax/core/taylor_expansion.hpp>
#include <tax/la/types.hpp>
#include <tax/ode/detail/brent_root.hpp>
#include <tax/ode/event.hpp>
#include <type_traits>
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

namespace detail
{

// Compile-time detection: is DenseData an Eigen::Matrix whose Scalar
// is a tax::TE-like type (i.e. exposes a static order_v and supports
// operator[]).
template < class, class = void >
struct has_te_scalar : std::false_type
{
};

template < class D >
struct has_te_scalar< D, std::void_t< typename D::Scalar, decltype( D::Scalar::order_v ) > >
    : std::true_type
{
};

}  // namespace detail

// ZeroCrossing(g, dir) — locate a root of g(x(τ), t_old + τ) in
// (0, h_used).
template < class GFn >
auto ZeroCrossing( GFn g, Direction dir = Direction::Any )
{
    return [g = std::move( g ),
            dir]< class Ctx >( const Ctx& ctx ) -> std::optional< typename Ctx::T_type > {
        using T = typename Ctx::T_type;
        using State = typename Ctx::State_type;
        using DenseData = typename Ctx::DenseData_type;
        using Stepper = typename Ctx::Stepper_type;

        // Scalar evaluation at the step boundaries (always supported —
        // user g is at minimum invocable on the concrete State type).
        const T s0 = T( g( ctx.x_old, ctx.t_old ) );
        const T s1 = T( g( ctx.x_new, ctx.t_old + ctx.h_used ) );
        if ( !dir_match( s0, s1, dir ) ) return std::nullopt;

        // Polynomial path: only when DenseData scalar is a TE-like type.
        if constexpr ( detail::has_te_scalar< DenseData >::value )
        {
            using TE = typename DenseData::Scalar;
            constexpr int Order = TE::order_v;
            constexpr int Rows = State::RowsAtCompileTime;
            using StateTE = tax::la::VecNT< Rows, TE >;

            // Compile-time check that g accepts TE-valued state.
            if constexpr ( std::is_invocable_v< const GFn&, const StateTE&, const TE& > )
            {
                const Eigen::Index dim = ctx.dense.size();

                // x_te(τ) = per-step expansion in τ; copy from dense.
                StateTE x_te{ dim };
                for ( Eigen::Index i = 0; i < dim; ++i ) x_te( i ) = ctx.dense( i );

                // t_te(τ) = t_old + τ as a TE in τ.
                TE t_te;
                t_te[0] = ctx.t_old;
                if constexpr ( Order >= 1 ) t_te[1] = T{ 1 };
                for ( int k = 2; k <= Order; ++k ) t_te[static_cast< std::size_t >( k )] = T{ 0 };

                TE g_poly = g( x_te, t_te );

                // Formal derivative of g_poly: coeff_k(g_poly') =
                // (k+1) * coeff_{k+1}(g_poly); top coefficient is 0.
                TE g_poly_deriv;
                for ( int k = 0; k <= Order; ++k )
                {
                    g_poly_deriv[static_cast< std::size_t >( k )] =
                        ( k + 1 <= Order )
                            ? T( k + 1 ) * g_poly[static_cast< std::size_t >( k + 1 )]
                            : T{ 0 };
                }

                // Safeguarded Newton on g_poly within τ ∈ [0, h_used].
                T tau_lo = T{ 0 };
                T tau_hi = ctx.h_used;
                T flo = s0;
                T tau = ( tau_lo + tau_hi ) * T{ 0.5 };

                for ( int it = 0; it < 50; ++it )
                {
                    // Horner: g_poly(tau) and g_poly'(tau).
                    T eval = g_poly[static_cast< std::size_t >( Order )];
                    for ( int k = Order - 1; k >= 0; --k )
                        eval = eval * tau + g_poly[static_cast< std::size_t >( k )];

                    T eval_d = g_poly_deriv[static_cast< std::size_t >( Order )];
                    for ( int k = Order - 1; k >= 0; --k )
                        eval_d = eval_d * tau + g_poly_deriv[static_cast< std::size_t >( k )];

                    // Tighten bracket using sign of eval vs flo.
                    if ( ( flo < T{ 0 } ) == ( eval < T{ 0 } ) )
                    {
                        tau_lo = tau;
                        flo = eval;
                    } else
                    {
                        tau_hi = tau;
                    }

                    // Convergence on bracket width.
                    const T width = tau_hi - tau_lo;
                    const T mid = ( tau_hi + tau_lo ) * T{ 0.5 };
                    const T scale = T{ 1 } + std::abs( mid );
                    if ( width < T{ 16 } * std::numeric_limits< T >::epsilon() * scale ) return mid;

                    // Newton trial; fall back to bisection if it leaves
                    // the bracket.
                    const T tau_newton =
                        ( eval_d != T{ 0 } ) ? tau - eval / eval_d : ( tau_lo + tau_hi ) * T{ 0.5 };
                    tau = ( tau_newton > tau_lo && tau_newton < tau_hi )
                              ? tau_newton
                              : ( tau_lo + tau_hi ) * T{ 0.5 };
                }
                return ( tau_lo + tau_hi ) * T{ 0.5 };
            }
            // else: g not invocable with TE — fall through to Brent.
        }

        // Brent fallback on scalar samples via Stepper::eval_dense.
        auto sample = [&]( T tau ) -> T {
            auto x_at = Stepper::eval_dense( ctx.dense, ctx.t_old, ctx.t_old + tau );
            return T( g( x_at, ctx.t_old + tau ) );
        };
        return detail::brent_root< T >( sample, T{ 0 }, ctx.h_used, s0, s1 );
    };
}

}  // namespace tax::ode
