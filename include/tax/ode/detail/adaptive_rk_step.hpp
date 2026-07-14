// include/tax/ode/detail/adaptive_rk_step.hpp
//
// Generic explicit Runge–Kutta step driver. Routes state arithmetic
// through tax::ode::VectorOps<State>, which decouples step-size
// control (always double) from the state's scalar layout. Same body
// serves double-state and DA-vector-state.

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <tax/ode/controllers.hpp>
#include <tax/ode/vector_ops.hpp>
#include <type_traits>
#include <utility>

namespace tax::ode::detail
{

template < class State, int NStages >
struct RKStepData
{
    std::array< State, NStages > k{};
};

template < class State >
struct RKStepOut
{
    State x_new;
    State y_emb;
    double err_norm;  // always double
};

template < class Tab, class F, class State, int NStages >
[[nodiscard]] RKStepOut< State > adaptive_rk_step( F&& f, const State& x, double t, double h,
                                                   RKStepData< State, NStages >& work )
{
    static_assert( NStages == Tab::n_stages,
                   "adaptive_rk_step: stage-count mismatch with tableau" );

    using Ops = VectorOps< State >;

    work.k[0] = f( x, t + Tab::c[0] * h );

    std::size_t a_off = 0;
    State y = x;  // hoisted stage accumulator (reassigned per stage)
    for ( int i = 1; i < NStages; ++i )
    {
        if ( i > 1 ) y = x;
        for ( int j = 0; j < i; ++j )
            Ops::axpy( y, h * Tab::a[a_off + std::size_t( j )], work.k[std::size_t( j )] );
        work.k[std::size_t( i )] = f( y, t + Tab::c[std::size_t( i )] * h );
        a_off += std::size_t( i );
    }

    State x_new = x;
    State y_emb = x;
    for ( int i = 0; i < NStages; ++i )
    {
        Ops::axpy( x_new, h * Tab::b[std::size_t( i )], work.k[std::size_t( i )] );
        Ops::axpy( y_emb, h * Tab::b_emb[std::size_t( i )], work.k[std::size_t( i )] );
    }

    // Error norm |x_new - y_emb|_inf, without materializing the difference
    // when the VectorOps specialization provides diff_norm.
    double err_norm;
    if constexpr ( requires( const State& u, const State& v ) {
                       { Ops::diff_norm( u, v ) } -> std::convertible_to< double >;
                   } )
    {
        err_norm = Ops::diff_norm( x_new, y_emb );
    } else
    {
        State diff = x_new;
        Ops::axpy( diff, -1.0, y_emb );
        err_norm = Ops::norm( diff );
    }

    return { std::move( x_new ), std::move( y_emb ), err_norm };
}

/**
 * @brief Resolve `(h_next, accepted)` for the next RK step.
 *
 * Centralises the three-arm controller dispatch shared by every
 * Stage 2a RK stepper:
 *   - `FixedStep` ............ always accepted, h_next = h.
 *   - `JorbaZou` ............. Taylor-only; no-op fallback for RK.
 *   - any other controller ... `controller.next_step(...)` + tolerance check.
 *
 * Feagin pairs feed their floored error to the controller via
 * `err_for_ctrl` while still using the raw `err_norm` for the
 * acceptance decision; all other RK steppers pass the same value for
 * both arguments.
 *
 * @param controller     Controller instance (state-mutating for PI/H211b).
 * @param h              Step size that was just taken.
 * @param err_for_ctrl   Error magnitude handed to the controller.
 * @param err_norm       Raw error magnitude compared against `tol`.
 * @param tol            Acceptance threshold.
 * @param order_emb      Embedded estimator order (Tab::order_emb).
 */
template < class Controller >
[[nodiscard]] inline std::pair< double, bool > select_rk_step( Controller& controller, double h,
                                                               double err_for_ctrl, double err_norm,
                                                               double tol, int order_emb )
{
    static_assert( !std::is_same_v< Controller, controllers::JorbaZou< double > >,
                   "JorbaZou is a Taylor-method controller (it needs the last two "
                   "time-Taylor coefficient magnitudes). Use it only with TaylorStepper, "
                   "not with the RK steppers." );

    if constexpr ( std::is_same_v< Controller, controllers::FixedStep< double > > )
    {
        (void)controller;
        (void)err_for_ctrl;
        (void)err_norm;
        (void)tol;
        (void)order_emb;
        return { h, true };
    } else
    {
        return { controller.next_step( h, err_for_ctrl, tol, order_emb ), err_norm <= tol };
    }
}

/**
 * @brief Resolve `(h_next, accepted)` for the next Taylor step.
 *
 * Mirrors @ref select_rk_step, but covers the two Taylor-specific
 * controller signatures:
 *   - `FixedStep` ....... always accepted, h_next = h.
 *   - `JorbaZou`  ....... 5-arg call `(h, c_N_norm, c_Nm1_norm, tol, N)`
 *                         that uses the last two time-Taylor coefficient
 *                         magnitudes directly.
 *   - any other ......... 4-arg call `(h, err_norm, tol, p_emb=N-1)`
 *                         consuming the truncation-indicator residual.
 *
 * `N` is the Taylor stepper's compile-time time order.
 */
template < int N, class Controller >
[[nodiscard]] inline std::pair< double, bool > select_taylor_step( Controller& controller, double h,
                                                                   double c_N_norm,
                                                                   double c_Nm1_norm,
                                                                   double err_norm, double tol )
{
    if constexpr ( std::is_same_v< Controller, controllers::FixedStep< double > > )
    {
        (void)controller;
        (void)c_N_norm;
        (void)c_Nm1_norm;
        (void)err_norm;
        (void)tol;
        return { h, true };
    } else if constexpr ( std::is_same_v< Controller, controllers::JorbaZou< double > > )
    {
        return { controller.next_step( h, c_N_norm, c_Nm1_norm, tol, N ), err_norm <= tol };
    } else
    {
        (void)c_N_norm;
        (void)c_Nm1_norm;
        // Floor a zero truncation indicator (exact on polynomial RHS, where the
        // top Taylor coefficients vanish) at eps*tol, mirroring the RK path. A
        // literal zero would make the generic I/PI controllers SHRINK the step
        // (ratio→1, factor 0.9) and stall the integration at min_step.
        const double err_for_ctrl =
            ( err_norm > 0.0 ) ? err_norm : tol * std::numeric_limits< double >::epsilon();
        return { controller.next_step( h, err_for_ctrl, tol, /*p_emb=*/N - 1 ), err_norm <= tol };
    }
}

}  // namespace tax::ode::detail
