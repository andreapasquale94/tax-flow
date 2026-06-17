// include/tax/ode/steppers/taylor.hpp
//
// TaylorStepper<N, StateT, Controller>.
//
// Propagates the ODE dx/dt = f(x, t) by computing the Taylor
// expansion of x(t) in time (univariate, order N) around the step
// start. Coefficients are obtained iteratively from f's polynomial
// composition: with x_te[i] holding c_0…c_{k-1} of component i, one
// evaluation of f(x_te, t_te) yields f_te[i].coeff(k) which
// determines x_te[i].coeff(k+1) = f_te[i].coeff(k) / (k+1).
// After N evaluations every coefficient is exact (up to truncation).
//
// Slice 2b ships the shell + eval_dense. The real Taylor expansion
// algorithm lands in slice 2c (Task 6).

#pragma once

#include <tax/la/types.hpp>
#include <cmath>
#include <functional>

#include <tax/core/taylor_expansion.hpp>
#include <tax/la.hpp>
#include <tax/operators/arithmetic.hpp>
#include <tax/operators/math_unary.hpp>
#include <tax/operators/math_binary.hpp>
#include <tax/ode/config.hpp>
#include <tax/ode/controllers.hpp>
#include <tax/ode/detail/adaptive_rk_step.hpp>
#include <tax/ode/step_result.hpp>

namespace tax::ode
{

template < int N,
           class StateT,
           class Controller = controllers::JorbaZou< typename StateT::Scalar > >
struct TaylorStepper
{
    static_assert( N >= 2,
                   "TaylorStepper: order N must be at least 2 for meaningful adaptive control" );

    using State            = StateT;
    using T               = typename StateT::Scalar;
    using Config          = IntegratorConfig< T >;
    using Rhs             = std::function< StateT( const StateT&, T ) >;

    static constexpr int D = StateT::RowsAtCompileTime;  // may be Eigen::Dynamic
    static constexpr bool is_adaptive       = true;
    static constexpr bool has_dense_output  = true;

    // DenseData: per-step Taylor expansion of x(t) in time around the
    // step start. We store one tax::TE<N> per state component.
    using TE        = tax::TE< N, 1 >;
    using DenseData = tax::la::VecNT< D, TE >;

    template < class F >
    [[nodiscard]] StepResult< StateT, TaylorStepper > step(
        const F& f, const StateT& x, T t, T h, const Config& cfg );

    [[nodiscard]] static StateT eval_dense(
        const DenseData& d, const T& t0, const T& tq );

private:
    Controller controller_{};
};

// -------- eval_dense --------
// x(tq) = sum_k d_i.coeff(k) * (tq - t0)^k
template < int N, class StateT, class Controller >
StateT TaylorStepper< N, StateT, Controller >::eval_dense(
    const DenseData& d, const T& t0, const T& tq )
{
    return tax::la::eval( d, T( tq - t0 ) );
}

// -------- step (real Taylor expansion — Task 6) --------
template < int N, class StateT, class Controller >
template < class F >
StepResult< StateT, TaylorStepper< N, StateT, Controller > >
TaylorStepper< N, StateT, Controller >::step(
    const F& f, const StateT& x, T t, T h, const Config& cfg )
{
    using std::abs;
    using std::pow;

    const Eigen::Index dim = x.size();

    // --- 1. Time as a univariate TE: c[0] = t, c[1] = 1, rest 0.
    const TE t_te = TE::variable( t );

    // --- 2. State TE per component: c[0] = x(i), rest 0.
    using StateTE = tax::la::VecNT< StateT::RowsAtCompileTime, TE >;
    StateTE x_te{ dim };
    for ( Eigen::Index i = 0; i < dim; ++i )
        x_te( i ) = TE::constant( x( i ) );

    // --- 3. Iterate to fill coefficients k = 1 .. N.
    for ( int order = 0; order < N; ++order )
    {
        StateTE f_te = f( x_te, t_te );
        for ( Eigen::Index i = 0; i < dim; ++i )
            x_te( i )[ static_cast< std::size_t >( order + 1 ) ] =
                f_te( i )[ static_cast< std::size_t >( order ) ] / T( order + 1 );
    }

    // --- 4. x_new = x(t + h) via the TE Eigen-eval helper.
    StateT x_new = tax::la::eval( x_te, h );

    // --- 5. Truncation indicator and last-two coefficient norms.
    T c_N_norm = T{ 0 }, c_Nm1_norm = T{ 0 };
    for ( Eigen::Index i = 0; i < dim; ++i )
    {
        c_N_norm   = std::max( c_N_norm,   T( abs( x_te( i )[ N ] ) ) );
        c_Nm1_norm = std::max( c_Nm1_norm, T( abs( x_te( i )[ N - 1 ] ) ) );
    }
    const T err_norm =
        c_N_norm * pow( abs( h ), T( N ) )
      + c_Nm1_norm * pow( abs( h ), T( N - 1 ) );

    T x_norm = T{ 0 };
    for ( Eigen::Index i = 0; i < dim; ++i )
        x_norm = std::max( x_norm, T( abs( x_new( i ) ) ) );
    const T tol = cfg.abstol + cfg.reltol * x_norm;

    // --- 6. Step-size control via the Taylor dispatch helper.
    const auto [ h_next, accepted ] = detail::select_taylor_step< N >(
        controller_, h, c_N_norm, c_Nm1_norm, err_norm, tol );

    StepResult< StateT, TaylorStepper > r;
    r.x_new    = std::move( x_new );
    r.h_used   = h;
    r.h_next   = h_next;
    r.err_norm = err_norm;
    r.accepted = accepted;
    r.dense    = std::move( x_te );
    return r;
}

}  // namespace tax::ode
