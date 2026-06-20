// include/tax/ode/detail/embedded_rk_stepper.hpp
//
// EmbeddedRKStepper<Tab, State, Controller> — the one explicit
// embedded-pair Runge-Kutta stepper. Every tableau-driven method
// (Verner 8(7)/9(8), Fehlberg 7(8), Feagin 12(10)/14(12)) is an alias
// of this template over its Butcher tableau; the per-method headers in
// steppers/ contain only the alias and the method documentation.
//
// Event location uses a controller-free re-step (step()) at full
// method order — no cubic-Hermite approximation.

#pragma once

#include <functional>
#include <limits>
#include <tax/la/types.hpp>
#include <tax/ode/config.hpp>
#include <tax/ode/controllers.hpp>
#include <tax/ode/detail/adaptive_rk_step.hpp>
#include <tax/ode/step_result.hpp>
#include <tax/ode/vector_ops.hpp>
#include <type_traits>
#include <utility>

namespace tax::ode::detail
{

template < class TabT, class StateT, class Controller = controllers::PI< double > >
struct EmbeddedRKStepper
{
    using State = StateT;
    using T = double;
    using Config = IntegratorConfig< double >;
    using Rhs = std::function< State( const State&, T ) >;
    using Tab = TabT;

    static constexpr bool is_adaptive = true;
    static constexpr bool has_step_expansion = false;
    static constexpr int order_v = Tab::order;
    static constexpr int order_emb_v = Tab::order_emb;

    // StepData: empty — RK steppers use step() for event location.
    struct StepData
    {
    };

    template < class F >
    [[nodiscard]] StepResult< State, EmbeddedRKStepper > step( F&& f, const State& x, T t, T h,
                                                               const Config& cfg )
    {
        RKStepData< State, Tab::n_stages > work;
        auto out = adaptive_rk_step< Tab >( f, x, t, h, work );

        const double x_norm = VectorOps< State >::norm( out.x_new );
        const double tol = cfg.abstol + cfg.reltol * x_norm;

        // Embedded estimators built from stage differences (Feagin's
        // `(k_2 - k_{n-1})`, the "Fehlberg coincidence") can underflow
        // to exactly zero on benign integrands at small h. Floor the
        // error norm at machine eps * tol so the controller can grow
        // the step instead of treating zero error as "use default
        // factor".
        const double err_for_ctrl =
            ( out.err_norm > 0.0 ) ? out.err_norm : tol * std::numeric_limits< double >::epsilon();

        const auto [h_next, accepted] =
            select_rk_step( controller_, h, err_for_ctrl, out.err_norm, tol, Tab::order_emb );

        StepResult< State, EmbeddedRKStepper > r;
        r.x_new = std::move( out.x_new );
        r.h_used = h;
        r.data = {};
        r.h_next = h_next;
        r.err_norm = out.err_norm;
        r.accepted = accepted;
        return r;
    }

    // Controller-free re-step to obtain state at t + tau. Used by the
    // integrator's per-step flow closure for accurate event location at
    // full method order.
    template < class F >
    [[nodiscard]] static State step( F&& f, const State& x, T t, T tau )
    {
        RKStepData< State, Tab::n_stages > work;
        return adaptive_rk_step< Tab >( std::forward< F >( f ), x, t, tau, work ).x_new;
    }

   private:
    Controller controller_{};
};

}  // namespace tax::ode::detail
