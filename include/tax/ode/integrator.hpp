// include/tax/ode/integrator.hpp
//
// Method-agnostic adaptive Integrator driven by a compile-time
// Stepper policy. Steppers satisfying concepts::AdaptiveStepper get
// the rejection-and-retry loop via `if constexpr`; bare concepts::
// Stepper (future fixed-step) skips it.
//
// The RHS callable F is template-deduced because the Taylor path
// needs to invoke f on tax::TE-valued state (a std::function with a
// fixed signature would not compose). For ergonomic construction,
// use the type aliases (Verner78, Verner89, Fehlberg78, Feagin12,
// Feagin14, Taylor) defined in <tax/ode/aliases.hpp>.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <tax/la/types.hpp>
#include <tax/ode/actions.hpp>
#include <tax/ode/concepts.hpp>
#include <tax/ode/config.hpp>
#include <tax/ode/event.hpp>
#include <tax/ode/solution.hpp>
#include <tax/ode/triggers.hpp>
#include <utility>
#include <vector>

namespace tax::ode
{

template < concepts::Stepper Stepper, class F, bool Dense = false >
class Integrator
{
   public:
    using State = typename Stepper::State;
    using T = typename Stepper::T;
    using Config = typename Stepper::Config;
    using Solution = tax::ode::Solution< Stepper, State, Dense >;
    using EventList = std::vector< Event< Stepper > >;

    explicit Integrator( F f, Config cfg = {}, EventList events = {} )
        : f_( std::move( f ) ), cfg_( std::move( cfg ) ), events_( std::move( events ) )
    {
        if ( !( cfg_.abstol > T{ 0 } ) )
            throw std::invalid_argument( "IntegratorConfig: abstol must be > 0" );
        if ( !( cfg_.reltol > T{ 0 } ) )
            throw std::invalid_argument( "IntegratorConfig: reltol must be > 0" );
        if ( cfg_.max_steps <= 0 )
            throw std::invalid_argument( "IntegratorConfig: max_steps must be > 0" );
        if ( cfg_.max_rejects_per_step <= 0 )
            throw std::invalid_argument( "IntegratorConfig: max_rejects_per_step must be > 0" );
    }

    [[nodiscard]] Solution integrate( const State& x0, const T& t0, const T& tmax ) const;

   private:
    F f_;
    Config cfg_;
    EventList events_;
};

template < concepts::Stepper Stepper, class F, bool Dense >
typename Integrator< Stepper, F, Dense >::Solution Integrator< Stepper, F, Dense >::integrate(
    const State& x0, const T& t0, const T& tmax ) const
{
    if ( !( tmax > t0 ) ) throw std::invalid_argument( "Integrator::integrate: tmax must be > t0" );

    Solution sol;
    sol.t.push_back( t0 );
    sol.x.push_back( x0 );

    Stepper stepper{};  // per-integration controller state
    State x = x0;
    T t = t0;
    const T span = tmax - t0;
    T h = ( cfg_.initial_step > T{ 0 } ) ? cfg_.initial_step : span / T{ 100 };
    if ( cfg_.max_step > T{ 0 } ) h = std::min( h, cfg_.max_step );
    h = std::min( h, tmax - t );

    const T h_min = ( cfg_.min_step > T{ 0 } )
                        ? cfg_.min_step
                        : std::numeric_limits< T >::epsilon() * std::abs( span ) * T{ 16 };

    EventStorage< State, T > storage{ &sol.events };

    int total_steps = 0;
    bool terminate = false;

    while ( t < tmax && !terminate )
    {
        // Floating-point accumulation in t can leave a sub-h_min remainder
        // to tmax. Treat that as "we are at tmax" and stop, rather than
        // throwing on the inevitable final-step underflow.
        if ( tmax - t < h_min ) break;

        if ( ++total_steps > cfg_.max_steps )
            throw std::runtime_error( "Integrator::integrate: max_steps exceeded" );

        int rejects = 0;
        while ( true )
        {
            if ( h < h_min )
                throw std::runtime_error( "Integrator::integrate: step size below min_step" );

            auto r = stepper.step( f_, x, t, h, cfg_ );

            if constexpr ( concepts::AdaptiveStepper< Stepper > )
            {
                if ( !r.accepted )
                {
                    h = std::max( r.h_next, h_min );
                    // Re-apply the span / max_step clamps on retry: otherwise a
                    // rejection near the end of the interval can retry with an
                    // unclamped step and overshoot tmax.
                    if ( cfg_.max_step > T{ 0 } ) h = std::min( h, cfg_.max_step );
                    h = std::min( h, tmax - t );
                    if ( ++rejects > cfg_.max_rejects_per_step )
                        throw std::runtime_error( "Integrator::integrate: rejection cap reached" );
                    continue;
                }
            }

            // Build the step context once for all events.
            using Ctx = StepperCtx< Stepper, State, T, typename Stepper::DenseData >;
            const Ctx ctx{ x, t, r.x_new, r.h_used, r.dense };

            struct Fired
            {
                T tau;
                std::size_t idx;
            };
            std::vector< Fired > fired;
            fired.reserve( events_.size() );
            for ( std::size_t i = 0; i < events_.size(); ++i )
            {
                auto tau = events_[i].test( ctx );
                if ( tau ) fired.push_back( { *tau, i } );
            }
            std::sort( fired.begin(), fired.end(),
                       []( const Fired& a, const Fired& b ) { return a.tau < b.tau; } );
            // Run fired events in time order. Stop at the first one that asks to
            // terminate: events scheduled strictly after it did not happen yet.
            T term_tau = T{ 0 };
            for ( const auto& fe : fired )
            {
                auto cf = events_[fe.idx].run( ctx, fe.tau, storage );
                if ( cf == ControlFlow::Terminate )
                {
                    terminate = true;
                    term_tau = fe.tau;
                    break;
                }
            }

            t += r.h_used;
            x = r.x_new;

            if ( terminate )
            {
                // Truncate at the *terminating* event's time (term_tau) — not the
                // earliest fired event — using the Stepper's continuous extension
                // (eval_dense) for a machine-precision x_term.
                State x_term = Stepper::eval_dense( ctx.dense, ctx.t_old, ctx.t_old + term_tau );
                sol.t.push_back( ctx.t_old + term_tau );
                sol.x.push_back( std::move( x_term ) );
                if constexpr ( Dense ) sol.dense.push_back( std::move( r.dense ) );
                break;
            }

            sol.t.push_back( t );
            sol.x.push_back( x );
            if constexpr ( Dense ) sol.dense.push_back( std::move( r.dense ) );

            if constexpr ( concepts::AdaptiveStepper< Stepper > ) h = r.h_next;

            if ( cfg_.max_step > T{ 0 } ) h = std::min( h, cfg_.max_step );
            h = std::min( h, tmax - t );
            break;
        }
    }

    return sol;
}

}  // namespace tax::ode
