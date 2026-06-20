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
#include <memory>
#include <stdexcept>
#include <tax/la.hpp>
#include <tax/la/types.hpp>
#include <tax/ode/concepts.hpp>
#include <tax/ode/config.hpp>
#include <tax/ode/event.hpp>
#include <tax/ode/events/grid_event.hpp>
#include <tax/ode/events/root_finding_event.hpp>
#include <tax/ode/events/step_event.hpp>
#include <tax/ode/solution.hpp>
#include <tax/ode/step_evaluator.hpp>
#include <utility>
#include <vector>

namespace tax::ode
{

template < concepts::Stepper Stepper, class F >
class Integrator
{
   public:
    using State = typename Stepper::State;
    using T = typename Stepper::T;
    using Config = typename Stepper::Config;
    using Solution = tax::ode::Solution< Stepper, State >;
    using EventPtr = std::shared_ptr< Event< State, T > >;
    using EventList = std::vector< EventPtr >;

    explicit Integrator( F f, Config cfg = {}, EventList events = {} )
        : f_( std::move( f ) ),
          cfg_( std::move( cfg ) ),
          eval_( std::make_shared< detail::StepEvaluatorImpl< Stepper, F > >() )
    {
        if ( !( cfg_.abstol > T{ 0 } ) )
            throw std::invalid_argument( "IntegratorConfig: abstol must be > 0" );
        if ( !( cfg_.reltol > T{ 0 } ) )
            throw std::invalid_argument( "IntegratorConfig: reltol must be > 0" );
        if ( cfg_.max_steps <= 0 )
            throw std::invalid_argument( "IntegratorConfig: max_steps must be > 0" );
        if ( cfg_.max_rejects_per_step <= 0 )
            throw std::invalid_argument( "IntegratorConfig: max_rejects_per_step must be > 0" );
        for ( auto& e : events ) addEvent( std::move( e ) );
    }

    // Register an event; binds it to this integrator's StepEvaluator.
    void addEvent( EventPtr e )
    {
        e->setEvaluator( eval_ );
        events_.push_back( std::move( e ) );
    }

    void addStepEvent( std::string name )
    {
        addEvent( std::make_shared< StepEvent< State, T > >( std::move( name ) ) );
    }

    template < class G >
    void addRootFindingEvent( G g, Direction dir, std::string name, bool terminal = false )
    {
        addEvent( std::make_shared< RootFindingEvent< State, T, G > >(
            std::move( g ), dir, std::move( name ), terminal ) );
    }

    void addGridEvent( std::vector< T > times, std::string name )
    {
        addEvent(
            std::make_shared< GridEvent< State, T > >( std::move( times ), std::move( name ) ) );
    }

    [[nodiscard]] Solution integrate( const State& x0, const T& t0, const T& tmax ) const;

   private:
    F f_;
    Config cfg_;
    std::shared_ptr< detail::StepEvaluatorImpl< Stepper, F > > eval_;
    EventList events_;
};

template < concepts::Stepper Stepper, class F >
typename Integrator< Stepper, F >::Solution Integrator< Stepper, F >::integrate(
    const State& x0, const T& t0, const T& tmax ) const
{
    if ( !( tmax > t0 ) ) throw std::invalid_argument( "Integrator::integrate: tmax must be > t0" );

    Solution sol;
    sol.t.push_back( t0 );
    sol.x.push_back( x0 );

    // save_steps == true : append every accepted boundary.
    // save_steps == false: keep [initial, final] — overwrite the trailing
    //                       "final" slot so only (t0,x0) and the latest state
    //                       remain. (Two entries, collapsing to one initial-only
    //                       entry if zero steps are taken.)
    auto record = [&]( const T& tt, const State& xx ) {
        if ( cfg_.save_steps || sol.t.size() == 1 )
        {
            sol.t.push_back( tt );
            sol.x.push_back( xx );
        } else
        {
            sol.t.back() = tt;
            sol.x.back() = xx;
        }
    };

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

    Recorder< State, T > rec{ &sol.events };

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

        // Pre-step clamp: never advance past tmax or any event's nextStop.
        {
            T cap = tmax - t;
            for ( const auto& e : events_ )
                if ( auto ns = e->nextStop( t ) )
                    if ( *ns - t < cap ) cap = *ns - t;
            if ( cap < h ) h = cap;
        }

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

            // Refresh the shared evaluator with this accepted step, then run
            // events in registration order. Stop at the first Terminate.
            T term_t{};
            State term_x{};
            bool have_term_point = false;
            if ( !events_.empty() )
            {
                eval_->setStep( f_, x, t, r.x_new, r.h_used, r.data );
                for ( const auto& e : events_ )
                {
                    const std::size_t before = rec.count();
                    const auto act = e->onStep( rec );
                    if ( act == Event< State, T >::Action::Terminate )
                    {
                        terminate = true;
                        // Truncate at the terminating event's last record if it
                        // emitted one; otherwise at the step boundary.
                        if ( rec.count() > before )
                        {
                            term_t = sol.events.back().t;
                            term_x = sol.events.back().x;
                            have_term_point = true;
                        }
                        break;
                    }
                }
            }

            t += r.h_used;
            x = r.x_new;

            if ( terminate )
            {
                if ( have_term_point )
                    record( term_t, term_x );
                else
                    record( t, x );
                break;
            }

            record( t, x );

            if constexpr ( concepts::AdaptiveStepper< Stepper > ) h = r.h_next;

            if ( cfg_.max_step > T{ 0 } ) h = std::min( h, cfg_.max_step );
            h = std::min( h, tmax - t );
            break;
        }
    }

    return sol;
}

}  // namespace tax::ode
