// include/tax/ode/step_evaluator.hpp
//
// StepEvaluator<State,T>: the Integrator's per-accepted-step capability
// surface, shared with events that need to evaluate the trajectory inside a
// step (root-finding, mid-step sampling). The RHS type F is erased exactly
// once, in detail::StepEvaluatorImpl::eval.

#pragma once

#include <optional>
#include <tax/la.hpp>
#include <tax/ode/detail/brent_root.hpp>

namespace tax::ode
{

enum class Direction
{
    Increasing,
    Decreasing,
    Any
};

// Half-open crossing detection: s0 strictly on one side of zero, s1 reaching
// or passing zero. Including s1 == 0 catches a root on a step boundary;
// excluding s0 == 0 stops the next step (whose s0 is that zero) re-firing it.
template < class T >
[[nodiscard]] inline bool dir_match( T s0, T s1, Direction d ) noexcept
{
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

template < class State, class T >
class StepEvaluator
{
   public:
    virtual ~StepEvaluator() = default;

    [[nodiscard]] virtual const State& xOld() const noexcept = 0;
    [[nodiscard]] virtual const State& xNew() const noexcept = 0;
    [[nodiscard]] virtual T tOld() const noexcept = 0;
    [[nodiscard]] virtual T hUsed() const noexcept = 0;

    // State at t_old + τ (τ ∈ [0, h_used]) at the method's full order.
    [[nodiscard]] virtual State eval( T tau ) const = 0;

    // First root of g(x(τ), t_old+τ) in (0, h_used] matching dir, or nullopt.
    // (Bracket is [0, h_used]; the τ=0 endpoint is excluded by the strict-sign
    // dir_match, so the returned-root set is the half-open (0, h_used].)
    template < class G >
    [[nodiscard]] std::optional< T > findRoot( G&& g, Direction dir ) const
    {
        const T s0 = T( g( xOld(), tOld() ) );
        const T s1 = T( g( xNew(), tOld() + hUsed() ) );
        if ( !dir_match( s0, s1, dir ) ) return std::nullopt;
        auto sample = [&]( T tau ) -> T { return T( g( eval( tau ), tOld() + tau ) ); };
        return detail::brent_root< T >( sample, T{ 0 }, hUsed(), s0, s1 );
    }
};

namespace detail
{

// Concrete evaluator owned by the Integrator. Holds non-owning pointers into
// the integrator's accepted-step frame; setStep refreshes them each step (no
// allocation). eval is the single F-erasure point.
template < class Stepper, class F >
class StepEvaluatorImpl final : public StepEvaluator< typename Stepper::State, typename Stepper::T >
{
   public:
    using State = typename Stepper::State;
    using T = typename Stepper::T;
    using StepData = typename Stepper::StepData;

    void setStep( const F& f, const State& x_old, T t_old, const State& x_new, T h_used,
                  const StepData& data )
    {
        f_ = &f;
        x_old_ = &x_old;
        t_old_ = t_old;
        x_new_ = &x_new;
        h_used_ = h_used;
        data_ = &data;
    }

    [[nodiscard]] const State& xOld() const noexcept override { return *x_old_; }
    [[nodiscard]] const State& xNew() const noexcept override { return *x_new_; }
    [[nodiscard]] T tOld() const noexcept override { return t_old_; }
    [[nodiscard]] T hUsed() const noexcept override { return h_used_; }

    [[nodiscard]] State eval( T tau ) const override
    {
        if constexpr ( Stepper::has_step_expansion )
            return tax::la::eval( *data_, tau );
        else
            return Stepper::step( *f_, *x_old_, t_old_, tau );
    }

   private:
    const F* f_ = nullptr;
    const State* x_old_ = nullptr;
    const State* x_new_ = nullptr;
    T t_old_{};
    T h_used_{};
    const StepData* data_ = nullptr;
};

}  // namespace detail

}  // namespace tax::ode
