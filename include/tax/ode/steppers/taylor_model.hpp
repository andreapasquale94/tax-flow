// include/tax/ode/steppers/taylor_model.hpp
//
// TaylorModelStepper<StateT> — VALIDATED integration of dx/dt = f(x, t) with
// Taylor-model states (tax::model::TaylorModel, tax PR "feat(model)"). The
// state is an Eigen vector of order-N, M-variate Taylor models over the
// normalized factor cube ξ ∈ [-1, 1]^M (the tax::domain convention); each
// component encloses the flow map of its initial-condition set with a
// rigorous remainder interval.
//
// One step over [t, t + h]:
//   1. LIFT the M-variate state into M+1 variables, appending the step-time
//      slot τ with expansion point t and domain [t, t + h].
//   2. PICARD-iterate  r ← ic + ∫_t^τ f(r, τ') dτ'  (antiderivation in the τ
//      slot). N + 1 iterations fix the polynomial part exactly (each pass
//      settles one more τ-order).
//   3. VERIFY the enclosure: one more Picard application must map the
//      candidate's remainder set into itself (then, by the Schauder
//      fixed-point argument, the true flow lies inside). If inclusion fails,
//      the candidate remainder is ε-inflated and re-verified a few times;
//      persistent failure rejects the step (h halves) — the standard
//      Makino/Berz verified-integration scheme.
//   4. FIX τ at the step end t + h (exact, no remainder growth) and PROJECT
//      back to the M factor variables.
//
// Step control targets the τ-TRUNCATION of the flow polynomial (the masses of
// the last two τ-orders, exactly like TaylorStepper / Jorba–Zou): that is the
// error stepping can influence. The remainder additionally absorbs the
// ξ-truncation of the DA representation, whose growth rate is INDEPENDENT of
// h (tax::model's antiderivation sweeps the order-N ξ-block into the
// remainder at O(h) per step) — shrinking steps cannot reduce it; splitting
// the domain can, which is precisely the ADS criterion's job. A step is also
// rejected (h halves) whenever the remainder verification fails. The
// integrator's accept/reject loop, events (SplitEvent, GridEvent,
// root-finding via the controller-free re-step) and Solution capture all
// work unchanged.
//
// The RHS is invoked in Taylor-model arithmetic: f(x, t) receives an Eigen
// vector of (M+1)-variate models and the time as an (M+1)-variate model, so
// it must be a generic callable (same convention as the Taylor stepper).
//
// Rigor contract: inherited from tax::model — interval computations are
// outward-rounded; polynomial coefficient round-off (~1 ulp/op) is not swept
// into the remainder.

#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <tax/domain/model.hpp>
#include <tax/la/types.hpp>
#include <tax/model.hpp>
#include <tax/ode/config.hpp>
#include <tax/ode/controllers.hpp>
#include <tax/ode/detail/adaptive_rk_step.hpp>
#include <tax/ode/propagate.hpp>
#include <tax/ode/step_result.hpp>
#include <utility>

namespace tax::ode
{

namespace detail
{

// Re-index a dense polynomial between the M-variate and (M+1)-variate
// isotropic schemes (the extra variable is appended LAST and its exponent is
// zero on every embedded monomial).
template < class PolyOut, class PolyIn, int MIn >
[[nodiscard]] PolyOut embedPoly( const PolyIn& p )
{
    using SchemeIn = typename PolyIn::scheme;
    using SchemeOut = typename PolyOut::scheme;
    PolyOut out{};
    for ( std::size_t k = 0; k < SchemeIn::nCoeff; ++k )
    {
        const auto c = p[k];
        if ( c == decltype( c ){ 0 } ) continue;
        const auto alpha = SchemeIn::multiOf( k );
        tax::MultiIndex< MIn + 1 > beta{};
        for ( int j = 0; j < MIn; ++j ) beta[std::size_t( j )] = alpha[std::size_t( j )];
        out[SchemeOut::flatOf( beta )] = c;
    }
    return out;
}

// Inverse of embedPoly for polynomials that no longer depend on the last
// variable (after fix): drop the τ axis.
template < class PolyOut, class PolyIn, int MOut >
[[nodiscard]] PolyOut projectPoly( const PolyIn& p )
{
    using SchemeIn = typename PolyIn::scheme;
    using SchemeOut = typename PolyOut::scheme;
    PolyOut out{};
    for ( std::size_t k = 0; k < SchemeIn::nCoeff; ++k )
    {
        const auto c = p[k];
        if ( c == decltype( c ){ 0 } ) continue;
        const auto alpha = SchemeIn::multiOf( k );
        if ( alpha[std::size_t( MOut )] != 0 ) continue;  // dead τ monomial (fix left none)
        tax::MultiIndex< MOut > beta{};
        for ( int j = 0; j < MOut; ++j ) beta[std::size_t( j )] = alpha[std::size_t( j )];
        out[SchemeOut::flatOf( beta )] = c;
    }
    return out;
}

}  // namespace detail

template < class StateT, class Controller = controllers::JorbaZou< double > >
struct TaylorModelStepper
{
    using TM = typename StateT::Scalar;  // tax::model::TaylorModel<T, N, M>
    using T = typename TM::scalar_type;
    static constexpr int N = TM::order_v;
    static constexpr int M = TM::vars_v;
    static constexpr int D = StateT::RowsAtCompileTime;  // may be Eigen::Dynamic

    static_assert( N >= 2,
                   "TaylorModelStepper: order N must be at least 2 for meaningful "
                   "adaptive control" );

    using State = StateT;
    using Config = IntegratorConfig< T >;
    using IntervalT = tax::model::Interval< T >;
    using LiftedTM = tax::model::TaylorModel< T, N, M + 1 >;  // slot M = step time τ
    using Lifted = Eigen::Matrix< LiftedTM, D, 1 >;
    using Rhs = std::function< Lifted( const Lifted&, const LiftedTM& ) >;

    static constexpr bool is_adaptive = true;
    static constexpr bool has_step_expansion = false;

    struct StepData
    {
    };

    // Verification-loop knobs (Makino/Berz ε-inflation).
    static constexpr int kMaxInflate = 4;

    // Domain → initial Taylor-model state. Detected by AdsDriver::run via a
    // requires-expression so the classic (TaylorExpansion) path is untouched;
    // forwards to tax::domain::createModel (identity models, zero remainder).
    template < int P, int MM, class DomainArg, class CenterVec >
    [[nodiscard]] static State createState( const DomainArg& d, const CenterVec& x0 )
    {
        return tax::domain::createModel< P >( d, x0 );
    }

    template < class F >
    [[nodiscard]] StepResult< StateT, TaylorModelStepper > step( const F& f, const StateT& x, T t,
                                                                 T h, const Config& cfg )
    {
        using std::pow;

        StepResult< StateT, TaylorModelStepper > r;
        r.h_used = h;
        r.data = {};

        StateT x_new{ x.size() };
        const CoreResult core = stepCore( f, x, t, h, x_new );
        if ( !core.validated )
        {
            // Remainder inclusion could not be verified at this h (the Picard
            // remainder map is not contractive here): reject and back off.
            r.x_new = x;
            r.h_next = h * T{ 0.5 };
            r.err_norm = std::numeric_limits< T >::infinity();
            r.accepted = false;
            return r;
        }

        // τ-truncation indicator from the masses of the last two τ-orders of
        // the flow polynomial — the same two-coefficient control as
        // TaylorStepper / Jorba–Zou. (The ξ-truncation floor swept into the
        // remainder is h-independent; ADS splitting, not stepping, reduces it.)
        T x_norm{ 0 };
        for ( Eigen::Index i = 0; i < x.size(); ++i )
            x_norm = std::max( x_norm, std::abs( x_new( i ).value() ) );
        const T tol = cfg.abstol + cfg.reltol * x_norm;
        const T err_norm =
            core.cN * pow( std::abs( h ), T( N ) ) + core.cNm1 * pow( std::abs( h ), T( N - 1 ) );

        const auto [h_next, accepted] =
            detail::select_taylor_step< N >( controller_, h, core.cN, core.cNm1, err_norm, tol );

        r.x_new = std::move( x_new );
        r.h_next = h_next;
        r.err_norm = err_norm;
        r.accepted = accepted;
        return r;
    }

    // Controller-free re-step to obtain the state at t + tau — used by the
    // integrator's StepEvaluator for event location. Runs the same verified
    // step over [t, t + tau]; if verification fails at this tau the last
    // Picard iterate is still returned (event location is a diagnostic path,
    // and the polynomial part is exact either way).
    template < class F >
    [[nodiscard]] static StateT step( const F& f, const StateT& x, T t, T tau )
    {
        if ( !( tau > T{ 0 } ) ) return x;
        StateT out{ x.size() };
        (void)stepCore( f, x, t, tau, out );
        return out;
    }

   private:
    struct CoreResult
    {
        bool validated = false;
        T cN{ 0 };    // Σ|coeff| over monomials with τ-exponent N  (max across rows)
        T cNm1{ 0 };  // Σ|coeff| over monomials with τ-exponent N-1 (max across rows)
    };

    // Shared verified-step core: lift, Picard-iterate, verify (with
    // ε-inflation), fix at t + h and project. Reports whether the remainder
    // inclusion was verified plus the τ-truncation masses; `out` always holds
    // the fixed/projected result of the last iterate.
    template < class F >
    static CoreResult stepCore( const F& f, const StateT& x, T t, T h, StateT& out )
    {
        using Point = typename LiftedTM::Point;
        using Dom = typename LiftedTM::Domain;
        using PolyM = typename TM::Poly;
        using PolyL = typename LiftedTM::Poly;

        // --- 1. Lift: shared expansion parameter = (factor axes of the input,
        //         appended τ axis over [t, t + h]).
        Point x0p{};
        Dom domp{};
        {
            // Factor axes from the first concrete (domain-carrying) component;
            // abstract constants (Eigen literals) carry none.
            Eigen::Index src = -1;
            for ( Eigen::Index i = 0; i < x.size() && src < 0; ++i )
                if ( !x( i ).isAbstractConstant() ) src = i;
            for ( int j = 0; j < M; ++j )
            {
                x0p[std::size_t( j )] =
                    src >= 0 ? x( src ).expansionPoint()[std::size_t( j )] : T{ 0 };
                domp[std::size_t( j )] =
                    src >= 0 ? x( src ).domain()[std::size_t( j )] : IntervalT{ T{ -1 }, T{ 1 } };
            }
            x0p[std::size_t( M )] = t;
            domp[std::size_t( M )] = IntervalT{ t, t + h };
        }

        Lifted ic{ x.size() };
        for ( Eigen::Index i = 0; i < x.size(); ++i )
            ic( i ) = LiftedTM{ detail::embedPoly< PolyL, PolyM, M >( x( i ).polynomial() ),
                                x( i ).remainder(), x0p, domp };

        // Time as an (M+1)-variate model: t(ξ, τ) = τ (absolute coordinate).
        const LiftedTM tvar = LiftedTM::variable( x0p, domp, M );

        auto picard = [&]( const Lifted& cur ) -> Lifted {
            Lifted fr = f( cur, tvar );
            Lifted next{ x.size() };
            for ( Eigen::Index i = 0; i < x.size(); ++i ) next( i ) = ic( i ) + fr( i ).integ( M );
            return next;
        };

        // --- 2. Polynomial fixed point: each pass settles one more τ-order.
        Lifted cand = ic;
        for ( int k = 0; k <= N; ++k ) cand = picard( cand );

        // --- 3. Verification with ε-inflation: T(cand) ⊆ cand ⇒ the true flow
        //         lies in cand, hence also in T(cand) (which is what we keep).
        bool validated = false;
        Lifted ver{ x.size() };
        for ( int attempt = 0; attempt <= kMaxInflate; ++attempt )
        {
            ver = picard( cand );
            bool included = true;
            for ( Eigen::Index i = 0; i < x.size() && included; ++i )
                included = cand( i ).remainder().contains( ver( i ).remainder() );
            if ( included )
            {
                validated = true;
                break;
            }
            // Inflate: take the (larger) verification iterate and widen it.
            for ( Eigen::Index i = 0; i < x.size(); ++i )
            {
                const IntervalT rem = ver( i ).remainder();
                const T d = T{ 0.1 } * rem.width() + std::numeric_limits< T >::denorm_min();
                cand( i ) = ver( i );
                cand( i ).remainder() += IntervalT{ -d, d };
            }
        }

        // --- 4. τ-truncation masses of the flow polynomial (per-row sums over
        //         the τ-exponent-N / N-1 monomial blocks; max across rows).
        CoreResult res;
        res.validated = validated;
        const Lifted& fin = validated ? ver : cand;
        for ( Eigen::Index i = 0; i < x.size(); ++i )
        {
            T mN{ 0 }, mNm1{ 0 };
            const auto& poly = fin( i ).polynomial();
            for ( std::size_t k = 0; k < LiftedTM::nCoefficients; ++k )
            {
                const T c = poly[k];
                if ( c == T{ 0 } ) continue;
                const int at = LiftedTM::scheme::multiOf( k )[std::size_t( M )];
                if ( at == N )
                    mN += std::abs( c );
                else if ( at == N - 1 )
                    mNm1 += std::abs( c );
            }
            res.cN = std::max( res.cN, mN );
            res.cNm1 = std::max( res.cNm1, mNm1 );
        }

        // --- 5. Fix τ = t + h (exact) and project the dead τ axis away.
        for ( Eigen::Index i = 0; i < x.size(); ++i )
        {
            const LiftedTM fixed = fin( i ).fix( M, t + h );
            typename TM::Point x0m{};
            typename TM::Domain domm{};
            for ( int j = 0; j < M; ++j )
            {
                x0m[std::size_t( j )] = x0p[std::size_t( j )];
                domm[std::size_t( j )] = domp[std::size_t( j )];
            }
            out( i ) = TM{ detail::projectPoly< PolyM, PolyL, M >( fixed.polynomial() ),
                           fixed.remainder(), x0m, domm };
        }
        return res;
    }

    Controller controller_{};
};

namespace detail
{
template < class State >
struct StepperFor< methods::Picard, State >
{
    using type = TaylorModelStepper< State >;
};
}  // namespace detail

}  // namespace tax::ode
