// include/tax/ode/controllers.hpp
//
// Step-size controllers for Stage 2a.
//   I       — classic integral (stateless, robust baseline)
//   PI      — Gustafsson PI (default for RK steppers)
//   H211b   — Söderlind digital filter (smoothed; robust on bumpy problems)
//   JorbaZou — Taylor-method predictor based on the last two polynomial
//              coefficient magnitudes (default for TaylorStepper)

#pragma once

#include <algorithm>
#include <cmath>

namespace tax::ode::controllers
{

namespace detail
{
template < class T >
constexpr T clamp_factor( T raw, T mn, T mx ) noexcept
{
    return std::min( std::max( raw, mn ), mx );
}
}  // namespace detail

// -------- I --------
template < class T = double >
struct I
{
    T safety = T{ 0.9 };
    T min_factor = T{ 0.2 };
    T max_factor = T{ 5.0 };

    [[nodiscard]] T next_step( T h_used, T err_norm, T tol, int p_emb ) const noexcept
    {
        using std::pow;
        const T ratio = ( err_norm > T{ 0 } ) ? ( tol / err_norm ) : T{ 1 };
        const T exp = T{ 1 } / T( p_emb + 1 );
        const T factor =
            detail::clamp_factor< T >( safety * pow( ratio, exp ), min_factor, max_factor );
        return h_used * factor;
    }
};

// -------- PI (Gustafsson) --------
template < class T = double >
struct PI
{
    T safety = T{ 0.9 };
    T alpha = T{ 0.7 };
    T beta = T{ 0.4 };
    T min_factor = T{ 0.2 };
    T max_factor = T{ 5.0 };

    [[nodiscard]] T next_step( T h_used, T err_norm, T tol, int p_emb ) noexcept
    {
        using std::pow;
        const T denom = ( err_norm > T{ 0 } ) ? err_norm : T{ 1 };
        const T inv = ( p_emb > 0 ) ? T{ 1 } / T( p_emb + 1 ) : T{ 1 };

        // Gustafsson PI (Hairer/Wanner II.4): an integral term (tol/err)^(alpha/k)
        // and a proportional term (tol/err_prev)^(-beta/k), so the net exponent on
        // the current error is -alpha/k (k = p_emb + 1). The previous form put err
        // in both denominators, giving -(alpha+beta)/k — ~1.6x too aggressive.
        T raw;
        if ( err_prev_ <= T{ 0 } )
            raw = pow( tol / denom, inv );  // first call: behave as the I-controller
        else
            raw = pow( tol / denom, alpha * inv ) * pow( tol / err_prev_, -beta * inv );

        // On a step that will be REJECTED (err > tol), the proportional term can
        // push the factor above 1 when the previous error was much larger,
        // proposing a retry step BIGGER than the one that just failed. Never
        // grow on a rejection (Hairer/Wanner II.4): cap the raw factor at 1.
        if ( denom > tol ) raw = std::min( raw, T{ 1 } );

        const T factor = detail::clamp_factor< T >( safety * raw, min_factor, max_factor );
        err_prev_ = denom;
        return h_used * factor;
    }

   private:
    T err_prev_ = T{ 0 };  // <= 0 sentinel: "first call"
};

// -------- H211b (Söderlind) --------
// Reference: G. Söderlind, "Digital filters in adaptive time-stepping",
// ACM TOMS 29 (2003). The H211b filter uses b ≈ 4 by default.
template < class T = double >
struct H211b
{
    T safety = T{ 0.9 };
    T b = T{ 4 };
    T min_factor = T{ 0.2 };
    T max_factor = T{ 5.0 };

    [[nodiscard]] T next_step( T h_used, T err_norm, T tol, int p_emb ) noexcept
    {
        using std::pow;
        const T denom = ( err_norm > T{ 0 } ) ? err_norm : T{ 1 };
        const T inv = ( p_emb > 0 ) ? T{ 1 } / T( p_emb + 1 ) : T{ 1 };

        if ( h_prev_ <= T{ 0 } )
        {
            // First call — behave as I-controller.
            const T factor = detail::clamp_factor< T >( safety * pow( tol / denom, inv ),
                                                        min_factor, max_factor );
            err_prev_ = denom;
            h_prev_ = h_used;
            return h_used * factor;
        }

        // H211b filter (Söderlind 2003): beta1 = beta2 = 1/(b*k), alpha2 = 1/b,
        // with k = p_emb + 1 (so inv = 1/k). The previous code used 1/b for the
        // error terms — independent of method order, ~k times too aggressive.
        const T expo = inv / b;  // 1 / (b * k)
        const T t1 = pow( tol / denom, expo );
        const T t2 = pow( tol / err_prev_, expo );
        const T t3 = pow( h_used / h_prev_, T{ -1 } / b );

        const T raw = t1 * t2 * t3;
        const T factor = detail::clamp_factor< T >( safety * raw, min_factor, max_factor );
        err_prev_ = denom;
        h_prev_ = h_used;
        return h_used * factor;
    }

   private:
    T err_prev_ = T{ 1 };
    T h_prev_ = T{ 0 };
};

// -------- JorbaZou (Taylor-specific) --------
// Reference: À. Jorba & M. Zou, "A software package for the numerical
// integration of ODE by means of high-order Taylor methods",
// Experimental Mathematics 14 (2005). Variant that uses the magnitudes
// of the last two polynomial coefficients to predict h_new directly.
template < class T = double >
struct JorbaZou
{
    T safety = T{ 0.9 };
    T min_factor = T{ 0.2 };
    T max_factor = T{ 5.0 };

    [[nodiscard]] T next_step( T h_used, T c_N_norm, T c_Nm1_norm, T tol,
                               int N_order ) const noexcept
    {
        using std::pow;
        // rho1 = (tol / |c_N|)^(1/N)
        // rho2 = (tol / |c_{N-1}|)^(1/(N-1))
        // h_new = safety * min(rho1, rho2)
        const T denom1 = ( c_N_norm > T{ 0 } ) ? c_N_norm : T{ 1 };
        const T denom2 = ( c_Nm1_norm > T{ 0 } ) ? c_Nm1_norm : T{ 1 };
        const T rho1 = pow( tol / denom1, T{ 1 } / T( N_order ) );
        const T rho2 = pow( tol / denom2, T{ 1 } / T( N_order - 1 ) );
        const T raw = safety * std::min( rho1, rho2 );
        // Clamp to a factor of h_used so we don't accept arbitrary jumps.
        const T factor = detail::clamp_factor< T >( raw / h_used, min_factor, max_factor );
        return h_used * factor;
    }
};

// -------- FixedStep --------
// No-op controller: returns the previously-used step size unchanged,
// regardless of error or tolerance. Used to force a user-prescribed
// step grid; steppers also treat this controller as a signal to mark
// every step accepted (so adaptive retry never kicks in).
template < class T = double >
struct FixedStep
{
    [[nodiscard]] T next_step( T h_used, T err_norm, T tol, int p_emb ) const noexcept
    {
        (void)err_norm;
        (void)tol;
        (void)p_emb;
        return h_used;
    }

    // Overload matching JorbaZou's call signature, used by TaylorStepper.
    [[nodiscard]] T next_step( T h_used, T c_N_norm, T c_Nm1_norm, T tol,
                               int N_order ) const noexcept
    {
        (void)c_N_norm;
        (void)c_Nm1_norm;
        (void)tol;
        (void)N_order;
        return h_used;
    }
};

}  // namespace tax::ode::controllers
