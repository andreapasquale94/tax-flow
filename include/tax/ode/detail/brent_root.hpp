// include/tax/ode/detail/brent_root.hpp
//
// Brent's method on a bracketed sign change. Used as the universal
// fall-back when a polynomial form of g is not available (Verner /
// Fehlberg / Feagin in Plan B; Taylor when the user-supplied g is
// non-generic).

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace tax::ode::detail
{

// Brent's method on g over [a, b] with the precondition g(a)*g(b) <= 0.
// Returns the located root or std::nullopt if the safeguard exhausts.
template < class T, class GFn >
[[nodiscard]] std::optional< T > brent_root( GFn g, T a, T b, T fa, T fb, int max_iter = 80 )
{
    using std::abs;
    using std::max;
    const T eps = std::numeric_limits< T >::epsilon() * T{ 16 };
    const T zero = T{ 0 };

    if ( fa * fb > zero ) return std::nullopt;

    T c = a, fc = fa;
    bool mflag = true;
    T d = a;

    for ( int it = 0; it < max_iter; ++it )
    {
        if ( abs( fa ) < abs( fb ) )
        {
            std::swap( a, b );
            std::swap( fa, fb );
        }
        if ( abs( b - a ) < eps * ( T{ 1 } + abs( b ) ) ) return ( a + b ) * T{ 0.5 };

        T s;
        if ( fa != fc && fb != fc )
        {
            // Inverse quadratic interpolation.
            s = a * fb * fc / ( ( fa - fb ) * ( fa - fc ) ) +
                b * fa * fc / ( ( fb - fa ) * ( fb - fc ) ) +
                c * fa * fb / ( ( fc - fa ) * ( fc - fb ) );
        } else
        {
            // Secant.
            s = b - fb * ( b - a ) / ( fb - fa );
        }

        const T bound_lo = ( T{ 3 } * a + b ) / T{ 4 };
        const T bound_hi = b;
        const T s_lo = std::min( bound_lo, bound_hi );
        const T s_hi = std::max( bound_lo, bound_hi );
        const bool bad = !( s_lo <= s && s <= s_hi ) ||
                         ( mflag && abs( s - b ) >= abs( b - c ) / T{ 2 } ) ||
                         ( !mflag && abs( s - b ) >= abs( c - d ) / T{ 2 } );
        if ( bad )
        {
            // Bisection.
            s = ( a + b ) * T{ 0.5 };
            mflag = true;
        } else
        {
            mflag = false;
        }

        const T fs = g( s );
        d = c;
        c = b;
        fc = fb;
        if ( fa * fs < zero )
        {
            b = s;
            fb = fs;
        } else
        {
            a = s;
            fa = fs;
        }
    }
    return std::nullopt;
}

}  // namespace tax::ode::detail
