// include/tax/sgp4/detail/scalar.hpp
//
// Scalar-genericity helpers that let the SGP4 routines run on either a plain
// floating-point type or a tax::TaylorExpansion.
//
//   cst(x)   — the constant (value) part of x as a double.  Every control-flow
//              branch and convergence test in SGP4 keys off this, never off the
//              expansion itself, so the reference trajectory drives the logic
//              and the polynomial parts ride along.
//   dabs(x)  — |x| as the same scalar type, with the sign chosen from cst(x).
//              Preserves derivatives (unlike a branch that returns a constant);
//              used where SGP4 takes fabs() of a *value*, not in a comparison.
//   mod(x,d) — Vallado's floor-based modulo, n - floor(n/d)*d (range [0,d) for
//              d>0).  Reduces only the constant part of an expansion: fmod is
//              locally a constant shift (derivative 1), and every angle reduced
//              this way feeds a periodic trig function, so the result is exact.
//
// The math functions sin/cos/sqrt/atan2/pow are reached via ADL: a `using
// std::foo;` in each routine picks std::foo for doubles and the tax:: overload
// for a TaylorExpansion argument (TaylorExpansion lives in namespace tax).

#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <type_traits>

namespace tax::sgp4
{

/// pi to SGP4's working precision (matches Vallado's literal).
inline constexpr double pi = 3.14159265358979323846;
inline constexpr double twopi = 2.0 * pi;
inline constexpr double deg2rad = pi / 180.0;

namespace detail
{

/// A truncated-expansion scalar: exposes value() and indexed coefficients.
/// Plain floating-point types deliberately do not satisfy this.
template < class T >
concept Expansion = requires( T x, std::size_t k ) {
    { x.value() };
    { x[k] };
};

/// Constant (value) part as a double.
template < std::floating_point T >
[[nodiscard]] constexpr double cst( T x ) noexcept
{
    return static_cast< double >( x );
}
template < Expansion T >
[[nodiscard]] constexpr double cst( const T& x ) noexcept
{
    return static_cast< double >( x.value() );
}

/// |x| with sign taken from the constant part (derivative-preserving for
/// expansions; identical to std::fabs on the value).
template < std::floating_point T >
[[nodiscard]] constexpr T dabs( T x ) noexcept
{
    return std::abs( x );
}
template < Expansion T >
[[nodiscard]] constexpr T dabs( const T& x ) noexcept
{
    return cst( x ) < 0.0 ? -x : x;
}

/// Vallado-style modulo: n - floor(n/d)*d.  Reduces the constant part only.
template < std::floating_point T >
[[nodiscard]] inline T mod( T numer, double denom ) noexcept
{
    return static_cast< T >( numer - std::floor( numer / denom ) * denom );
}
template < Expansion T >
[[nodiscard]] inline T mod( const T& numer, double denom ) noexcept
{
    return numer - std::floor( cst( numer ) / denom ) * denom;
}

}  // namespace detail
}  // namespace tax::sgp4
