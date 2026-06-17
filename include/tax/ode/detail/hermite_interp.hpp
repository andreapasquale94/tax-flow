// include/tax/ode/detail/hermite_interp.hpp
//
// Cubic-Hermite interpolation between two state samples and their
// derivatives. Reproduces (x0, x1) exactly at the boundaries and is
// C^1 across the step. Used as the dense-output fallback by RK
// steppers without a published built-in continuous extension.
//
// State arithmetic is routed through VectorOps<State> so the same
// implementation works for double-state and DA-vector-state without
// requiring Eigen's ScalarBinaryOpTraits to know about tax::TEn.

#pragma once

#include <tax/la/types.hpp>

#include <tax/ode/vector_ops.hpp>

namespace tax::ode::detail
{

template < class State, class T >
[[nodiscard]] State hermite_interp(
    const State& x0, const State& x1,
    const State& f0, const State& f1,
    const T& h_step, const T& tau )
{
    using Ops = VectorOps< State >;

    const T theta = tau / h_step;
    const T om    = T{ 1 } - theta;

    const double h00 = static_cast< double >( ( T{ 1 } + T{ 2 } * theta ) * om * om );
    const double h10 = static_cast< double >( theta * om * om );
    const double h01 = static_cast< double >( theta * theta * ( T{ 3 } - T{ 2 } * theta ) );
    const double h11 = static_cast< double >( -theta * theta * om );
    const double hd  = static_cast< double >( h_step );

    State out;
    Ops::scale_assign( out, h00,       x0 );
    Ops::axpy        ( out, h10 * hd,  f0 );
    Ops::axpy        ( out, h01,       x1 );
    Ops::axpy        ( out, h11 * hd,  f1 );
    return out;
}

}  // namespace tax::ode::detail
