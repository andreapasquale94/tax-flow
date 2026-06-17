// include/tax/ode/config.hpp
//
// Stage 2a ODE integrator configuration.

#pragma once

namespace tax::ode
{

/**
 * @brief Runtime configuration for the Stage 2a `Integrator`.
 *
 * Defaults are conservative for double-precision ODEs. Field
 * conventions:
 *   - `*_step` values of 0 ⇒ the stepper picks a reasonable default
 *     (initial_step: heuristic from RHS magnitude; min_step:
 *     ~eps × (tmax - t0); max_step: tmax - t0).
 */
template < class T = double >
struct IntegratorConfig
{
    T   abstol               = T{ 1e-12 };
    T   reltol               = T{ 1e-12 };
    T   initial_step         = T{ 0 };
    T   min_step             = T{ 0 };
    T   max_step             = T{ 0 };
    int max_steps            = 100'000;
    int max_rejects_per_step = 16;
};

}  // namespace tax::ode
