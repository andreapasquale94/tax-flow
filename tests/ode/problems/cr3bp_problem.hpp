// tests/ode/cr3bp_problem.hpp
//
// Shared planar Circular-Restricted Three-Body Problem (Earth–Moon)
// fixture for the CR3BP propagation test, events test, and benchmark.
//
// Synodic rotating frame with the Earth–Moon barycentre at the origin.
// Earth at (-μ, 0); Moon at (1 - μ, 0).
// State = (x, y, vx, vy).
// Equations of motion (canonical Hamiltonian form):
//   dx/dt  = vx
//   dy/dt  = vy
//   dvx/dt =  2 vy + x - (1-μ)(x+μ)/r1^3 - μ(x-1+μ)/r2^3
//   dvy/dt = -2 vx + y - (1-μ)y/r1^3      - μy/r2^3
// where r1 = √((x+μ)² + y²), r2 = √((x-1+μ)² + y²).
//
// Jacobi constant (conserved):
//   C = 2*Omega(x, y) - vx² - vy²
//   Omega = ½(x² + y²) + (1-μ)/r1 + μ/r2 + ½μ(1-μ)
//
// Lagrange points (numerical roots of dOmega/dx = 0 on y = 0):
//   L1 ≈ 0.8369180073...
//   L2 ≈ 1.1556824692...
//
// IC chosen to produce a trajectory that begins near Earth, transits
// the L1 neck, loops around the Moon, and exits via L2 within
// T_final = 7 non-dim. time units.
//
// Note: the original IC (x=0.836, vy=0.0095) produced a bounded lunar
// orbit that never exits via L2.  It was replaced (Task 22) with
// (x=0.82, vy=0.175), which gives a clean interior-to-exterior transit:
//   L1 crossing at t≈0.47,  two Moon periapses (r2≈0.049–0.067),
//   L2 crossing at t≈4.2.  Min Moon distance ≈ 0.049 (no singularity).

#pragma once

#include <cmath>
#include <tax/la/types.hpp>

namespace tax::ode::test
{

constexpr double kCR3BPMu = 0.01215058560962404;  // Earth–Moon
constexpr double kCR3BPL1 = 0.8369180073407246;
constexpr double kCR3BPL2 = 1.1556824692238923;

using CR3BPState = tax::la::VecNT< 4, double >;

inline auto cr3bp_rhs( double mu = kCR3BPMu )
{
    return [mu]( const auto& s, const auto& /*t*/ ) -> std::decay_t< decltype( s ) > {
        using S = std::decay_t< decltype( s ) >;
        using V = typename S::Scalar;
        using std::sqrt;  // double → std::sqrt; TaylorExpansion → tax::sqrt via ADL

        S out;
        const V x = s( 0 );
        const V y = s( 1 );
        const V vx = s( 2 );
        const V vy = s( 3 );

        const V x1 = x + V( mu );
        const V x2 = x - V( 1.0 - mu );
        const V r1_2 = x1 * x1 + y * y;
        const V r2_2 = x2 * x2 + y * y;
        const V r1_3 = r1_2 * sqrt( r1_2 );
        const V r2_3 = r2_2 * sqrt( r2_2 );

        out( 0 ) = vx;
        out( 1 ) = vy;
        out( 2 ) = V( 2 ) * vy + x - V( 1.0 - mu ) * x1 / r1_3 - V( mu ) * x2 / r2_3;
        out( 3 ) = -V( 2 ) * vx + y - V( 1.0 - mu ) * y / r1_3 - V( mu ) * y / r2_3;
        return out;
    };
}

inline double cr3bp_jacobi( const CR3BPState& s, double mu = kCR3BPMu )
{
    const double x = s( 0 ), y = s( 1 );
    const double vx = s( 2 ), vy = s( 3 );
    const double r1 = std::hypot( x + mu, y );
    const double r2 = std::hypot( x - 1.0 + mu, y );
    const double Omega =
        0.5 * ( x * x + y * y ) + ( 1.0 - mu ) / r1 + mu / r2 + 0.5 * mu * ( 1.0 - mu );
    return 2.0 * Omega - ( vx * vx + vy * vy );
}

inline CR3BPState cr3bp_transit_ic()
{
    // Interior-to-exterior transit: crosses L1, loops Moon twice, exits L2.
    // Adjusted in Task 22 from (0.836, 0, 0, 0.0095) — that IC was bounded.
    CR3BPState x0;
    x0( 0 ) = 0.82;  // inside L1 neck
    x0( 1 ) = 0.0;
    x0( 2 ) = 0.0;
    x0( 3 ) = 0.175;  // velocity normal to x-axis, strong enough to transit
    return x0;
}

constexpr double kCR3BPTFinal = 7.0;

}  // namespace tax::ode::test
