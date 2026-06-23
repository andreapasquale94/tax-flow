// include/tax/ode/detail/dormand_prince_tableaus.hpp
//
// Butcher tableau for the Dormand–Prince RK 5(4) pair (Dormand & Prince
// 1980) — the classical "RK45" used by MATLAB `ode45` and SciPy `RK45`:
//   DormandPrince45Tab — 7-stage, propagates at order 5 with an order-4
//                        embedded estimator. FSAL in the original (the
//                        last stage equals the next step's first); the
//                        shared stepper re-evaluates every stage, so the
//                        flag is left false and all 7 stages are stored.
//
// Coefficients are the exact textbook rationals, written as constexpr
// fractions (IEEE division is correctly rounded, so each is the nearest
// double to the rational).
//
// Layout matches the tax::ode::detail::adaptive_rk_step contract:
//   c     : nodes  (size n_stages)
//   a     : lower-triangular row-major  (size n_stages*(n_stages-1)/2)
//   b     : main weights (size n_stages)
//   b_emb : embedded weights (size n_stages)

#pragma once

#include <array>

namespace tax::ode::detail
{

struct DormandPrince45Tab
{
    static constexpr int n_stages = 7;
    static constexpr int order = 5;
    static constexpr int order_emb = 4;
    static constexpr bool fsal = false;

    static constexpr std::array< double, 7 > c{ 0.0,       1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0,
                                                8.0 / 9.0, 1.0,       1.0 };

    // 21 values, row-major lower-triangular (without diagonal):
    //   index 0          = a21             (stage 2, depends on stage 1 only)
    //   indices 1..2     = a31, a32        (stage 3)
    //   ...
    //   indices 15..20   = a71 .. a76      (stage 7)
    static constexpr std::array< double, 21 > a{
        1.0 / 5.0,  // stage 2
        3.0 / 40.0,
        9.0 / 40.0,  // stage 3
        44.0 / 45.0,       -56.0 / 15.0,
        32.0 / 9.0,  // stage 4
        19372.0 / 6561.0,  -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,  // stage 5
        9017.0 / 3168.0,   -355.0 / 33.0,
        46732.0 / 5247.0,  49.0 / 176.0,
        -5103.0 / 18656.0,  // stage 6
        35.0 / 384.0,      0.0,
        500.0 / 1113.0,    125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0  // stage 7
    };

    // Order-5 propagation weights (stage 2 and stage 7 do not contribute).
    static constexpr std::array< double, 7 > b{
        35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0 };

    // Embedded order-4 weights.
    static constexpr std::array< double, 7 > b_emb{
        5179.0 / 57600.0, 0.0,       7571.0 / 16695.0, 393.0 / 640.0, -92097.0 / 339200.0,
        187.0 / 2100.0,   1.0 / 40.0 };
};

}  // namespace tax::ode::detail
