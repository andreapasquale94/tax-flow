// include/tax/ode/detail/fehlberg_tableaus.hpp
//
// E. Fehlberg, "Classical Fifth-, Sixth-, Seventh-, and Eighth-Order
// Runge-Kutta Formulas with Stepsize Control", NASA TR R-287, 1968.
// The classical RK 7(8) pair: 13 stages, propagates at order 7,
// embedded order-8 error estimator (the "Fehlberg coincidence" is a
// known limitation where the embedded estimator is zero on certain
// steps; see spec Risks table).
//
// Coefficient values reproduced from Boost.Odeint
//   boost/numeric/odeint/stepper/runge_kutta_fehlberg78.hpp
// Copyright 2011-2013 Mario Mulansky, Copyright 2012-2013 Karsten Ahnert
// Distributed under the Boost Software License, Version 1.0.
// (See https://www.boost.org/LICENSE_1_0.txt)
//
// Layout matches the tax::ode::detail::adaptive_rk_step contract:
//   c     : nodes  (size n_stages)
//   a     : lower-triangular row-major  (size n_stages*(n_stages-1)/2)
//   b     : main weights (size n_stages)  — propagates at order 7
//   b_emb : embedded weights (size n_stages) — order-8 error estimator

#pragma once

#include <array>

namespace tax::ode::detail
{

struct Fehlberg78Tab
{
    static constexpr int n_stages = 13;
    static constexpr int order = 7;
    static constexpr int order_emb = 8;
    static constexpr bool fsal = false;

    // c[0] = 0.0 (first stage at t), then c2..c13 from Boost.Odeint source.
    // Note: c[11] = 0.0, c[12] = 1.0 (stages 12 and 13 share c = 0 and 1).
    static constexpr std::array< double, 13 > c{
        0.0,         // stage 1
        2.0 / 27.0,  // stage 2
        1.0 / 9.0,   // stage 3
        1.0 / 6.0,   // stage 4
        5.0 / 12.0,  // stage 5
        1.0 / 2.0,   // stage 6
        5.0 / 6.0,   // stage 7
        1.0 / 6.0,   // stage 8
        2.0 / 3.0,   // stage 9
        1.0 / 3.0,   // stage 10
        1.0,         // stage 11
        0.0,         // stage 12
        1.0          // stage 13
    };

    // 78 values, row-major lower-triangular (without diagonal):
    //   index 0          = a21             (stage 2, depends on stage 1 only)
    //   indices 1..2     = a31, a32        (stage 3)
    //   indices 3..5     = a41, a42, a43   (stage 4)
    //   ...
    //   indices 66..77   = a13_1 .. a13_12 (stage 13)
    // Zero entries are explicit.
    static constexpr std::array< double, 78 > a{
        // stage 2 (1 entry): a21
        2.0 / 27.0,
        // stage 3 (2 entries): a31, a32
        1.0 / 36.0, 1.0 / 12.0,
        // stage 4 (3 entries): a41, a42, a43
        1.0 / 24.0, 0.0, 1.0 / 8.0,
        // stage 5 (4 entries): a51, a52, a53, a54
        5.0 / 12.0, 0.0, -25.0 / 16.0, 25.0 / 16.0,
        // stage 6 (5 entries)
        1.0 / 20.0, 0.0, 0.0, 1.0 / 4.0, 1.0 / 5.0,
        // stage 7 (6 entries)
        -25.0 / 108.0, 0.0, 0.0, 125.0 / 108.0, -65.0 / 27.0, 125.0 / 54.0,
        // stage 8 (7 entries)
        31.0 / 300.0, 0.0, 0.0, 0.0, 61.0 / 225.0, -2.0 / 9.0, 13.0 / 900.0,
        // stage 9 (8 entries)
        2.0, 0.0, 0.0, -53.0 / 6.0, 704.0 / 45.0, -107.0 / 9.0, 67.0 / 90.0, 3.0,
        // stage 10 (9 entries)
        -91.0 / 108.0, 0.0, 0.0, 23.0 / 108.0, -976.0 / 135.0, 311.0 / 54.0, -19.0 / 60.0,
        17.0 / 6.0, -1.0 / 12.0,
        // stage 11 (10 entries)
        2383.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0, -301.0 / 82.0, 2133.0 / 4100.0,
        45.0 / 82.0, 45.0 / 164.0, 18.0 / 41.0,
        // stage 12 (11 entries)
        3.0 / 205.0, 0.0, 0.0, 0.0, 0.0, -6.0 / 41.0, -3.0 / 205.0, -3.0 / 41.0, 3.0 / 41.0,
        6.0 / 41.0, 0.0,
        // stage 13 (12 entries)
        -1777.0 / 4100.0, 0.0, 0.0, -341.0 / 164.0, 4496.0 / 1025.0, -289.0 / 82.0, 2193.0 / 4100.0,
        51.0 / 82.0, 33.0 / 164.0, 12.0 / 41.0, 0.0, 1.0 };

    // Order-7 propagation weights (Boost b array).
    // Stages 1-5 and 11 do not contribute.
    static constexpr std::array< double, 13 > b{
        0.0,        0.0,         0.0,         0.0, 0.0,          34.0 / 105.0, 9.0 / 35.0,
        9.0 / 35.0, 9.0 / 280.0, 9.0 / 280.0, 0.0, 41.0 / 840.0, 41.0 / 840.0 };

    // Embedded order-8 weights: b_emb = b + db, where db is the Boost
    // rk78_coefficients_db array (the error difference b8 - b7).
    // db[0] = -41/840, db[10] = -41/840, db[11] = +41/840, db[12] = +41/840,
    // all others zero.
    static constexpr std::array< double, 13 > b_emb{
        -41.0 / 840.0, 0.0,          0.0,         0.0,         0.0,
        34.0 / 105.0,  9.0 / 35.0,   9.0 / 35.0,  9.0 / 280.0, 9.0 / 280.0,
        -41.0 / 840.0, 41.0 / 420.0, 41.0 / 420.0 };
};

}  // namespace tax::ode::detail
