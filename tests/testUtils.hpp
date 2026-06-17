#pragma once
#include <gtest/gtest.h>
#include <tax/tax.hpp>

namespace tax::test
{

constexpr double kTol = 1e-10;

/**
 * @brief Check that two TaylorExpansion objects agree coefficient-wise to `tol`.
 *
 * Uses `EXPECT_NEAR` for each flat coefficient, reporting the flat index on
 * failure. `actual.nCoefficients` must equal `expected.nCoefficients` (always
 * true when both have the same static type).
 */
template < typename TE >
inline void ExpectCoeffsNear( const TE& actual, const TE& expected, double tol = kTol )
{
    ASSERT_EQ( actual.nCoefficients, expected.nCoefficients );
    for ( std::size_t k = 0; k < actual.nCoefficients; ++k )
    {
        EXPECT_NEAR( actual[k], expected[k], tol )
            << "Coefficient mismatch at flat index " << k;
    }
}

}  // namespace tax::test
