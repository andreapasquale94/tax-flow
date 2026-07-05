// include/tax/domain/ellipsoid.hpp
//
// Coverings of an ellipsoid E = { center + L·u : ‖u‖₂ ≤ 1 } (L is any factor
// of the shape matrix, e.g. a Cholesky factor of a covariance) by the domain
// primitives — the geometry behind the flow-aligned oriented-ADS recipe
// (docs/domain/zonotope.md):
//
//   ellipsoidCover(c, L[, R])   — the parallelotope { c + (L·R)·ξ : ξ ∈ cube }
//                                 covers E for ANY orthogonal R, because
//                                 R·cube ⊇ R·ball = ball. R is the free
//                                 orientation of the covering — pick it from
//                                 flowAlignedRotation to align splits with the
//                                 flow's stretching. The covering property
//                                 REQUIRES R orthogonal (asserted in debug
//                                 builds): a non-orthogonal R silently yields
//                                 a parallelotope that does NOT enclose E.
//   ellipsoidIntervalHull(c, L) — the EXACT axis-aligned interval hull of E:
//                                 per axis i the extent is ‖row_i(L)‖₂ (the
//                                 support of the unit ball). Note this is the
//                                 L2 row norm — the hull of the *ellipsoid* —
//                                 whereas Zonotope::intervalHull uses L1 row
//                                 norms — the hull of the *parallelotope*.
//                                 Handing the ellipsoid hull to a box ADS run
//                                 covers E but NOT ellipsoidCover(c, L).

#pragma once

#include <Eigen/Dense>
#include <cassert>
#include <tax/domain/box.hpp>
#include <tax/domain/zonotope.hpp>
#include <tax/la/types.hpp>

namespace tax::domain
{

namespace detail
{
// R·cube ⊇ ball holds iff R is orthogonal — the covering contract of
// ellipsoidCover. Debug-only guard (mirrors create()'s center check).
template < class T, class DerivedR >
[[nodiscard]] inline bool isOrthogonal( const Eigen::MatrixBase< DerivedR >& R,
                                        T tol = T{ 1e-9 } ) noexcept
{
    const auto I =
        Eigen::Matrix< T, DerivedR::RowsAtCompileTime, DerivedR::ColsAtCompileTime >::Identity(
            R.rows(), R.cols() );
    return ( R * R.transpose() - I ).cwiseAbs().maxCoeff() <= tol;
}
}  // namespace detail

template < class T, int M, class DerivedL >
[[nodiscard]] Zonotope< T, M > ellipsoidCover( const tax::la::VecNT< M, T >& center,
                                               const Eigen::MatrixBase< DerivedL >& L )
{
    Zonotope< T, M > z;
    z.center = center;
    z.generators = L;
    return z;
}

template < class T, int M, class DerivedL, class DerivedR >
[[nodiscard]] Zonotope< T, M > ellipsoidCover( const tax::la::VecNT< M, T >& center,
                                               const Eigen::MatrixBase< DerivedL >& L,
                                               const Eigen::MatrixBase< DerivedR >& R )
{
    assert( detail::isOrthogonal< T >( R ) &&
            "tax::domain::ellipsoidCover(): R must be orthogonal — a non-orthogonal R does NOT "
            "cover the ellipsoid" );
    Zonotope< T, M > z;
    z.center = center;
    z.generators = L * R;
    return z;
}

template < class T, int M, class DerivedL >
[[nodiscard]] Box< T, M > ellipsoidIntervalHull( const tax::la::VecNT< M, T >& center,
                                                 const Eigen::MatrixBase< DerivedL >& L )
{
    tax::la::VecNT< M, T > hw;
    for ( int i = 0; i < M; ++i ) hw( i ) = L.row( i ).norm();
    return Box< T, M >{ center, hw };
}

}  // namespace tax::domain
