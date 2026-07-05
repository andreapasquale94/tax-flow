// include/tax/domain/model.hpp
//
// Bridge between the tax::domain set primitives and tax::model Taylor models
// (requires the tax core's tax::model module, tax PR "feat(model)").
//
//   createModel<P>(domain, x0)   — identity Taylor-model state seeded by a
//                                  Box / Zonotope / PolynomialZonotope: the
//                                  state vector whose component i spans the
//                                  domain's factor axis i over ξ ∈ [-1, 1]^M,
//                                  with zero remainder (mirrors create()).
//   intervalHull(tm_state[, comps])   — rigorous axis-aligned Box enclosure:
//                                  per component the model's bound()
//                                  (polynomial range bound + remainder).
//   zonotopeEnclosure(tm_state, comps) — ImageZonotope over-approximation:
//                                  the even-exponent shift on the polynomial
//                                  part plus one axis-aligned generator per
//                                  component covering the remainder interval.
//   zonotopeFrame(tm_state[, comps])  — the EXACT degree-1 part (remainder
//                                  and curvature excluded — a frame, not an
//                                  enclosure; parity with the TE overload).
//   toPolynomialZonotope(tm_state)    — polynomial parts as a PZ. Drops the
//                                  remainders: exact only when every
//                                  remainder is [0, 0] (asserted in debug);
//                                  use intervalHull/zonotopeEnclosure for
//                                  rigorous set output.
//
// Convention: all models produced here live over the normalized factor cube —
// expansion point 0^M, domain [-1, 1]^M — so the ADS split substitution
// ξ_dim → ±0.5 + 0.5·ξ'_dim applies to the polynomial part unchanged and the
// remainder stays valid on the child (a subset of the parent's domain).

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <tax/core/multi_index.hpp>
#include <tax/domain/box.hpp>
#include <tax/domain/create.hpp>
#include <tax/domain/enclosure.hpp>
#include <tax/domain/polynomial_zonotope.hpp>
#include <tax/domain/zonotope.hpp>
#include <tax/la/types.hpp>
#include <tax/model.hpp>

namespace tax::domain
{

namespace detail
{
// Shared normalized expansion parameter: x0 = 0^M, dom = [-1, 1]^M.
template < class T, int P, int M >
struct NormalizedParam
{
    using TM = tax::model::TaylorModel< T, P, M >;
    typename TM::Point x0{};
    typename TM::Domain dom{};
    constexpr NormalizedParam()
    {
        for ( std::size_t j = 0; j < std::size_t( M ); ++j )
            dom[j] = tax::model::Interval< T >{ T{ -1 }, T{ 1 } };
    }
};
}  // namespace detail

// createModel<P>(box, x0): identity Taylor-model state on a Box (component i
// = x0(i) + halfWidth_i·ξ_i for i < M, constant x0(i) beyond), remainder 0.
template < int P, class T, int M, int D >
[[nodiscard]] Eigen::Matrix< tax::model::TaylorModel< T, P, M >, D, 1 > createModel(
    const Box< T, M >& box, const Eigen::Matrix< T, D, 1 >& x0 )
{
    static_assert( D == Eigen::Dynamic || D >= M,
                   "tax::domain::createModel(): state dimension D must be >= M" );
    using TM = tax::model::TaylorModel< T, P, M >;
    const detail::NormalizedParam< T, P, M > par;
    Eigen::Matrix< TM, D, 1 > out;
    if constexpr ( D == Eigen::Dynamic ) out.resize( x0.size() );
    for ( Eigen::Index i = 0; i < x0.size(); ++i )
    {
        typename TM::Poly poly{};
        poly[0] = x0( i );
        if ( i < M )
        {
            assert( detail::centerMatches( box.center( i ), x0( i ) ) &&
                    "tax::domain::createModel(): x0.head(M) must equal the domain center" );
            tax::MultiIndex< M > e{};
            e[std::size_t( i )] = 1;
            poly[TM::scheme::flatOf( e )] = box.halfWidth( i );
        }
        out( i ) = TM{ std::move( poly ), tax::model::Interval< T >{}, par.x0, par.dom };
    }
    return out;
}

// createModel<P>(zono, x0): identity Taylor-model state on a Zonotope
// (component i < M gets generator row i), remainder 0.
template < int P, class T, int M, int D >
[[nodiscard]] Eigen::Matrix< tax::model::TaylorModel< T, P, M >, D, 1 > createModel(
    const Zonotope< T, M >& zono, const Eigen::Matrix< T, D, 1 >& x0 )
{
    static_assert( D == Eigen::Dynamic || D >= M,
                   "tax::domain::createModel(): state dimension D must be >= M" );
    using TM = tax::model::TaylorModel< T, P, M >;
    const detail::NormalizedParam< T, P, M > par;
    Eigen::Matrix< TM, D, 1 > out;
    if constexpr ( D == Eigen::Dynamic ) out.resize( x0.size() );
    for ( Eigen::Index i = 0; i < x0.size(); ++i )
    {
        typename TM::Poly poly{};
        poly[0] = x0( i );
        if ( i < M )
        {
            assert( detail::centerMatches( zono.center( i ), x0( i ) ) &&
                    "tax::domain::createModel(): x0.head(M) must equal the domain center" );
            for ( int j = 0; j < M; ++j )
            {
                const T g = zono.generators( static_cast< Eigen::Index >( i ), j );
                if ( g == T{ 0 } ) continue;
                tax::MultiIndex< M > e{};
                e[std::size_t( j )] = 1;
                poly[TM::scheme::flatOf( e )] = g;
            }
        }
        out( i ) = TM{ std::move( poly ), tax::model::Interval< T >{}, par.x0, par.dom };
    }
    return out;
}

// createModel<P>(polyZono, x0): Taylor-model state on a PolynomialZonotope —
// component i < M carries the PZ's polynomial generators (constant term
// overwritten by the authoritative x0(i), which must agree with the PZ's own
// center; see create.hpp), remainder 0. P and M are fixed by the PZ itself.
template < int P, class T, int M, class Storage, int D >
[[nodiscard]] Eigen::Matrix< tax::model::TaylorModel< T, P, M >, D, 1 > createModel(
    const PolynomialZonotope< T, P, M, Storage >& z, const Eigen::Matrix< T, D, 1 >& x0 )
{
    static_assert( std::is_same_v< Storage, tax::storage::Dense >,
                   "tax::domain::createModel(): TaylorModel polynomials are dense — the "
                   "PolynomialZonotope must use dense storage" );
    static_assert( D == Eigen::Dynamic || D >= M,
                   "tax::domain::createModel(): state dimension D must be >= M" );
    using TM = tax::model::TaylorModel< T, P, M >;
    const detail::NormalizedParam< T, P, M > par;
    Eigen::Matrix< TM, D, 1 > out;
    if constexpr ( D == Eigen::Dynamic ) out.resize( x0.size() );
    for ( Eigen::Index i = 0; i < x0.size(); ++i )
    {
        typename TM::Poly poly{};
        if ( i < M )
        {
            assert( detail::centerMatches( z.center( static_cast< int >( i ) ), x0( i ) ) &&
                    "tax::domain::createModel(): x0.head(M) must equal the domain center" );
            poly = z.value[std::size_t( i )];
        }
        poly[0] = x0( i );
        out( i ) = TM{ std::move( poly ), tax::model::Interval< T >{}, par.x0, par.dom };
    }
    return out;
}

// ---------------------------------------------------------------------------
// Enclosure layer for Taylor-model states (parity with enclosure.hpp)
// ---------------------------------------------------------------------------

// Rigorous axis-aligned interval hull of the image of selected components:
// per component the model's total enclosure bound() = B(P) + I.
template < std::size_t K, class T, int N, int M, int D >
[[nodiscard]] Box< T, static_cast< int >( K ) > intervalHull(
    const Eigen::Matrix< tax::model::TaylorModel< T, N, M >, D, 1 >& x,
    const std::array< int, K >& comps )
{
    Box< T, static_cast< int >( K ) > out;
    for ( std::size_t i = 0; i < K; ++i )
    {
        const tax::model::Interval< T > b = x( comps[i] ).bound();
        const T m = b.mid();
        // Outward-rounded radius: mid() rounds to nearest, so bump one ulp.
        const T r = std::nextafter( std::max( b.upper() - m, m - b.lower() ),
                                    std::numeric_limits< T >::infinity() );
        out.center( static_cast< Eigen::Index >( i ) ) = m;
        out.halfWidth( static_cast< Eigen::Index >( i ) ) = r;
    }
    return out;
}

// Convenience: rigorous interval hull over ALL D components.
template < class T, int N, int M, int D >
[[nodiscard]] Box< T, D > intervalHull(
    const Eigen::Matrix< tax::model::TaylorModel< T, N, M >, D, 1 >& x )
    requires( D != Eigen::Dynamic )
{
    std::array< int, static_cast< std::size_t >( D ) > comps{};
    for ( int i = 0; i < D; ++i ) comps[std::size_t( i )] = i;
    return intervalHull( x, comps );
}

// Zonotope over-approximation of the image of selected components: the
// even-exponent shift applied to the polynomial parts (exact-coefficient
// enclosure, as the TE overload) plus, per component with a nonzero
// remainder, one axis-aligned generator covering the remainder interval.
template < std::size_t K, class T, int N, int M, int D >
[[nodiscard]] ImageZonotope< T, static_cast< int >( K ) > zonotopeEnclosure(
    const Eigen::Matrix< tax::model::TaylorModel< T, N, M >, D, 1 >& x,
    const std::array< int, K >& comps )
{
    using TM = tax::model::TaylorModel< T, N, M >;
    constexpr std::size_t Ncoef = TM::nCoefficients;
    ImageZonotope< T, static_cast< int >( K ) > out;
    out.generators.resize( static_cast< Eigen::Index >( K ), 0 );
    for ( std::size_t i = 0; i < K; ++i )
        out.center( static_cast< Eigen::Index >( i ) ) = x( comps[i] ).polynomial()[0];

    for ( std::size_t k = 1; k < Ncoef; ++k )
    {
        tax::la::VecNT< static_cast< int >( K ), T > g;
        bool nonzero = false;
        for ( std::size_t i = 0; i < K; ++i )
        {
            g( static_cast< Eigen::Index >( i ) ) = x( comps[i] ).polynomial()[k];
            nonzero = nonzero || g( static_cast< Eigen::Index >( i ) ) != T{ 0 };
        }
        if ( !nonzero ) continue;

        const auto alpha = TM::scheme::multiOf( k );
        bool allEven = true;
        for ( int j = 0; j < M; ++j )
            if ( alpha[std::size_t( j )] % 2 != 0 ) allEven = false;

        // ξ^α ∈ [0,1] for all-even α (center ½, radius ½); ⊆ [-1,1] otherwise.
        if ( allEven )
        {
            out.center += T{ 0.5 } * g;
            g *= T{ 0.5 };
        }
        const Eigen::Index col = out.generators.cols();
        out.generators.conservativeResize( Eigen::NoChange, col + 1 );
        out.generators.col( col ) = g;
    }

    // Remainder intervals: shift the center by mid and add an axis-aligned
    // generator of the (outward-rounded) radius per affected component.
    for ( std::size_t i = 0; i < K; ++i )
    {
        const tax::model::Interval< T > rem = x( comps[i] ).remainder();
        if ( rem.lower() == T{ 0 } && rem.upper() == T{ 0 } ) continue;
        const T m = rem.mid();
        const T r = std::nextafter( std::max( rem.upper() - m, m - rem.lower() ),
                                    std::numeric_limits< T >::infinity() );
        out.center( static_cast< Eigen::Index >( i ) ) += m;
        const Eigen::Index col = out.generators.cols();
        out.generators.conservativeResize( Eigen::NoChange, col + 1 );
        out.generators.col( col ).setZero();
        out.generators( static_cast< Eigen::Index >( i ), col ) = r;
    }
    return out;
}

// The exact degree-1 frame of the polynomial parts (remainder and curvature
// excluded — a frame, not an enclosure). Parity with the TE overload.
template < class T, int N, int M, int D >
[[nodiscard]] Zonotope< T, M > zonotopeFrame(
    const Eigen::Matrix< tax::model::TaylorModel< T, N, M >, D, 1 >& x,
    const std::array< int, static_cast< std::size_t >( M ) >& comps )
{
    Zonotope< T, M > z;
    for ( std::size_t i = 0; i < static_cast< std::size_t >( M ); ++i )
    {
        const auto& f = x( comps[i] ).polynomial();
        z.center( static_cast< Eigen::Index >( i ) ) = f[0];
        for ( int j = 0; j < M; ++j )
        {
            tax::MultiIndex< M > e{};
            e[std::size_t( j )] = 1;
            z.generators( static_cast< Eigen::Index >( i ), j ) = f.coeff( e );
        }
    }
    return z;
}

// Convenience: frame over the first M components.
template < class T, int N, int M, int D >
[[nodiscard]] Zonotope< T, M > zonotopeFrame(
    const Eigen::Matrix< tax::model::TaylorModel< T, N, M >, D, 1 >& x )
{
    std::array< int, static_cast< std::size_t >( M ) > comps{};
    for ( int i = 0; i < M; ++i ) comps[std::size_t( i )] = i;
    return zonotopeFrame( x, comps );
}

// Polynomial parts of the first M components as a PolynomialZonotope. The
// remainders are DROPPED — the result is the represented set only when every
// remainder is [0, 0] (debug-asserted); otherwise it under-represents by up
// to the remainder radii. Use intervalHull / zonotopeEnclosure for rigorous
// set output.
template < class T, int N, int M, int D >
[[nodiscard]] PolynomialZonotope< T, N, M > toPolynomialZonotope(
    const Eigen::Matrix< tax::model::TaylorModel< T, N, M >, D, 1 >& x )
{
    static_assert( D == Eigen::Dynamic || D >= M,
                   "tax::domain::toPolynomialZonotope(): need at least M components" );
    PolynomialZonotope< T, N, M > z;
    for ( int i = 0; i < M; ++i )
    {
        assert( x( i ).remainder().lower() == T{ 0 } && x( i ).remainder().upper() == T{ 0 } &&
                "tax::domain::toPolynomialZonotope(): dropping a nonzero remainder" );
        z.value[std::size_t( i )] = x( i ).polynomial();
    }
    return z;
}

}  // namespace tax::domain
