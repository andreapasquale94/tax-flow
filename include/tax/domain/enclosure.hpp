// include/tax/domain/enclosure.hpp
//
// Enclosures of the image of the factor cube under a DA state — the layer
// that turns a propagated flow map (an ADS leaf payload) into a consumable
// set: a DA state x over ξ ∈ [-1,1]^M IS a polynomial zonotope, so its image
// can be bounded without any new representation.
//
//   intervalHull(x[, comps])   — axis-aligned Box bound per component:
//                                 c_i ± Σ_{α≠0} |coeff_i(α)|.
//   zonotopeEnclosure(x, comps) — ImageZonotope over-approximation via the
//                                 even-exponent shift (Kochdumper & Althoff
//                                 2020, zonotope enclosure of a sparse
//                                 polynomial zonotope): a monomial ξ^α with
//                                 all-even α ranges over [0,1] on the cube,
//                                 so its coefficient contributes ½·coeff to
//                                 the center and ½·coeff as a generator; any
//                                 other monomial ranges over [-1,1] and
//                                 contributes its coefficient as a generator.
//                                 Always at least as tight as intervalHull.
//   zonotopeFrame(x, comps)     — the EXACT degree-1 part as a Zonotope
//                                 (center = constant terms, G = linear
//                                 coefficients); an enclosure only where the
//                                 map is linear, a frame otherwise.
//
// `comps` selects which state components span the enclosure (e.g. position
// coordinates only); K is its size.
//
// Rigor: every construction here is a mathematical enclosure in EXACT
// coefficient arithmetic (each monomial's range over the cube is covered).
// Floating-point round-off of the coefficient sums (~1 ulp per addition) is
// NOT outward-rounded — the same contract as the rest of the TE pipeline.
// For guaranteed bounds use the tax::model Taylor-model layer when available.

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <cstddef>
#include <tax/core/multi_index.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/domain/box.hpp>
#include <tax/domain/zonotope.hpp>
#include <tax/la/types.hpp>

namespace tax::domain
{

// A zonotope with an arbitrary number of generators (columns) — the result
// type of zonotopeEnclosure. Unlike the square-G Zonotope domain it is not a
// Domain (no split/localize); it is a value describing an over-approximation.
template < class T, int K >
struct ImageZonotope
{
    tax::la::VecNT< K, T > center = tax::la::VecNT< K, T >::Zero();
    Eigen::Matrix< T, K, Eigen::Dynamic > generators;

    // Support function ρ(d) = d·c + Σ_j |d·g_j|: the signed extent of the
    // zonotope along direction d. p is inside iff d·p <= ρ(d) for all d.
    template < class Derived >
    [[nodiscard]] T support( const Eigen::MatrixBase< Derived >& d ) const
    {
        T s = d.dot( center );
        for ( Eigen::Index j = 0; j < generators.cols(); ++j )
            s += std::abs( d.dot( generators.col( j ) ) );
        return s;
    }

    // Tightest axis-aligned Box covering the zonotope (L1 row norms).
    [[nodiscard]] Box< T, K > intervalHull() const noexcept
    {
        tax::la::VecNT< K, T > hw;
        for ( int i = 0; i < K; ++i ) hw( i ) = generators.row( i ).cwiseAbs().sum();
        return Box< T, K >{ center, hw };
    }
};

// Axis-aligned interval hull of the image of selected components.
template < std::size_t K, class T, int N, int M, class Storage, int D >
[[nodiscard]] Box< T, static_cast< int >( K ) > intervalHull(
    const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 >&
        x,
    const std::array< int, K >& comps ) noexcept
{
    constexpr std::size_t Ncoef = tax::numMonomials( N, M );
    Box< T, static_cast< int >( K ) > out;
    for ( std::size_t i = 0; i < K; ++i )
    {
        const auto& f = x( comps[i] );
        T r{ 0 };
        for ( std::size_t k = 1; k < Ncoef; ++k ) r += std::abs( f[k] );
        out.center( static_cast< Eigen::Index >( i ) ) = f[0];
        out.halfWidth( static_cast< Eigen::Index >( i ) ) = r;
    }
    return out;
}

// Convenience: interval hull over ALL D components.
template < class T, int N, int M, class Storage, int D >
[[nodiscard]] Box< T, D > intervalHull(
    const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 >&
        x ) noexcept
    requires( D != Eigen::Dynamic )
{
    std::array< int, static_cast< std::size_t >( D ) > comps{};
    for ( int i = 0; i < D; ++i ) comps[static_cast< std::size_t >( i )] = i;
    return intervalHull( x, comps );
}

// Zonotope over-approximation of the image of selected components via the
// even-exponent shift (see file header). Zero-coefficient monomials produce
// no generator, so the generator count is the number of distinct monomials
// with a nonzero coefficient in at least one selected component.
template < std::size_t K, class T, int N, int M, class Storage, int D >
[[nodiscard]] ImageZonotope< T, static_cast< int >( K ) > zonotopeEnclosure(
    const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 >&
        x,
    const std::array< int, K >& comps )
{
    constexpr std::size_t Ncoef = tax::numMonomials( N, M );
    ImageZonotope< T, static_cast< int >( K ) > out;
    out.generators.resize( static_cast< Eigen::Index >( K ), 0 );
    for ( std::size_t i = 0; i < K; ++i )
        out.center( static_cast< Eigen::Index >( i ) ) = x( comps[i] )[0];

    for ( std::size_t k = 1; k < Ncoef; ++k )
    {
        tax::la::VecNT< static_cast< int >( K ), T > g;
        bool nonzero = false;
        for ( std::size_t i = 0; i < K; ++i )
        {
            g( static_cast< Eigen::Index >( i ) ) = x( comps[i] )[k];
            nonzero = nonzero || g( static_cast< Eigen::Index >( i ) ) != T{ 0 };
        }
        if ( !nonzero ) continue;

        const auto alpha = tax::unflatIndex< M >( k );
        bool allEven = true;
        for ( int j = 0; j < M; ++j )
            if ( alpha[static_cast< std::size_t >( j )] % 2 != 0 ) allEven = false;

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
    return out;
}

// The exact degree-1 (parallelotope) frame of the image of selected
// components: center = constant terms, G(i,j) = coefficient of ξ_j. Requires
// K == M so the frame is a square-G Zonotope. NOT an enclosure unless the map
// is linear — use zonotopeEnclosure for a guaranteed cover.
template < class T, int N, int M, class Storage, int D >
[[nodiscard]] Zonotope< T, M > zonotopeFrame(
    const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 >&
        x,
    const std::array< int, static_cast< std::size_t >( M ) >& comps )
{
    Zonotope< T, M > z;
    for ( std::size_t i = 0; i < static_cast< std::size_t >( M ); ++i )
    {
        const auto& f = x( comps[i] );
        z.center( static_cast< Eigen::Index >( i ) ) = f[0];
        for ( int j = 0; j < M; ++j )
        {
            tax::MultiIndex< M > e{};
            e[static_cast< std::size_t >( j )] = 1;
            z.generators( static_cast< Eigen::Index >( i ), j ) = f.coeff( e );
        }
    }
    return z;
}

// Convenience: frame over the first M components (the square case).
template < class T, int N, int M, class Storage, int D >
[[nodiscard]] Zonotope< T, M > zonotopeFrame(
    const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 >&
        x )
{
    std::array< int, static_cast< std::size_t >( M ) > comps{};
    for ( int i = 0; i < M; ++i ) comps[static_cast< std::size_t >( i )] = i;
    return zonotopeFrame( x, comps );
}

}  // namespace tax::domain
