// include/tax/ads/poly_zonotope.hpp
//
// PolyZonotope<T, N, M, Storage, D> — a *curved* initial-condition set used as
// an ADS domain.
//
// Box and Zonotope describe the initial set as a LINEAR image of the factor
// box: ξ ↦ center + G·ξ (degree 1). A polynomial zonotope lets the initial set
// be a POLYNOMIAL image,
//
//     ξ ↦ map(ξ) = c + Σ_i (Π_k ξ_k^{E_ki}) g_i,    ξ ∈ [-1,1]^M,
//
// i.e. a genuinely bent set — the kind you get when an uncertainty that is
// Gaussian in one coordinate system (orbital elements, a phase angle) is
// expressed in another (Cartesian state). The map IS a DA value, so it is built
// with ordinary TE arithmetic and split with the same per-axis substitution as
// the flow-map payload (da_state::split). Nothing else in the ADS pipeline
// changes: Leaf / AdsTree / AdsDriver are already generic over the domain, and
// a PolyZonotope satisfies the same small interface as Zonotope —
//   center · split · contains · denormalize · splitOrdinate
// — so a curved initial set propagates and subdivides like any other.
//
// Cost: each leaf carries its t₀ map (the curved geometry) in addition to the
// propagated payload, so memory per leaf is ~2× the linear-domain case.

#pragma once

#include <Eigen/Dense>
#include <tax/ads/da_state.hpp>
#include <tax/core/multi_index.hpp>
#include <tax/core/taylor_expansion.hpp>
#include <tax/la/types.hpp>
#include <utility>

namespace tax::ads
{

template < class T, int N, int M, class Storage = tax::storage::Dense, int D = Eigen::Dynamic >
struct PolyZonotope
{
    using TE = tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >;
    using State = Eigen::Matrix< TE, D, 1 >;

    State map{};                      // ξ ↦ initial state (the curved geometry)
    tax::la::VecNT< D, T > center{};  // map(0), the physical centre

    [[nodiscard]] static PolyZonotope fromMap( State m )
    {
        PolyZonotope z;
        z.map = std::move( m );
        z.recenter();
        return z;
    }

    void recenter()
    {
        if constexpr ( D == Eigen::Dynamic ) center.resize( map.size() );
        for ( Eigen::Index i = 0; i < map.size(); ++i ) center( i ) = map( i )[0];
    }

    // Jacobian ∂map/∂ξ|₀ (D×M): the degree-1 coefficients — the local linear
    // image used by contains / splitOrdinate.
    [[nodiscard]] Eigen::Matrix< T, D, M > jacobian() const
    {
        Eigen::Matrix< T, D, M > J;
        if constexpr ( D == Eigen::Dynamic ) J.resize( map.size(), M );
        for ( Eigen::Index i = 0; i < map.size(); ++i )
            for ( int j = 0; j < M; ++j )
            {
                tax::MultiIndex< M > e{};
                e[static_cast< std::size_t >( j )] = 1;
                J( i, j ) = map( i )[tax::flatIndex< M >( e )];
            }
        return J;
    }

    // Map factor coordinates to the physical initial state.
    template < class Derived >
    [[nodiscard]] tax::la::VecNT< D, T > denormalize( const Eigen::MatrixBase< Derived >& d ) const
    {
        tax::la::VecNT< D, T > out;
        if constexpr ( D == Eigen::Dynamic ) out.resize( map.size() );
        for ( Eigen::Index i = 0; i < map.size(); ++i ) out( i ) = map( i ).eval( d );
        return out;
    }

    // Approximate membership: recover ξ from the linear part and test the box.
    // (The set is curved, so this is a first-order test — exact for the linear
    // case and adequate for point location; the propagation path never calls
    // it.)
    template < class Derived >
    [[nodiscard]] bool contains( const Eigen::MatrixBase< Derived >& pt, T tol = T{ 1e-9 } ) const
    {
        const Eigen::Matrix< T, D, M > J = jacobian();
        const tax::la::VecNT< D, T > rhs = pt - center;
        const tax::la::VecNT< M, T > xi = J.colPivHouseholderQr().solve( rhs );
        for ( int i = 0; i < M; ++i )
            if ( xi( i ) > T{ 1 } + tol || xi( i ) < T{ -1 } - tol ) return false;
        return ( J * xi - rhs ).cwiseAbs().maxCoeff() <=
               tol * ( T{ 1 } + rhs.cwiseAbs().maxCoeff() );
    }

    // Bisect along factor `dim`: split the map with the same substitution as the
    // payload, ξ_dim → ±0.5 + 0.5·ξ'_dim, and re-centre the children.
    [[nodiscard]] std::pair< PolyZonotope, PolyZonotope > split( int dim ) const
    {
        auto pr = tax::ads::split( map, *this, dim );
        return { fromMap( std::move( pr.first ) ), fromMap( std::move( pr.second ) ) };
    }

    // Centre position along the linear image of factor `dim` — sibling ordering
    // for merge(), matching Box/Zonotope::splitOrdinate.
    [[nodiscard]] T splitOrdinate( int dim ) const
    {
        const Eigen::Matrix< T, D, M > J = jacobian();
        return center.dot( J.col( dim ) );
    }
};

// create(poly_zonotope, ic_center): the initial DA state IS the curved map.
template < int P, int M, class Storage = tax::storage::Dense, class T, int D >
[[nodiscard]] Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< P, M >, Storage >, D,
                             1 >
create( const PolyZonotope< T, P, M, Storage, D >& z,
        const Eigen::Matrix< T, D, 1 >& /*ic_center*/ )
{
    return z.map;
}

}  // namespace tax::ads
