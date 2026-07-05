// include/tax/ads/da_state.hpp
//
// split — re-identify a DA state's domain on the two halves of its
//         parent domain along factor `dim`. Substitution:
//           ξ_dim  →  -0.5 + 0.5 · ξ'_dim   (left half)
//           ξ_dim  →  +0.5 + 0.5 · ξ'_dim   (right half)
//         so the children carry polynomials in their own local
//         [-1, 1] coordinates. The domain-to-state bridge (create)
//         lives in tax/domain/create.hpp.

#pragma once

#include <tax/core/taylor_expansion.hpp>
#include <tax/domain/create.hpp>
#include <tax/domain/detail/substitute_axis.hpp>
#include <tax/la/types.hpp>
#include <utility>

namespace tax::ads
{

// split(state, dim): produce the left/right halves.
// Deduces Storage from the input.
template < class T, int N, int M, class Storage, int D >
[[nodiscard]] std::pair<
    Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 >,
    Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 > >
split( const Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D,
                            1 >& state,
       int dim )
{
    using State =
        Eigen::Matrix< tax::TaylorExpansion< T, tax::IsotropicScheme< N, M >, Storage >, D, 1 >;
    State L{ state.size() };
    State R{ state.size() };
    for ( Eigen::Index i = 0; i < state.size(); ++i )
    {
        L( i ) = tax::domain::detail::substituteAxis( state( i ), dim, T{ -0.5 }, T{ 0.5 } );
        R( i ) = tax::domain::detail::substituteAxis( state( i ), dim, T{ 0.5 }, T{ 0.5 } );
    }
    return { std::move( L ), std::move( R ) };
}

}  // namespace tax::ads
