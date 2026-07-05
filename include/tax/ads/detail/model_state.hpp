// include/tax/ads/detail/model_state.hpp
//
// Duck-typed detection of a Taylor-model-valued ADS state: an Eigen vector
// whose scalar exposes polynomial() + remainder() (tax::model::TaylorModel).
// Structural on purpose — the ADS overloads keyed on this concept never have
// to name (or include) the tax::model module, so every core ADS header keeps
// building against a tax checkout without it.

#pragma once

namespace tax::ads::detail
{

template < class S >
concept ModelValuedState = requires( const S& s ) {
    s( 0 ).polynomial();
    s( 0 ).remainder();
};

}  // namespace tax::ads::detail
