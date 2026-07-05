// include/tax/ads.hpp
//
// Umbrella header for the tax::ads module. Users include only this.
// The domain primitives (Box, Zonotope, PolynomialZonotope, create, the
// enclosure/query layer) live in the tax::domain module — <tax/domain.hpp> —
// re-included here because every ADS run needs an IC domain.

#pragma once

#include <tax/ads/da_state.hpp>
#include <tax/ads/driver.hpp>
#include <tax/ads/leaf.hpp>
#include <tax/ads/merge.hpp>
#include <tax/ads/propagate.hpp>
#include <tax/ads/refine.hpp>
#include <tax/ads/refine_criteria.hpp>
#include <tax/ads/solution.hpp>
#include <tax/ads/split_criteria.hpp>
#include <tax/ads/split_event.hpp>
#include <tax/ads/tree.hpp>
#include <tax/domain.hpp>

// ADS over validated Taylor-model states (methods::Picard) — only when the
// tax core ships the tax::model module (tax PR "feat(model)").
#if __has_include( <tax/model.hpp>)
#include <tax/ads/model.hpp>
#endif
