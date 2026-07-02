// include/tax/domain.hpp
//
// Facade: set-valued domain module (namespace tax::domain).
//
// Geometric primitives describing sets of initial conditions as images of the
// normalized factor cube ξ ∈ [-1,1]^M, the DA-state bridge (create), and the
// query/enclosure layer (localize, interval hulls, zonotope enclosures,
// ellipsoid coverings, factor-frame reorientation). Consumed by tax::ads,
// usable standalone.

#pragma once

#include <tax/domain/box.hpp>
#include <tax/domain/create.hpp>
#include <tax/domain/domain.hpp>
#include <tax/domain/ellipsoid.hpp>
#include <tax/domain/enclosure.hpp>
#include <tax/domain/polynomial_zonotope.hpp>
#include <tax/domain/reorient.hpp>
#include <tax/domain/zonotope.hpp>
