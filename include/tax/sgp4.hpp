// include/tax/sgp4.hpp
//
// Opt-in SGP4 umbrella — include only this header to access the full SGP4
// surface (gravity models, TLE parsing, the templated propagator and the
// Satellite/Seeds/State API in namespace tax::sgp4).
//
// A port of the SGP4/SDP4 satellite propagator (Vallado / aholinch), templated
// on the scalar type so it runs on `double` for ordinary ephemerides or on a
// tax::TaylorExpansion to obtain a polynomial map of the state w.r.t. the
// seeded TLE mean elements.

#pragma once

#include <tax/sgp4/detail/deep_space.hpp>
#include <tax/sgp4/detail/scalar.hpp>
#include <tax/sgp4/detail/sgp4_core.hpp>
#include <tax/sgp4/detail/time.hpp>
#include <tax/sgp4/elset_rec.hpp>
#include <tax/sgp4/gravconst.hpp>
#include <tax/sgp4/satellite.hpp>
#include <tax/sgp4/tle.hpp>
