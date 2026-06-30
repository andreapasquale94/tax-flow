// include/tax/sgp4/satellite.hpp
//
// The user-facing SGP4 surface.  A Satellite<T> is built from a parsed Tle and
// a set of element Seeds<T>, runs sgp4init in its constructor, and evaluates
// the TEME state at any time past epoch.
//
// Taylor compatibility: with T = double you get ordinary ephemerides.  With T
// a tax::TaylorExpansion, seed whichever mean elements you want to expand as
// identity expansions (value + delta) via Seeds<T>; the returned State<T> is
// then the polynomial map of position/velocity w.r.t. those elements — i.e.
// the sensitivity / uncertainty of the orbit to the TLE inputs.

#pragma once

#include <tax/la/types.hpp>
#include <tax/sgp4/detail/sgp4_core.hpp>
#include <tax/sgp4/elset_rec.hpp>
#include <tax/sgp4/gravconst.hpp>
#include <tax/sgp4/tle.hpp>

namespace tax::sgp4
{

/// TEME position [km] and velocity [km/s] of a propagated state.
template < class T >
struct State
{
    tax::la::VecNT< 3, T > r;  ///< position [km]
    tax::la::VecNT< 3, T > v;  ///< velocity [km/s]
};

/// The nine SGP4 mean-element inputs, as scalar T.  Override any field with an
/// identity Taylor expansion to expand the orbit w.r.t. that element.
template < class T >
struct Seeds
{
    T bstar{};
    T ndot{};
    T nddot{};
    T inclo{};
    T nodeo{};
    T ecco{};
    T argpo{};
    T mo{};
    T no_kozai{};
};

/// Build all-constant seeds from a parsed TLE (each element a flat T).
template < class T >
[[nodiscard]] Seeds< T > seedsFrom( const Tle& tle )
{
    return Seeds< T >{ T( tle.bstar ), T( tle.ndot ),  T( tle.nddot ),
                       T( tle.inclo ), T( tle.nodeo ), T( tle.ecco ),
                       T( tle.argpo ), T( tle.mo ),    T( tle.no_kozai ) };
}

/// SGP4 propagator over a single satellite, scalar T.
template < class T >
class Satellite
{
   public:
    /// Construct from a parsed TLE.  By default every element is seeded as a
    /// constant (plain ephemeris).  Pass custom `seeds` to expand selected
    /// elements as Taylor expansions.  `opsmode` is 'i' (improved) or 'a'
    /// (afspc).
    explicit Satellite( const Tle& tle, GravModel model = GravModel::Wgs72, Seeds< T > seeds = {},
                        char opsmode = 'i' )
    {
        rec_.setGrav( model );
        rec_.jdsatepoch = tle.jdsatepoch;
        rec_.jdsatepochF = tle.jdsatepochF;
        rec_.epochdays = tle.epochdays;
        rec_.epochyr = tle.epochyr;

        rec_.bstar = seeds.bstar;
        rec_.ndot = seeds.ndot;
        rec_.nddot = seeds.nddot;
        rec_.inclo = seeds.inclo;
        rec_.nodeo = seeds.nodeo;
        rec_.ecco = seeds.ecco;
        rec_.argpo = seeds.argpo;
        rec_.mo = seeds.mo;
        rec_.no_kozai = seeds.no_kozai;

        detail::sgp4init( rec_, opsmode );
    }

    /// Convenience: seed every element as a constant from the TLE.
    [[nodiscard]] static Satellite fromTle( const Tle& tle, GravModel model = GravModel::Wgs72,
                                            char opsmode = 'i' )
    {
        return Satellite( tle, model, seedsFrom< T >( tle ), opsmode );
    }

    /// Evaluate the TEME state at `tsinceMinutes` past epoch.
    [[nodiscard]] State< T > propagate( double tsinceMinutes )
    {
        State< T > s;
        detail::sgp4( rec_, tsinceMinutes, s.r, s.v );
        return s;
    }

    /// SGP4 error code from the most recent propagate() (0 == ok).
    [[nodiscard]] int error() const noexcept { return rec_.error; }

    /// Access the underlying record (mean elements, derived constants).
    [[nodiscard]] ElsetRec< T >& rec() noexcept { return rec_; }
    [[nodiscard]] const ElsetRec< T >& rec() const noexcept { return rec_; }

   private:
    ElsetRec< T > rec_{};
};

}  // namespace tax::sgp4
