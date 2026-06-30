// include/tax/sgp4/detail/sgp4_core.hpp
//
// Near-earth SGP4 core: initl (epoch initialisation), sgp4 (the combined
// SGP4/SDP4 evaluation) and sgp4init (build all derived constants from the
// mean elements).  Templated on the scalar T; deep-space orbits dispatch into
// the routines in deep_space.hpp.  Branches and convergence tests go through
// detail::cst, angle reductions through detail::mod, and |.| of a value
// through detail::dabs — so the same source serves double and TaylorExpansion.

#pragma once

#include <cmath>
#include <tax/la/types.hpp>
#include <tax/sgp4/detail/deep_space.hpp>
#include <tax/sgp4/detail/scalar.hpp>
#include <tax/sgp4/detail/time.hpp>
#include <tax/sgp4/elset_rec.hpp>

namespace tax::sgp4::detail
{

// ---------------------------------------------------------------------------
// initl — auxiliary epoch quantities and un-kozai of the mean motion.
// ---------------------------------------------------------------------------
template < class T >
void initl( ElsetRec< T >& rec, double epoch )
{
    using std::cos;
    using std::pow;
    using std::sin;
    using std::sqrt;

    constexpr double x2o3 = 2.0 / 3.0;

    rec.eccsq = rec.ecco * rec.ecco;
    rec.omeosq = 1.0 - rec.eccsq;
    rec.rteosq = sqrt( rec.omeosq );
    rec.cosio = cos( rec.inclo );
    rec.cosio2 = rec.cosio * rec.cosio;

    // ---- un-kozai the mean motion ----
    T ak = pow( rec.xke / rec.no_kozai, x2o3 );
    T d1 = 0.75 * rec.j2 * ( 3.0 * rec.cosio2 - 1.0 ) / ( rec.rteosq * rec.omeosq );
    T del = d1 / ( ak * ak );
    T adel = ak * ( 1.0 - del * del - del * ( 1.0 / 3.0 + 134.0 * del * del / 81.0 ) );
    del = d1 / ( adel * adel );
    rec.no_unkozai = rec.no_kozai / ( 1.0 + del );

    rec.ao = pow( rec.xke / rec.no_unkozai, x2o3 );
    rec.sinio = sin( rec.inclo );
    T po = rec.ao * rec.omeosq;
    rec.con42 = 1.0 - 5.0 * rec.cosio2;
    rec.con41 = -rec.con42 - rec.cosio2 - rec.cosio2;
    rec.ainv = 1.0 / rec.ao;
    rec.posq = po * po;
    rec.rp = rec.ao * ( 1.0 - rec.ecco );
    rec.method = 'n';

    // sgp4fix modern approach to finding sidereal time
    rec.gsto = gstime( epoch + 2433281.5 );
}

// ---------------------------------------------------------------------------
// sgp4 — evaluate the propagator at `tsince` minutes past epoch.  Writes the
// TEME position [km] and velocity [km/s] into r and v.  Returns false (and
// sets rec.error) on a propagation error.
// ---------------------------------------------------------------------------
template < class T >
bool sgp4( ElsetRec< T >& rec, double tsince, tax::la::VecNT< 3, T >& r, tax::la::VecNT< 3, T >& v )
{
    using std::atan2;
    using std::cos;
    using std::pow;
    using std::sin;
    using std::sqrt;

    constexpr double temp4 = 1.5e-12;
    constexpr double x2o3 = 2.0 / 3.0;
    const double vkmpersec = rec.radiusearthkm * rec.xke / 60.0;

    rec.t = tsince;
    rec.error = 0;

    // ---- secular gravity and atmospheric drag ----
    T xmdf = rec.mo + rec.mdot * rec.t;
    T argpdf = rec.argpo + rec.argpdot * rec.t;
    T nodedf = rec.nodeo + rec.nodedot * rec.t;
    rec.argpm = argpdf;
    rec.mm = xmdf;
    double t2 = rec.t * rec.t;
    rec.nodem = nodedf + rec.nodecf * t2;
    T tempa = 1.0 - rec.cc1 * rec.t;
    T tempe = rec.bstar * rec.cc4 * rec.t;
    T templ = rec.t2cof * t2;

    if ( rec.isimp != 1 )
    {
        T delomg = rec.omgcof * rec.t;
        T delmtemp = 1.0 + rec.eta * cos( xmdf );
        T delm = rec.xmcof * ( delmtemp * delmtemp * delmtemp - rec.delmo );
        T temp = delomg + delm;
        rec.mm = xmdf + temp;
        rec.argpm = argpdf - temp;
        double t3 = t2 * rec.t;
        double t4 = t3 * rec.t;
        tempa = tempa - rec.d2 * t2 - rec.d3 * t3 - rec.d4 * t4;
        tempe = tempe + rec.bstar * rec.cc5 * ( sin( rec.mm ) - rec.sinmao );
        templ = templ + rec.t3cof * t3 + t4 * ( rec.t4cof + rec.t * rec.t5cof );
    }

    rec.nm = rec.no_unkozai;
    rec.em = rec.ecco;
    rec.inclm = rec.inclo;
    if ( rec.method == 'd' )
    {
        double tc = rec.t;
        dspace( rec, tc );
    }

    if ( cst( rec.nm ) <= 0.0 )
    {
        rec.error = 2;
        return false;
    }

    rec.am = pow( rec.xke / rec.nm, x2o3 ) * tempa * tempa;
    rec.nm = rec.xke / pow( rec.am, 1.5 );
    rec.em = rec.em - tempe;

    if ( ( cst( rec.em ) >= 1.0 ) || ( cst( rec.em ) < -0.001 ) )
    {
        rec.error = 1;
        return false;
    }
    if ( cst( rec.em ) < 1.0e-6 ) rec.em = T( 1.0e-6 );
    rec.mm = rec.mm + rec.no_unkozai * templ;
    T xlm = rec.mm + rec.argpm + rec.nodem;
    rec.emsq = rec.em * rec.em;

    rec.nodem = mod( rec.nodem, twopi );
    rec.argpm = mod( rec.argpm, twopi );
    xlm = mod( xlm, twopi );
    rec.mm = mod( xlm - rec.argpm - rec.nodem, twopi );

    // sgp4fix recover singly averaged mean elements
    rec.am = rec.am;
    rec.em = rec.em;
    rec.im = rec.inclm;
    rec.Om = rec.nodem;
    rec.om = rec.argpm;
    rec.mm = rec.mm;
    rec.nm = rec.nm;

    // ---- extra mean quantities ----
    rec.sinim = sin( rec.inclm );
    rec.cosim = cos( rec.inclm );

    // ---- add lunar-solar periodics ----
    rec.ep = rec.em;
    T xincp = rec.inclm;
    rec.inclp = rec.inclm;
    rec.argpp = rec.argpm;
    rec.nodep = rec.nodem;
    rec.mp = rec.mm;
    T sinip = rec.sinim;
    T cosip = rec.cosim;
    if ( rec.method == 'd' )
    {
        dpper( rec, 'n' );
        xincp = rec.inclp;
        if ( cst( xincp ) < 0.0 )
        {
            xincp = -xincp;
            rec.nodep = rec.nodep + pi;
            rec.argpp = rec.argpp - pi;
        }
        if ( ( cst( rec.ep ) < 0.0 ) || ( cst( rec.ep ) > 1.0 ) )
        {
            rec.error = 3;
            return false;
        }
    }

    // ---- long period periodics ----
    if ( rec.method == 'd' )
    {
        sinip = sin( xincp );
        cosip = cos( xincp );
        rec.aycof = -0.5 * rec.j3oj2 * sinip;
        // sgp4fix for divide by zero for xincp = 180 deg
        if ( std::abs( cst( cosip ) + 1.0 ) > 1.5e-12 )
            rec.xlcof = -0.25 * rec.j3oj2 * sinip * ( 3.0 + 5.0 * cosip ) / ( 1.0 + cosip );
        else
            rec.xlcof = -0.25 * rec.j3oj2 * sinip * ( 3.0 + 5.0 * cosip ) / temp4;
    }
    T axnl = rec.ep * cos( rec.argpp );
    T temp = 1.0 / ( rec.am * ( 1.0 - rec.ep * rec.ep ) );
    T aynl = rec.ep * sin( rec.argpp ) + temp * rec.aycof;
    T xl = rec.mp + rec.argpp + rec.nodep + temp * rec.xlcof * axnl;

    // ---- solve kepler's equation ----
    T u = mod( xl - rec.nodep, twopi );
    T eo1 = u;
    double tem5val = 9999.9;
    int ktr = 1;
    T sineo1{};
    T coseo1{};
    T tem5{};
    while ( ( std::abs( tem5val ) >= 1.0e-12 ) && ( ktr <= 10 ) )
    {
        sineo1 = sin( eo1 );
        coseo1 = cos( eo1 );
        tem5 = 1.0 - coseo1 * axnl - sineo1 * aynl;
        tem5 = ( u - aynl * coseo1 + axnl * sineo1 - eo1 ) / tem5;
        tem5val = cst( tem5 );
        if ( std::abs( tem5val ) >= 0.95 ) tem5 = T( tem5val > 0.0 ? 0.95 : -0.95 );
        eo1 = eo1 + tem5;
        ktr = ktr + 1;
    }

    // ---- short period preliminary quantities ----
    T ecose = axnl * coseo1 + aynl * sineo1;
    T esine = axnl * sineo1 - aynl * coseo1;
    T el2 = axnl * axnl + aynl * aynl;
    T pl = rec.am * ( 1.0 - el2 );
    if ( cst( pl ) < 0.0 )
    {
        rec.error = 4;
        return false;
    }

    T rl = rec.am * ( 1.0 - ecose );
    T rdotl = sqrt( rec.am ) * esine / rl;
    T rvdotl = sqrt( pl ) / rl;
    T betal = sqrt( 1.0 - el2 );
    temp = esine / ( 1.0 + betal );
    T sinu = rec.am / rl * ( sineo1 - aynl - axnl * temp );
    T cosu = rec.am / rl * ( coseo1 - axnl + aynl * temp );
    T su = atan2( sinu, cosu );
    T sin2u = ( cosu + cosu ) * sinu;
    T cos2u = 1.0 - 2.0 * sinu * sinu;
    temp = 1.0 / pl;
    T temp1 = 0.5 * rec.j2 * temp;
    T temp2 = temp1 * temp;

    // ---- update for short period periodics ----
    if ( rec.method == 'd' )
    {
        T cosisq = cosip * cosip;
        rec.con41 = 3.0 * cosisq - 1.0;
        rec.x1mth2 = 1.0 - cosisq;
        rec.x7thm1 = 7.0 * cosisq - 1.0;
    }
    T mrt = rl * ( 1.0 - 1.5 * temp2 * betal * rec.con41 ) + 0.5 * temp1 * rec.x1mth2 * cos2u;
    su = su - 0.25 * temp2 * rec.x7thm1 * sin2u;
    T xnode = rec.nodep + 1.5 * temp2 * cosip * sin2u;
    T xinc = xincp + 1.5 * temp2 * cosip * sinip * cos2u;
    T mvt = rdotl - rec.nm * temp1 * rec.x1mth2 * sin2u / rec.xke;
    T rvdot = rvdotl + rec.nm * temp1 * ( rec.x1mth2 * cos2u + 1.5 * rec.con41 ) / rec.xke;

    // ---- orientation vectors ----
    T sinsu = sin( su );
    T cossu = cos( su );
    T snod = sin( xnode );
    T cnod = cos( xnode );
    T sini = sin( xinc );
    T cosi = cos( xinc );
    T xmx = -snod * cosi;
    T xmy = cnod * cosi;
    T ux = xmx * sinsu + cnod * cossu;
    T uy = xmy * sinsu + snod * cossu;
    T uz = sini * sinsu;
    T vx = xmx * cossu - cnod * sinsu;
    T vy = xmy * cossu - snod * sinsu;
    T vz = sini * cossu;

    // ---- position [km] and velocity [km/s] ----
    r( 0 ) = ( mrt * ux ) * rec.radiusearthkm;
    r( 1 ) = ( mrt * uy ) * rec.radiusearthkm;
    r( 2 ) = ( mrt * uz ) * rec.radiusearthkm;
    v( 0 ) = ( mvt * ux + rvdot * vx ) * vkmpersec;
    v( 1 ) = ( mvt * uy + rvdot * vy ) * vkmpersec;
    v( 2 ) = ( mvt * uz + rvdot * vz ) * vkmpersec;

    // sgp4fix for decaying satellites
    if ( cst( mrt ) < 1.0 )
    {
        rec.error = 6;
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// sgp4init — build all derived constants from the mean elements already set
// in `rec` (bstar, ecco, inclo, nodeo, argpo, mo, no_kozai, epoch, grav model),
// then propagate to epoch to initialise the remaining state.
// ---------------------------------------------------------------------------
template < class T >
bool sgp4init( ElsetRec< T >& rec, char opsmode )
{
    using std::cos;
    using std::pow;
    using std::sin;

    constexpr double temp4 = 1.5e-12;
    constexpr double x2o3 = 2.0 / 3.0;

    const double epoch = ( rec.jdsatepoch + rec.jdsatepochF ) - 2433281.5;

    rec.setGrav( rec.whichconst );
    rec.error = 0;
    rec.operationmode = opsmode;
    rec.method = 'n';
    rec.init = 'y';
    rec.t = 0.0;

    const double ss = 78.0 / rec.radiusearthkm + 1.0;
    const double qzms2ttemp = ( 120.0 - 78.0 ) / rec.radiusearthkm;
    const double qzms2t = qzms2ttemp * qzms2ttemp * qzms2ttemp * qzms2ttemp;

    initl( rec, epoch );

    rec.a = pow( rec.no_unkozai * rec.tumin, -2.0 / 3.0 );
    rec.alta = rec.a * ( 1.0 + rec.ecco ) - 1.0;
    rec.altp = rec.a * ( 1.0 - rec.ecco ) - 1.0;
    rec.error = 0;

    if ( ( cst( rec.omeosq ) >= 0.0 ) || ( cst( rec.no_unkozai ) >= 0.0 ) )
    {
        rec.isimp = 0;
        if ( cst( rec.rp ) < ( 220.0 / rec.radiusearthkm + 1.0 ) ) rec.isimp = 1;
        T sfour = ss;
        T qzms24 = qzms2t;
        T perige = ( rec.rp - 1.0 ) * rec.radiusearthkm;

        // ---- for perigees below 156 km, s and qoms2t are altered ----
        if ( cst( perige ) < 156.0 )
        {
            sfour = perige - 78.0;
            if ( cst( perige ) < 98.0 ) sfour = T( 20.0 );
            T qzms24temp = ( 120.0 - sfour ) / rec.radiusearthkm;
            qzms24 = qzms24temp * qzms24temp * qzms24temp * qzms24temp;
            sfour = sfour / rec.radiusearthkm + 1.0;
        }
        T pinvsq = 1.0 / rec.posq;

        T tsi = 1.0 / ( rec.ao - sfour );
        rec.eta = rec.ao * rec.ecco * tsi;
        T etasq = rec.eta * rec.eta;
        T eeta = rec.ecco * rec.eta;
        T psisq = dabs( 1.0 - etasq );
        T coef = qzms24 * pow( tsi, 4.0 );
        T coef1 = coef / pow( psisq, 3.5 );
        T cc2 =
            coef1 * rec.no_unkozai *
            ( rec.ao * ( 1.0 + 1.5 * etasq + eeta * ( 4.0 + etasq ) ) +
              0.375 * rec.j2 * tsi / psisq * rec.con41 * ( 8.0 + 3.0 * etasq * ( 8.0 + etasq ) ) );
        rec.cc1 = rec.bstar * cc2;
        T cc3 = T( 0.0 );
        if ( cst( rec.ecco ) > 1.0e-4 )
            cc3 = -2.0 * coef * tsi * rec.j3oj2 * rec.no_unkozai * rec.sinio / rec.ecco;
        rec.x1mth2 = 1.0 - rec.cosio2;
        rec.cc4 = 2.0 * rec.no_unkozai * coef1 * rec.ao * rec.omeosq *
                  ( rec.eta * ( 2.0 + 0.5 * etasq ) + rec.ecco * ( 0.5 + 2.0 * etasq ) -
                    rec.j2 * tsi / ( rec.ao * psisq ) *
                        ( -3.0 * rec.con41 * ( 1.0 - 2.0 * eeta + etasq * ( 1.5 - 0.5 * eeta ) ) +
                          0.75 * rec.x1mth2 * ( 2.0 * etasq - eeta * ( 1.0 + etasq ) ) *
                              cos( 2.0 * rec.argpo ) ) );
        rec.cc5 =
            2.0 * coef1 * rec.ao * rec.omeosq * ( 1.0 + 2.75 * ( etasq + eeta ) + eeta * etasq );
        T cosio4 = rec.cosio2 * rec.cosio2;
        T temp1 = 1.5 * rec.j2 * pinvsq * rec.no_unkozai;
        T temp2 = 0.5 * temp1 * rec.j2 * pinvsq;
        T temp3 = -0.46875 * rec.j4 * pinvsq * pinvsq * rec.no_unkozai;
        rec.mdot = rec.no_unkozai + 0.5 * temp1 * rec.rteosq * rec.con41 +
                   0.0625 * temp2 * rec.rteosq * ( 13.0 - 78.0 * rec.cosio2 + 137.0 * cosio4 );
        rec.argpdot = -0.5 * temp1 * rec.con42 +
                      0.0625 * temp2 * ( 7.0 - 114.0 * rec.cosio2 + 395.0 * cosio4 ) +
                      temp3 * ( 3.0 - 36.0 * rec.cosio2 + 49.0 * cosio4 );
        T xhdot1 = -temp1 * rec.cosio;
        rec.nodedot = xhdot1 + ( 0.5 * temp2 * ( 4.0 - 19.0 * rec.cosio2 ) +
                                 2.0 * temp3 * ( 3.0 - 7.0 * rec.cosio2 ) ) *
                                   rec.cosio;
        T xpidot = rec.argpdot + rec.nodedot;
        rec.omgcof = rec.bstar * cc3 * cos( rec.argpo );
        rec.xmcof = T( 0.0 );
        if ( cst( rec.ecco ) > 1.0e-4 ) rec.xmcof = -x2o3 * coef * rec.bstar / eeta;
        rec.nodecf = 3.5 * rec.omeosq * xhdot1 * rec.cc1;
        rec.t2cof = 1.5 * rec.cc1;
        // sgp4fix for divide by zero with xinco = 180 deg
        if ( std::abs( cst( rec.cosio ) + 1.0 ) > 1.5e-12 )
            rec.xlcof =
                -0.25 * rec.j3oj2 * rec.sinio * ( 3.0 + 5.0 * rec.cosio ) / ( 1.0 + rec.cosio );
        else
            rec.xlcof = -0.25 * rec.j3oj2 * rec.sinio * ( 3.0 + 5.0 * rec.cosio ) / temp4;
        rec.aycof = -0.5 * rec.j3oj2 * rec.sinio;
        T delmotemp = 1.0 + rec.eta * cos( rec.mo );
        rec.delmo = delmotemp * delmotemp * delmotemp;
        rec.sinmao = sin( rec.mo );
        rec.x7thm1 = 7.0 * rec.cosio2 - 1.0;

        // ---- deep space initialisation ----
        if ( ( 2.0 * pi / cst( rec.no_unkozai ) ) >= 225.0 )
        {
            rec.method = 'd';
            rec.isimp = 1;
            double tc = 0.0;
            rec.inclm = rec.inclo;

            dscom( rec, epoch, rec.ecco, rec.argpo, tc, rec.inclo, rec.nodeo, rec.no_unkozai );

            rec.ep = rec.ecco;
            rec.inclp = rec.inclo;
            rec.nodep = rec.nodeo;
            rec.argpp = rec.argpo;
            rec.mp = rec.mo;

            dpper( rec, rec.init );

            rec.ecco = rec.ep;
            rec.inclo = rec.inclp;
            rec.nodeo = rec.nodep;
            rec.argpo = rec.argpp;
            rec.mo = rec.mp;

            rec.argpm = T( 0.0 );
            rec.nodem = T( 0.0 );
            rec.mm = T( 0.0 );

            dsinit( rec, tc, xpidot );
        }

        // ---- set variables if not deep space ----
        if ( rec.isimp != 1 )
        {
            T cc1sq = rec.cc1 * rec.cc1;
            rec.d2 = 4.0 * rec.ao * tsi * cc1sq;
            T temp = rec.d2 * tsi * rec.cc1 / 3.0;
            rec.d3 = ( 17.0 * rec.ao + sfour ) * temp;
            rec.d4 = 0.5 * temp * rec.ao * tsi * ( 221.0 * rec.ao + 31.0 * sfour ) * rec.cc1;
            rec.t3cof = rec.d2 + 2.0 * cc1sq;
            rec.t4cof = 0.25 * ( 3.0 * rec.d3 + rec.cc1 * ( 12.0 * rec.d2 + 10.0 * cc1sq ) );
            rec.t5cof = 0.2 * ( 3.0 * rec.d4 + 12.0 * rec.cc1 * rec.d3 + 6.0 * rec.d2 * rec.d2 +
                                15.0 * cc1sq * ( 2.0 * rec.d2 + cc1sq ) );
        }
    }

    // propagate to zero epoch to initialise all other quantities
    tax::la::VecNT< 3, T > r, v;
    sgp4( rec, 0.0, r, v );

    rec.init = 'n';
    return true;
}

}  // namespace tax::sgp4::detail
