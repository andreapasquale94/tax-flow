// include/tax/sgp4/detail/deep_space.hpp
//
// SDP4 deep-space routines, templated on the scalar T: dpper (lunar-solar long
// period periodics), dscom (deep-space common terms), dsinit (resonance
// initialisation) and dspace (secular resonance integration).  Ported from
// Vallado's SGP4.c with all control-flow branches keyed off the constant part
// (detail::cst) and angle reductions through detail::mod, so the routines are
// correct for both double and TaylorExpansion scalars.

#pragma once

#include <cmath>
#include <tax/sgp4/detail/scalar.hpp>
#include <tax/sgp4/elset_rec.hpp>

namespace tax::sgp4::detail
{

// ---------------------------------------------------------------------------
// dpper — deep-space long-period periodic contributions to the mean elements.
// `init` == 'y' on the epoch (sgp4init) call, 'n' on propagation calls.
// ---------------------------------------------------------------------------
template < class T >
void dpper( ElsetRec< T >& rec, char init )
{
    using std::atan2;
    using std::cos;
    using std::sin;

    constexpr double zns = 1.19459e-5;
    constexpr double zes = 0.01675;
    constexpr double znl = 1.5835218e-4;
    constexpr double zel = 0.05490;

    // ---- time-varying periodics (epoch-only scalars stay double) ----
    double zm = rec.zmos + zns * rec.t;
    if ( init == 'y' ) zm = rec.zmos;
    double zf = zm + 2.0 * zes * std::sin( zm );
    double sinzf = std::sin( zf );
    double f2 = 0.5 * sinzf * sinzf - 0.25;
    double f3 = -0.5 * sinzf * std::cos( zf );
    T ses = rec.se2 * f2 + rec.se3 * f3;
    T sis = rec.si2 * f2 + rec.si3 * f3;
    T sls = rec.sl2 * f2 + rec.sl3 * f3 + rec.sl4 * sinzf;
    T sghs = rec.sgh2 * f2 + rec.sgh3 * f3 + rec.sgh4 * sinzf;
    T shs = rec.sh2 * f2 + rec.sh3 * f3;

    zm = rec.zmol + znl * rec.t;
    if ( init == 'y' ) zm = rec.zmol;
    zf = zm + 2.0 * zel * std::sin( zm );
    sinzf = std::sin( zf );
    f2 = 0.5 * sinzf * sinzf - 0.25;
    f3 = -0.5 * sinzf * std::cos( zf );
    T sel = rec.ee2 * f2 + rec.e3 * f3;
    T sil = rec.xi2 * f2 + rec.xi3 * f3;
    T sll = rec.xl2 * f2 + rec.xl3 * f3 + rec.xl4 * sinzf;
    T sghl = rec.xgh2 * f2 + rec.xgh3 * f3 + rec.xgh4 * sinzf;
    T shll = rec.xh2 * f2 + rec.xh3 * f3;

    T pe = ses + sel;
    T pinc = sis + sil;
    T pl = sls + sll;
    T pgh = sghs + sghl;
    T ph = shs + shll;

    if ( init == 'n' )
    {
        pe = pe - rec.peo;
        pinc = pinc - rec.pinco;
        pl = pl - rec.plo;
        pgh = pgh - rec.pgho;
        ph = ph - rec.pho;
        rec.inclp = rec.inclp + pinc;
        rec.ep = rec.ep + pe;
        T sinip = sin( rec.inclp );
        T cosip = cos( rec.inclp );

        // sgp4fix gsfc lyddane choice (perturbed inclination)
        if ( cst( rec.inclp ) >= 0.2 )
        {
            ph = ph / sinip;
            pgh = pgh - cosip * ph;
            rec.argpp = rec.argpp + pgh;
            rec.nodep = rec.nodep + ph;
            rec.mp = rec.mp + pl;
        } else
        {
            // ---- apply periodics with lyddane modification ----
            T sinop = sin( rec.nodep );
            T cosop = cos( rec.nodep );
            T alfdp = sinip * sinop;
            T betdp = sinip * cosop;
            T dalf = ph * cosop + pinc * cosip * sinop;
            T dbet = -ph * sinop + pinc * cosip * cosop;
            alfdp = alfdp + dalf;
            betdp = betdp + dbet;
            rec.nodep = mod( rec.nodep, twopi );
            // sgp4fix for afspc written intrinsic functions
            if ( ( cst( rec.nodep ) < 0.0 ) && ( rec.operationmode == 'a' ) )
                rec.nodep = rec.nodep + twopi;
            T xls = rec.mp + rec.argpp + cosip * rec.nodep;
            T dls = pl + pgh - pinc * rec.nodep * sinip;
            xls = xls + dls;
            xls = mod( xls, twopi );
            T xnoh = rec.nodep;
            rec.nodep = atan2( alfdp, betdp );
            if ( ( cst( rec.nodep ) < 0.0 ) && ( rec.operationmode == 'a' ) )
                rec.nodep = rec.nodep + twopi;
            if ( std::abs( cst( xnoh ) - cst( rec.nodep ) ) > pi )
            {
                if ( cst( rec.nodep ) < cst( xnoh ) )
                    rec.nodep = rec.nodep + twopi;
                else
                    rec.nodep = rec.nodep - twopi;
            }
            rec.mp = rec.mp + pl;
            rec.argpp = xls - rec.mp - cosip * rec.nodep;
        }
    }
}

// ---------------------------------------------------------------------------
// dscom — deep-space common items shared by the secular and periodic terms.
// ---------------------------------------------------------------------------
template < class T >
void dscom( ElsetRec< T >& rec, double epoch, const T& ep, const T& argpp, double tc,
            const T& inclp, const T& nodep, const T& np )
{
    using std::atan2;
    using std::cos;
    using std::sin;
    using std::sqrt;

    constexpr double zes = 0.01675;
    constexpr double zel = 0.05490;
    constexpr double c1ss = 2.9864797e-6;
    constexpr double c1l = 4.7968065e-7;
    constexpr double zsinis = 0.39785416;
    constexpr double zcosis = 0.91744867;
    constexpr double zcosgs = 0.1945905;
    constexpr double zsings = -0.98088458;

    rec.nm = np;
    rec.em = ep;
    rec.snodm = sin( nodep );
    rec.cnodm = cos( nodep );
    rec.sinomm = sin( argpp );
    rec.cosomm = cos( argpp );
    rec.sinim = sin( inclp );
    rec.cosim = cos( inclp );
    rec.emsq = rec.em * rec.em;
    T betasq = 1.0 - rec.emsq;
    rec.rtemsq = sqrt( betasq );

    rec.peo = T( 0.0 );
    rec.pinco = T( 0.0 );
    rec.plo = T( 0.0 );
    rec.pgho = T( 0.0 );
    rec.pho = T( 0.0 );
    rec.day = epoch + 18261.5 + tc / 1440.0;
    double xnodce = mod( 4.5236020 - 9.2422029e-4 * rec.day, twopi );
    double stem = std::sin( xnodce );
    double ctem = std::cos( xnodce );
    double zcosil = 0.91375164 - 0.03568096 * ctem;
    double zsinil = std::sqrt( 1.0 - zcosil * zcosil );
    double zsinhl = 0.089683511 * stem / zsinil;
    double zcoshl = std::sqrt( 1.0 - zsinhl * zsinhl );
    rec.gam = 5.8351514 + 0.0019443680 * rec.day;
    double zx = 0.39785416 * stem / zsinil;
    double zy = zcoshl * ctem + 0.91744867 * zsinhl * stem;
    zx = std::atan2( zx, zy );
    zx = rec.gam + zx - xnodce;
    double zcosgl = std::cos( zx );
    double zsingl = std::sin( zx );

    // ---- solar / lunar terms (two passes) ----
    double zcosg = zcosgs, zsing = zsings, zcosi = zcosis, zsini = zsinis;
    T zcosh = rec.cnodm;
    T zsinh = rec.snodm;
    double cc = c1ss;
    T xnoi = 1.0 / rec.nm;

    for ( int lsflg = 1; lsflg <= 2; ++lsflg )
    {
        T a1 = zcosg * zcosh + zsing * zcosi * zsinh;
        T a3 = -zsing * zcosh + zcosg * zcosi * zsinh;
        T a7 = -zcosg * zsinh + zsing * zcosi * zcosh;
        T a8 = zsing * zsini;
        T a9 = zsing * zsinh + zcosg * zcosi * zcosh;
        T a10 = zcosg * zsini;
        T a2 = rec.cosim * a7 + rec.sinim * a8;
        T a4 = rec.cosim * a9 + rec.sinim * a10;
        T a5 = -rec.sinim * a7 + rec.cosim * a8;
        T a6 = -rec.sinim * a9 + rec.cosim * a10;

        T x1 = a1 * rec.cosomm + a2 * rec.sinomm;
        T x2 = a3 * rec.cosomm + a4 * rec.sinomm;
        T x3 = -a1 * rec.sinomm + a2 * rec.cosomm;
        T x4 = -a3 * rec.sinomm + a4 * rec.cosomm;
        T x5 = a5 * rec.sinomm;
        T x6 = a6 * rec.sinomm;
        T x7 = a5 * rec.cosomm;
        T x8 = a6 * rec.cosomm;

        rec.z31 = 12.0 * x1 * x1 - 3.0 * x3 * x3;
        rec.z32 = 24.0 * x1 * x2 - 6.0 * x3 * x4;
        rec.z33 = 12.0 * x2 * x2 - 3.0 * x4 * x4;
        rec.z1 = 3.0 * ( a1 * a1 + a2 * a2 ) + rec.z31 * rec.emsq;
        rec.z2 = 6.0 * ( a1 * a3 + a2 * a4 ) + rec.z32 * rec.emsq;
        rec.z3 = 3.0 * ( a3 * a3 + a4 * a4 ) + rec.z33 * rec.emsq;
        rec.z11 = -6.0 * a1 * a5 + rec.emsq * ( -24.0 * x1 * x7 - 6.0 * x3 * x5 );
        rec.z12 = -6.0 * ( a1 * a6 + a3 * a5 ) +
                  rec.emsq * ( -24.0 * ( x2 * x7 + x1 * x8 ) - 6.0 * ( x3 * x6 + x4 * x5 ) );
        rec.z13 = -6.0 * a3 * a6 + rec.emsq * ( -24.0 * x2 * x8 - 6.0 * x4 * x6 );
        rec.z21 = 6.0 * a2 * a5 + rec.emsq * ( 24.0 * x1 * x5 - 6.0 * x3 * x7 );
        rec.z22 = 6.0 * ( a4 * a5 + a2 * a6 ) +
                  rec.emsq * ( 24.0 * ( x2 * x5 + x1 * x6 ) - 6.0 * ( x4 * x7 + x3 * x8 ) );
        rec.z23 = 6.0 * a4 * a6 + rec.emsq * ( 24.0 * x2 * x6 - 6.0 * x4 * x8 );
        rec.z1 = rec.z1 + rec.z1 + betasq * rec.z31;
        rec.z2 = rec.z2 + rec.z2 + betasq * rec.z32;
        rec.z3 = rec.z3 + rec.z3 + betasq * rec.z33;
        rec.s3 = cc * xnoi;
        rec.s2 = -0.5 * rec.s3 / rec.rtemsq;
        rec.s4 = rec.s3 * rec.rtemsq;
        rec.s1 = -15.0 * rec.em * rec.s4;
        rec.s5 = x1 * x3 + x2 * x4;
        rec.s6 = x2 * x3 + x1 * x4;
        rec.s7 = x2 * x4 - x1 * x3;

        if ( lsflg == 1 )
        {
            rec.ss1 = rec.s1;
            rec.ss2 = rec.s2;
            rec.ss3 = rec.s3;
            rec.ss4 = rec.s4;
            rec.ss5 = rec.s5;
            rec.ss6 = rec.s6;
            rec.ss7 = rec.s7;
            rec.sz1 = rec.z1;
            rec.sz2 = rec.z2;
            rec.sz3 = rec.z3;
            rec.sz11 = rec.z11;
            rec.sz12 = rec.z12;
            rec.sz13 = rec.z13;
            rec.sz21 = rec.z21;
            rec.sz22 = rec.z22;
            rec.sz23 = rec.z23;
            rec.sz31 = rec.z31;
            rec.sz32 = rec.z32;
            rec.sz33 = rec.z33;
            zcosg = zcosgl;
            zsing = zsingl;
            zcosi = zcosil;
            zsini = zsinil;
            zcosh = zcoshl * rec.cnodm + zsinhl * rec.snodm;
            zsinh = rec.snodm * zcoshl - rec.cnodm * zsinhl;
            cc = c1l;
        }
    }

    rec.zmol = mod( 4.7199672 + 0.22997150 * rec.day - rec.gam, twopi );
    rec.zmos = mod( 6.2565837 + 0.017201977 * rec.day, twopi );

    // ---- solar terms ----
    rec.se2 = 2.0 * rec.ss1 * rec.ss6;
    rec.se3 = 2.0 * rec.ss1 * rec.ss7;
    rec.si2 = 2.0 * rec.ss2 * rec.sz12;
    rec.si3 = 2.0 * rec.ss2 * ( rec.sz13 - rec.sz11 );
    rec.sl2 = -2.0 * rec.ss3 * rec.sz2;
    rec.sl3 = -2.0 * rec.ss3 * ( rec.sz3 - rec.sz1 );
    rec.sl4 = -2.0 * rec.ss3 * ( -21.0 - 9.0 * rec.emsq ) * zes;
    rec.sgh2 = 2.0 * rec.ss4 * rec.sz32;
    rec.sgh3 = 2.0 * rec.ss4 * ( rec.sz33 - rec.sz31 );
    rec.sgh4 = -18.0 * rec.ss4 * zes;
    rec.sh2 = -2.0 * rec.ss2 * rec.sz22;
    rec.sh3 = -2.0 * rec.ss2 * ( rec.sz23 - rec.sz21 );

    // ---- lunar terms ----
    rec.ee2 = 2.0 * rec.s1 * rec.s6;
    rec.e3 = 2.0 * rec.s1 * rec.s7;
    rec.xi2 = 2.0 * rec.s2 * rec.z12;
    rec.xi3 = 2.0 * rec.s2 * ( rec.z13 - rec.z11 );
    rec.xl2 = -2.0 * rec.s3 * rec.z2;
    rec.xl3 = -2.0 * rec.s3 * ( rec.z3 - rec.z1 );
    rec.xl4 = -2.0 * rec.s3 * ( -21.0 - 9.0 * rec.emsq ) * zel;
    rec.xgh2 = 2.0 * rec.s4 * rec.z32;
    rec.xgh3 = 2.0 * rec.s4 * ( rec.z33 - rec.z31 );
    rec.xgh4 = -18.0 * rec.s4 * zel;
    rec.xh2 = -2.0 * rec.s2 * rec.z22;
    rec.xh3 = -2.0 * rec.s2 * ( rec.z23 - rec.z21 );
}

// ---------------------------------------------------------------------------
// dsinit — deep-space contributions to mean-motion-dot from geopotential
// resonance with half-day and one-day orbits, plus resonance setup.
// ---------------------------------------------------------------------------
template < class T >
void dsinit( ElsetRec< T >& rec, double tc, const T& xpidot )
{
    using std::pow;
    using std::sin;

    constexpr double q22 = 1.7891679e-6;
    constexpr double q31 = 2.1460748e-6;
    constexpr double q33 = 2.2123015e-7;
    constexpr double root22 = 1.7891679e-6;
    constexpr double root44 = 7.3636953e-9;
    constexpr double root54 = 2.1765803e-9;
    constexpr double rptim = 4.37526908801129966e-3;
    constexpr double root32 = 3.7393792e-7;
    constexpr double root52 = 1.1428639e-7;
    constexpr double x2o3 = 2.0 / 3.0;
    constexpr double znl = 1.5835218e-4;
    constexpr double zns = 1.19459e-5;

    // ---- deep-space initialisation ----
    rec.irez = 0;
    if ( ( cst( rec.nm ) < 0.0052359877 ) && ( cst( rec.nm ) > 0.0034906585 ) ) rec.irez = 1;
    if ( ( cst( rec.nm ) >= 8.26e-3 ) && ( cst( rec.nm ) <= 9.24e-3 ) && ( cst( rec.em ) >= 0.5 ) )
        rec.irez = 2;

    // ---- solar terms ----
    T ses = rec.ss1 * zns * rec.ss5;
    T sis = rec.ss2 * zns * ( rec.sz11 + rec.sz13 );
    T sls = -zns * rec.ss3 * ( rec.sz1 + rec.sz3 - 14.0 - 6.0 * rec.emsq );
    T sghs = rec.ss4 * zns * ( rec.sz31 + rec.sz33 - 6.0 );
    T shs = -zns * rec.ss2 * ( rec.sz21 + rec.sz23 );
    // sgp4fix for 180 deg incl
    if ( ( cst( rec.inclm ) < 5.2359877e-2 ) || ( cst( rec.inclm ) > pi - 5.2359877e-2 ) )
        shs = T( 0.0 );
    if ( cst( rec.sinim ) != 0.0 ) shs = shs / rec.sinim;
    T sgs = sghs - rec.cosim * shs;

    // ---- lunar terms ----
    rec.dedt = ses + rec.s1 * znl * rec.s5;
    rec.didt = sis + rec.s2 * znl * ( rec.z11 + rec.z13 );
    rec.dmdt = sls - znl * rec.s3 * ( rec.z1 + rec.z3 - 14.0 - 6.0 * rec.emsq );
    T sghl = rec.s4 * znl * ( rec.z31 + rec.z33 - 6.0 );
    T shll = -znl * rec.s2 * ( rec.z21 + rec.z23 );
    if ( ( cst( rec.inclm ) < 5.2359877e-2 ) || ( cst( rec.inclm ) > pi - 5.2359877e-2 ) )
        shll = T( 0.0 );
    rec.domdt = sgs + sghl;
    rec.dnodt = shs;
    if ( cst( rec.sinim ) != 0.0 )
    {
        rec.domdt = rec.domdt - rec.cosim / rec.sinim * shll;
        rec.dnodt = rec.dnodt + shll / rec.sinim;
    }

    // ---- deep-space resonance effects ----
    rec.dndt = T( 0.0 );
    double theta = mod( rec.gsto + tc * rptim, twopi );
    rec.em = rec.em + rec.dedt * rec.t;
    rec.inclm = rec.inclm + rec.didt * rec.t;
    rec.argpm = rec.argpm + rec.domdt * rec.t;
    rec.nodem = rec.nodem + rec.dnodt * rec.t;
    rec.mm = rec.mm + rec.dmdt * rec.t;

    if ( rec.irez != 0 )
    {
        T aonv = pow( rec.nm / rec.xke, x2o3 );

        // ---- geopotential resonance for 12-hour orbits ----
        if ( rec.irez == 2 )
        {
            T cosisq = rec.cosim * rec.cosim;
            T emo = rec.em;
            rec.em = rec.ecco;
            T emsqo = rec.emsq;
            rec.emsq = rec.eccsq;
            T eoc = rec.em * rec.emsq;
            T g201 = -0.306 - ( rec.em - 0.64 ) * 0.440;

            T g211, g310, g322, g410, g422, g520, g521, g532, g533;
            if ( cst( rec.em ) <= 0.65 )
            {
                g211 = 3.616 - 13.2470 * rec.em + 16.2900 * rec.emsq;
                g310 = -19.302 + 117.3900 * rec.em - 228.4190 * rec.emsq + 156.5910 * eoc;
                g322 = -18.9068 + 109.7927 * rec.em - 214.6334 * rec.emsq + 146.5816 * eoc;
                g410 = -41.122 + 242.6940 * rec.em - 471.0940 * rec.emsq + 313.9530 * eoc;
                g422 = -146.407 + 841.8800 * rec.em - 1629.014 * rec.emsq + 1083.4350 * eoc;
                g520 = -532.114 + 3017.977 * rec.em - 5740.032 * rec.emsq + 3708.2760 * eoc;
            } else
            {
                g211 = -72.099 + 331.819 * rec.em - 508.738 * rec.emsq + 266.724 * eoc;
                g310 = -346.844 + 1582.851 * rec.em - 2415.925 * rec.emsq + 1246.113 * eoc;
                g322 = -342.585 + 1554.908 * rec.em - 2366.899 * rec.emsq + 1215.972 * eoc;
                g410 = -1052.797 + 4758.686 * rec.em - 7193.992 * rec.emsq + 3651.957 * eoc;
                g422 = -3581.690 + 16178.110 * rec.em - 24462.770 * rec.emsq + 12422.520 * eoc;
                if ( cst( rec.em ) > 0.715 )
                    g520 = -5149.66 + 29936.92 * rec.em - 54087.36 * rec.emsq + 31324.56 * eoc;
                else
                    g520 = 1464.74 - 4664.75 * rec.em + 3763.64 * rec.emsq;
            }
            if ( cst( rec.em ) < 0.7 )
            {
                g533 = -919.22770 + 4988.6100 * rec.em - 9064.7700 * rec.emsq + 5542.21 * eoc;
                g521 = -822.71072 + 4568.6173 * rec.em - 8491.4146 * rec.emsq + 5337.524 * eoc;
                g532 = -853.66600 + 4690.2500 * rec.em - 8624.7700 * rec.emsq + 5341.4 * eoc;
            } else
            {
                g533 = -37995.780 + 161616.52 * rec.em - 229838.20 * rec.emsq + 109377.94 * eoc;
                g521 = -51752.104 + 218913.95 * rec.em - 309468.16 * rec.emsq + 146349.42 * eoc;
                g532 = -40023.880 + 170470.89 * rec.em - 242699.48 * rec.emsq + 115605.82 * eoc;
            }

            T sini2 = rec.sinim * rec.sinim;
            T f220 = 0.75 * ( 1.0 + 2.0 * rec.cosim + cosisq );
            T f221 = 1.5 * sini2;
            T f321 = 1.875 * rec.sinim * ( 1.0 - 2.0 * rec.cosim - 3.0 * cosisq );
            T f322 = -1.875 * rec.sinim * ( 1.0 + 2.0 * rec.cosim - 3.0 * cosisq );
            T f441 = 35.0 * sini2 * f220;
            T f442 = 39.3750 * sini2 * sini2;
            T f522 = 9.84375 * rec.sinim *
                     ( sini2 * ( 1.0 - 2.0 * rec.cosim - 5.0 * cosisq ) +
                       0.33333333 * ( -2.0 + 4.0 * rec.cosim + 6.0 * cosisq ) );
            T f523 = rec.sinim * ( 4.92187512 * sini2 * ( -2.0 - 4.0 * rec.cosim + 10.0 * cosisq ) +
                                   6.56250012 * ( 1.0 + 2.0 * rec.cosim - 3.0 * cosisq ) );
            T f542 =
                29.53125 * rec.sinim *
                ( 2.0 - 8.0 * rec.cosim + cosisq * ( -12.0 + 8.0 * rec.cosim + 10.0 * cosisq ) );
            T f543 =
                29.53125 * rec.sinim *
                ( -2.0 - 8.0 * rec.cosim + cosisq * ( 12.0 + 8.0 * rec.cosim - 10.0 * cosisq ) );
            T xno2 = rec.nm * rec.nm;
            T ainv2 = aonv * aonv;
            T temp1 = 3.0 * xno2 * ainv2;
            T temp = temp1 * root22;
            rec.d2201 = temp * f220 * g201;
            rec.d2211 = temp * f221 * g211;
            temp1 = temp1 * aonv;
            temp = temp1 * root32;
            rec.d3210 = temp * f321 * g310;
            rec.d3222 = temp * f322 * g322;
            temp1 = temp1 * aonv;
            temp = 2.0 * temp1 * root44;
            rec.d4410 = temp * f441 * g410;
            rec.d4422 = temp * f442 * g422;
            temp1 = temp1 * aonv;
            temp = temp1 * root52;
            rec.d5220 = temp * f522 * g520;
            rec.d5232 = temp * f523 * g532;
            temp = 2.0 * temp1 * root54;
            rec.d5421 = temp * f542 * g521;
            rec.d5433 = temp * f543 * g533;
            rec.xlamo = mod( rec.mo + rec.nodeo + rec.nodeo - theta - theta, twopi );
            rec.xfact =
                rec.mdot + rec.dmdt + 2.0 * ( rec.nodedot + rec.dnodt - rptim ) - rec.no_unkozai;
            rec.em = emo;
            rec.emsq = emsqo;
        }

        // ---- synchronous resonance terms ----
        if ( rec.irez == 1 )
        {
            T g200 = 1.0 + rec.emsq * ( -2.5 + 0.8125 * rec.emsq );
            T g310 = 1.0 + 2.0 * rec.emsq;
            T g300 = 1.0 + rec.emsq * ( -6.0 + 6.60937 * rec.emsq );
            T f220 = 0.75 * ( 1.0 + rec.cosim ) * ( 1.0 + rec.cosim );
            T f311 = 0.9375 * rec.sinim * rec.sinim * ( 1.0 + 3.0 * rec.cosim ) -
                     0.75 * ( 1.0 + rec.cosim );
            T f330 = 1.0 + rec.cosim;
            f330 = 1.875 * f330 * f330 * f330;
            rec.del1 = 3.0 * rec.nm * rec.nm * aonv * aonv;
            rec.del2 = 2.0 * rec.del1 * f220 * g200 * q22;
            rec.del3 = 3.0 * rec.del1 * f330 * g300 * q33 * aonv;
            rec.del1 = rec.del1 * f311 * g310 * q31 * aonv;
            rec.xlamo = mod( rec.mo + rec.nodeo + rec.argpo - theta, twopi );
            rec.xfact =
                rec.mdot + xpidot - rptim + rec.dmdt + rec.domdt + rec.dnodt - rec.no_unkozai;
        }

        // ---- initialise the integrator ----
        rec.xli = rec.xlamo;
        rec.xni = rec.no_unkozai;
        rec.atime = 0.0;
        rec.nm = rec.no_unkozai + rec.dndt;
    }
}

// ---------------------------------------------------------------------------
// dspace — secular deep-space resonance integration (Euler-Maclaurin).
// ---------------------------------------------------------------------------
template < class T >
void dspace( ElsetRec< T >& rec, double tc )
{
    using std::cos;
    using std::sin;

    constexpr double fasx2 = 0.13130908;
    constexpr double fasx4 = 2.8843198;
    constexpr double fasx6 = 0.37448087;
    constexpr double g22 = 5.7686396;
    constexpr double g32 = 0.95240898;
    constexpr double g44 = 1.8014998;
    constexpr double g52 = 1.0508330;
    constexpr double g54 = 4.4108898;
    constexpr double rptim = 4.37526908801129966e-3;
    constexpr double stepp = 720.0;
    constexpr double stepn = -720.0;
    constexpr double step2 = 259200.0;

    T xndt = T( 0.0 );
    T xnddt = T( 0.0 );
    T xldot = T( 0.0 );

    rec.dndt = T( 0.0 );
    double theta = mod( rec.gsto + tc * rptim, twopi );
    rec.em = rec.em + rec.dedt * rec.t;
    rec.inclm = rec.inclm + rec.didt * rec.t;
    rec.argpm = rec.argpm + rec.domdt * rec.t;
    rec.nodem = rec.nodem + rec.dnodt * rec.t;
    rec.mm = rec.mm + rec.dmdt * rec.t;

    double ft = 0.0;
    if ( rec.irez != 0 )
    {
        // sgp4fix streamline epoch-restart check
        if ( ( rec.atime == 0.0 ) || ( rec.t * rec.atime <= 0.0 ) ||
             ( std::abs( rec.t ) < std::abs( rec.atime ) ) )
        {
            rec.atime = 0.0;
            rec.xni = rec.no_unkozai;
            rec.xli = rec.xlamo;
        }
        double delt = ( rec.t > 0.0 ) ? stepp : stepn;

        int iretn = 381;
        while ( iretn == 381 )
        {
            // ---- dot terms ----
            if ( rec.irez != 2 )
            {
                // near-synchronous resonance terms
                xndt = rec.del1 * sin( rec.xli - fasx2 ) +
                       rec.del2 * sin( 2.0 * ( rec.xli - fasx4 ) ) +
                       rec.del3 * sin( 3.0 * ( rec.xli - fasx6 ) );
                xldot = rec.xni + rec.xfact;
                xnddt = rec.del1 * cos( rec.xli - fasx2 ) +
                        2.0 * rec.del2 * cos( 2.0 * ( rec.xli - fasx4 ) ) +
                        3.0 * rec.del3 * cos( 3.0 * ( rec.xli - fasx6 ) );
                xnddt = xnddt * xldot;
            } else
            {
                // near-half-day resonance terms
                T xomi = rec.argpo + rec.argpdot * rec.atime;
                T x2omi = xomi + xomi;
                T x2li = rec.xli + rec.xli;
                xndt = rec.d2201 * sin( x2omi + rec.xli - g22 ) + rec.d2211 * sin( rec.xli - g22 ) +
                       rec.d3210 * sin( xomi + rec.xli - g32 ) +
                       rec.d3222 * sin( -xomi + rec.xli - g32 ) +
                       rec.d4410 * sin( x2omi + x2li - g44 ) + rec.d4422 * sin( x2li - g44 ) +
                       rec.d5220 * sin( xomi + rec.xli - g52 ) +
                       rec.d5232 * sin( -xomi + rec.xli - g52 ) +
                       rec.d5421 * sin( xomi + x2li - g54 ) + rec.d5433 * sin( -xomi + x2li - g54 );
                xldot = rec.xni + rec.xfact;
                xnddt =
                    rec.d2201 * cos( x2omi + rec.xli - g22 ) + rec.d2211 * cos( rec.xli - g22 ) +
                    rec.d3210 * cos( xomi + rec.xli - g32 ) +
                    rec.d3222 * cos( -xomi + rec.xli - g32 ) +
                    rec.d5220 * cos( xomi + rec.xli - g52 ) +
                    rec.d5232 * cos( -xomi + rec.xli - g52 ) +
                    2.0 * ( rec.d4410 * cos( x2omi + x2li - g44 ) + rec.d4422 * cos( x2li - g44 ) +
                            rec.d5421 * cos( xomi + x2li - g54 ) +
                            rec.d5433 * cos( -xomi + x2li - g54 ) );
                xnddt = xnddt * xldot;
            }

            // ---- integrator ----
            if ( std::abs( rec.t - rec.atime ) >= stepp )
            {
                iretn = 381;
            } else
            {
                ft = rec.t - rec.atime;
                iretn = 0;
            }

            if ( iretn == 381 )
            {
                rec.xli = rec.xli + xldot * delt + xndt * step2;
                rec.xni = rec.xni + xndt * delt + xnddt * step2;
                rec.atime = rec.atime + delt;
            }
        }

        rec.nm = rec.xni + xndt * ft + xnddt * ft * ft * 0.5;
        T xl = rec.xli + xldot * ft + xndt * ft * ft * 0.5;
        if ( rec.irez != 1 )
            rec.mm = xl - 2.0 * rec.nodem + 2.0 * theta;
        else
            rec.mm = xl - rec.nodem - rec.argpm + theta;
        rec.dndt = rec.nm - rec.no_unkozai;
        rec.nm = rec.no_unkozai + rec.dndt;
    }
}

}  // namespace tax::sgp4::detail
