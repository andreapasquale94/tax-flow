// include/tax/sgp4/elset_rec.hpp
//
// ElsetRec<T> — the SGP4 "satrec": the mean elements plus every derived
// constant the propagator carries between init and evaluation.  Templated on
// the scalar T so the whole propagator works for `double` (ordinary
// ephemerides) or `tax::TaylorExpansion` (a polynomial map of the state w.r.t.
// the seeded elements).
//
// Field-type convention:
//   * T      — anything that depends (even transitively) on the orbital
//              elements: the mean-element inputs and all derived/working
//              quantities.  These carry the Taylor derivatives.
//   * double — quantities that depend only on the epoch or are pure
//              constants: the gravity model, epoch Julian date, sidereal
//              time `gsto`, the lunar-solar epoch terms day/gam/zmol/zmos,
//              the resonance integrator clock `atime`, and `t` (= tsince).
//   * int/char — flags and error/method codes.

#pragma once

#include <tax/sgp4/gravconst.hpp>

namespace tax::sgp4
{

template < class T >
struct ElsetRec
{
    // ---- configuration / status ----
    GravModel whichconst = GravModel::Wgs72;
    int error = 0;
    char operationmode = 'i';  ///< 'a' = afspc, 'i' = improved
    char init = 'n';
    char method = 'n';  ///< 'n' = near-earth, 'd' = deep-space
    int isimp = 0;
    int epochyr = 0;

    // ---- gravity constants (pure double; see gravConstants()) ----
    double tumin = 0.0;
    double mu = 0.0;
    double radiusearthkm = 0.0;
    double xke = 0.0;
    double j2 = 0.0;
    double j3 = 0.0;
    double j4 = 0.0;
    double j3oj2 = 0.0;

    // ---- epoch (double; element-independent) ----
    double jdsatepoch = 0.0;
    double jdsatepochF = 0.0;
    double epochdays = 0.0;
    double gsto = 0.0;
    double t = 0.0;  ///< current tsince [min]

    // ---- mean-element inputs (T; the expandable parameters) ----
    T bstar{};
    T ndot{};
    T nddot{};
    T inclo{};
    T nodeo{};
    T ecco{};
    T argpo{};
    T mo{};
    T no_kozai{};
    T no_unkozai{};

    // ---- singly-averaged mean elements ----
    T a{}, altp{}, alta{};
    T am{}, em{}, im{}, Om{}, om{}, mm{}, nm{};

    // ---- temporary perturbed elements ----
    T ep{}, inclp{}, nodep{}, argpp{}, mp{};

    // ---- near-earth secular / periodic coefficients ----
    T aycof{}, con41{}, con42{};
    T cc1{}, cc4{}, cc5{};
    T d2{}, d3{}, d4{};
    T delmo{}, eta{}, argpdot{}, omgcof{}, sinmao{};
    T t2cof{}, t3cof{}, t4cof{}, t5cof{};
    T x1mth2{}, x7thm1{};
    T mdot{}, nodedot{}, xlcof{}, xmcof{}, nodecf{};

    // ---- deep-space resonance ----
    int irez = 0;
    double atime = 0.0;
    T d2201{}, d2211{}, d3210{}, d3222{}, d4410{}, d4422{};
    T d5220{}, d5232{}, d5421{}, d5433{};
    T dedt{}, didt{}, dmdt{}, dnodt{}, domdt{};
    T del1{}, del2{}, del3{};
    T xfact{}, xlamo{}, xli{}, xni{}, dndt{};

    // ---- deep-space lunar-solar periodics ----
    T e3{}, ee2{};
    T peo{}, pgho{}, pho{}, pinco{}, plo{};
    T se2{}, se3{}, sgh2{}, sgh3{}, sgh4{};
    T sh2{}, sh3{}, si2{}, si3{}, sl2{}, sl3{}, sl4{};
    T xgh2{}, xgh3{}, xgh4{}, xh2{}, xh3{};
    T xi2{}, xi3{}, xl2{}, xl3{}, xl4{};

    // epoch-only lunar-solar terms (double)
    double day = 0.0, gam = 0.0, zmol = 0.0, zmos = 0.0;

    // ---- dscom working set ----
    T snodm{}, cnodm{}, sinim{}, cosim{}, sinomm{}, cosomm{};
    T emsq{}, eccsq{}, rtemsq{};
    T s1{}, s2{}, s3{}, s4{}, s5{}, s6{}, s7{};
    T ss1{}, ss2{}, ss3{}, ss4{}, ss5{}, ss6{}, ss7{};
    T sz1{}, sz2{}, sz3{}, sz11{}, sz12{}, sz13{};
    T sz21{}, sz22{}, sz23{}, sz31{}, sz32{}, sz33{};
    T z1{}, z2{}, z3{}, z11{}, z12{}, z13{};
    T z21{}, z22{}, z23{}, z31{}, z32{}, z33{};
    T argpm{}, inclm{}, nodem{};

    // ---- initl outputs ----
    T ainv{}, ao{}, cosio{}, cosio2{}, omeosq{}, posq{}, rp{}, rteosq{}, sinio{};

    /// Load the gravity constants for `model` into the scalar fields.
    void setGrav( GravModel model ) noexcept
    {
        whichconst = model;
        const GravConstants g = gravConstants( model );
        tumin = g.tumin;
        mu = g.mu;
        radiusearthkm = g.radiusearthkm;
        xke = g.xke;
        j2 = g.j2;
        j3 = g.j3;
        j4 = g.j4;
        j3oj2 = g.j3oj2;
    }
};

}  // namespace tax::sgp4
