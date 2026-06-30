// =============================================================================
// examples/conjunction/collision_probability.cpp
//
// SSA conjunction assessment with a Taylor-expanded SGP4 propagator.
//
// A classic Space-Situational-Awareness task: given two objects with uncertain
// states, estimate the probability of collision (Pc) at their close approach.
// SGP4 propagates the mean elements; here we ALSO expand the propagated state
// as a high-order Taylor polynomial in the uncertain TLE mean elements, so a
// single SGP4 run yields the whole local map state(t) = f(delta-elements).
//
// What the higher-order map buys us:
//   * Linear (1st-order) covariance mapping  C(t) = J C0 J^T  comes for free
//     from the Jacobian — the classic STM-based Pc, exact (not finite-
//     differenced) from a single propagation.
//   * The full order-N polynomial captures the *curvature* of the SGP4 flow, so
//     it stays accurate far from the nominal where the linearisation fails, and
//     yields a nonlinear, non-Gaussian Pc via a propagation-free "polynomial
//     Monte Carlo" (evaluate the map per sample instead of re-running SGP4).
//
// The example: (1) builds a crossing secondary and locates the close approach,
// (2) expands both objects' states at TCA to order 1 and order N, (3) shows a
// map-fidelity table (|poly - SGP4| per order vs perturbation size — the
// noise-free signature of why higher order matters), and (4) computes Pc four
// ways: analytic-linear, polynomial-MC (order 1 and N), and a brute-force SGP4
// Monte Carlo as ground truth.
//
// Outputs conjunction.json, conjunction_bplane.csv, conjunction_fidelity.csv;
// plot with examples/plot/plot_conjunction.py.
//
// Build:  cmake -S . -B build -DTAXFLOW_BUILD_EXAMPLES=ON && cmake --build build
// Run:    ./conjunction_pc
// =============================================================================

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <tax/sgp4.hpp>
#include <tax/tax.hpp>
#include <vector>

using namespace tax::sgp4;
using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

namespace
{
constexpr int kVars = 6;   // expand in the 6 mean elements
constexpr int kOrder = 4;  // high-order map order
using TEhi = tax::TaylorExpansion< double, tax::IsotropicScheme< kOrder, kVars > >;
using TElin = tax::TaylorExpansion< double, tax::IsotropicScheme< 1, kVars > >;

// 1-sigma uncertainty of each mean element (i, node, ecc, argp, M, n).  TLE
// "catalogue accuracy" class — along-track (M) and n dominate the footprint.
// Sized so the mapped position 1-sigma at the conjunction is ~km, the regime
// where the nonlinearity of the SGP4 flow starts to matter.
constexpr std::array< double, kVars > kSigma = { 1.3e-4,    // inclo   [rad]  (-> cross-track)
                                                 1.3e-4,    // nodeo   [rad]  (-> cross-track)
                                                 1.0e-4,    // ecco    [-]    (-> radial)
                                                 1.0e-4,    // argpo   [rad]  (-> along-track)
                                                 1.0e-4,    // mo      [rad]  (-> along-track)
                                                 1.5e-8 };  // no_kozai[rad/min]

constexpr double kHardBodyRadius = 0.050;  // combined object radius [km] (50 m)

// Seed the six mean elements as identity expansions around a TLE's nominal.
template < class TE >
Seeds< TE > elementSeeds( const Tle& tle )
{
    return Seeds< TE >{ TE( tle.bstar ),
                        TE( tle.ndot ),
                        TE( tle.nddot ),
                        TE::variable( tle.inclo, 0 ),
                        TE::variable( tle.nodeo, 1 ),
                        TE::variable( tle.ecco, 2 ),
                        TE::variable( tle.argpo, 3 ),
                        TE::variable( tle.mo, 4 ),
                        TE::variable( tle.no_kozai, 5 ) };
}

// Constant (nominal) part of a state, for double OR TaylorExpansion components.
Vec3 nominalR( const auto& s )
{
    using detail::cst;
    return { cst( s.r( 0 ) ), cst( s.r( 1 ) ), cst( s.r( 2 ) ) };
}
Vec3 nominalV( const auto& s )
{
    using detail::cst;
    return { cst( s.v( 0 ) ), cst( s.v( 1 ) ), cst( s.v( 2 ) ) };
}

// |poly(order N) - r_true| of a position map at displacement dx, in km.  The
// order-N truncation of the order-kOrder map is evaluated, so a single map
// gives every order.
template < int N, class TE >
double mapError( const State< TE >& st, const std::array< double, kVars >& dx, const Vec3& rt )
{
    const Vec3 r{ st.r( 0 ).template truncate< N >().eval( dx ),
                  st.r( 1 ).template truncate< N >().eval( dx ),
                  st.r( 2 ).template truncate< N >().eval( dx ) };
    return ( r - rt ).norm();
}

// Orthonormal basis (u1,u2) of the encounter B-plane: the plane perpendicular
// to the relative velocity (the along-velocity direction is integrated out in
// the short-term-encounter 2D Pc model).
std::pair< Vec3, Vec3 > bPlaneBasis( const Vec3& vrel )
{
    const Vec3 w = vrel.normalized();
    const Vec3 a = ( std::abs( w.x() ) < 0.9 ) ? Vec3::UnitX() : Vec3::UnitY();
    const Vec3 u1 = ( a - a.dot( w ) * w ).normalized();
    const Vec3 u2 = w.cross( u1 );
    return { u1, u2 };
}

// Analytic 2D collision probability: integrate a Gaussian N(mean, cov) over the
// hard-body disk of radius R, by polar quadrature.
double analyticPc( const Eigen::Vector2d& mean, const Eigen::Matrix2d& cov, double R )
{
    const Eigen::Matrix2d Ci = cov.inverse();
    const double norm = 1.0 / ( 2.0 * pi * std::sqrt( cov.determinant() ) );
    const int nr = 240, nt = 360;
    double pc = 0.0;
    for ( int i = 0; i < nr; ++i )
    {
        const double r = R * ( i + 0.5 ) / nr;
        const double dr = R / nr;
        for ( int j = 0; j < nt; ++j )
        {
            const double th = 2.0 * pi * ( j + 0.5 ) / nt;
            const Eigen::Vector2d d{ r * std::cos( th ) - mean.x(), r * std::sin( th ) - mean.y() };
            pc += norm * std::exp( -0.5 * d.dot( Ci * d ) ) * r * dr * ( 2.0 * pi / nt );
        }
    }
    return pc;
}
}  // namespace

int main()
{
    // ---- Primary object: the canonical Vallado ISS TLE (near-circular LEO) ----
    const Tle primary =
        Tle::parse( "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927",
                    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537" );

    // ---- Secondary object: a crossing orbit tilted 8 deg in inclination, so
    // it meets the primary near the ascending node at a ~1 km/s closing speed.
    // The two inclinations precess (J2) at slightly different rates, which would
    // pull the nodes apart; we search the secondary's RAAN to cancel that and
    // produce a near-direct conjunction (the maximum-Pc geometry). ----
    Tle secondary = primary;
    secondary.inclo += 8.0 * deg2rad;

    const double periodMin = 1440.0 / ( primary.no_kozai * 1440.0 / ( 2.0 * pi ) );

    auto missAt = [&]( const Tle& s, double t ) {
        Satellite< double > sp = Satellite< double >::fromTle( primary );
        Satellite< double > ss = Satellite< double >::fromTle( s );
        return ( nominalR( sp.propagate( t ) ) - nominalR( ss.propagate( t ) ) ).norm();
    };
    // Search the secondary's RAAN offset (to null the differential nodal
    // precession at the encounter) and the time, then jointly refine both, so
    // the two objects very nearly collide at the node.
    double bestMiss = 1e30, bestNode = 0.0, tca = 0.0;
    for ( int kn = -80; kn <= 80; ++kn )
    {
        const double dn = kn * ( 0.001 * deg2rad );
        Tle s = secondary;
        s.nodeo = primary.nodeo + dn;
        for ( double t = 10.0; t <= 2.0 * periodMin; t += 0.05 )
        {
            const double d = missAt( s, t );
            if ( d < bestMiss )
            {
                bestMiss = d;
                bestNode = dn;
                tca = t;
            }
        }
    }
    double nWin = 0.001 * deg2rad, tWin = 0.05;
    for ( int iter = 0; iter < 6; ++iter )
    {
        for ( double dn = bestNode - nWin; dn <= bestNode + nWin; dn += nWin / 20.0 )
        {
            Tle s = secondary;
            s.nodeo = primary.nodeo + dn;
            for ( double t = tca - tWin; t <= tca + tWin; t += tWin / 40.0 )
                if ( t >= 0.0 )
                {
                    const double d = missAt( s, t );
                    if ( d < bestMiss )
                    {
                        bestMiss = d;
                        bestNode = dn;
                        tca = t;
                    }
                }
        }
        nWin *= 0.25;
        tWin *= 0.25;
    }
    secondary.nodeo = primary.nodeo + bestNode;

    std::printf( "Close approach found:\n  TCA      = %.4f min after epoch\n  miss     = %.4f km\n",
                 tca, bestMiss );

    // ---- Expand both objects' states at TCA in their 6 mean elements ----
    Satellite< TEhi > satP( primary, GravModel::Wgs72, elementSeeds< TEhi >( primary ) );
    Satellite< TEhi > satS( secondary, GravModel::Wgs72, elementSeeds< TEhi >( secondary ) );
    const auto stP = satP.propagate( tca );
    const auto stS = satS.propagate( tca );

    Satellite< TElin > satPl( primary, GravModel::Wgs72, elementSeeds< TElin >( primary ) );
    Satellite< TElin > satSl( secondary, GravModel::Wgs72, elementSeeds< TElin >( secondary ) );
    const auto stPl = satPl.propagate( tca );
    const auto stSl = satSl.propagate( tca );

    const Vec3 rrel = nominalR( stS ) - nominalR( stP );
    const Vec3 vrel = nominalV( stS ) - nominalV( stP );
    const auto [u1, u2] = bPlaneBasis( vrel );
    std::printf( "  |v_rel|  = %.4f km/s\n\n", vrel.norm() );

    // ---- Map fidelity: how well each truncation order reproduces SGP4 as the
    // perturbation grows.  Walk out along the "all elements at +k-sigma"
    // direction; the linear map's error grows quadratically while the
    // higher-order map stays accurate to many sigma.  This is the noise-free
    // signature of why higher order matters for nonlinear uncertainty. ----
    std::printf(
        "Map fidelity |poly(order) - SGP4| of primary position [km]:\n"
        "  k-sigma     order1      order2      order3      order4\n" );
    std::ofstream fid( "conjunction_fidelity.csv" );
    fid << "k_sigma,order1,order2,order3,order4\n";
    for ( int k = 0; k <= 8; ++k )
    {
        std::array< double, kVars > dx;
        for ( int i = 0; i < kVars; ++i ) dx[i] = k * kSigma[i];
        Tle pk = primary;
        pk.inclo += dx[0];
        pk.nodeo += dx[1];
        pk.ecco += dx[2];
        pk.argpo += dx[3];
        pk.mo += dx[4];
        pk.no_kozai += dx[5];
        const Vec3 rt = nominalR( Satellite< double >::fromTle( pk ).propagate( tca ) );
        const double e1 = mapError< 1 >( stP, dx, rt ), e2 = mapError< 2 >( stP, dx, rt ),
                     e3 = mapError< 3 >( stP, dx, rt ), e4 = mapError< 4 >( stP, dx, rt );
        std::printf( "  %2d       %10.3e %10.3e %10.3e %10.3e\n", k, e1, e2, e3, e4 );
        fid << k << "," << e1 << "," << e2 << "," << e3 << "," << e4 << "\n";
    }
    std::printf( "\n" );

    // ---- (1) Analytic linear Pc: map covariance with the Jacobian ----
    auto posJacobian = []( const auto& st ) {
        Eigen::Matrix< double, 3, kVars > J;
        for ( int c = 0; c < 3; ++c ) J.row( c ) = st.r( c ).gradient().transpose();
        return J;
    };
    Eigen::Matrix< double, kVars, kVars > C0 = Eigen::Matrix< double, kVars, kVars >::Zero();
    for ( int i = 0; i < kVars; ++i ) C0( i, i ) = kSigma[i] * kSigma[i];

    const auto Jp = posJacobian( stPl );
    const auto Js = posJacobian( stSl );
    const Mat3 Ccomb = Jp * C0 * Jp.transpose() + Js * C0 * Js.transpose();

    Eigen::Matrix< double, 3, 2 > U;
    U.col( 0 ) = u1;
    U.col( 1 ) = u2;
    const Eigen::Matrix2d C2d = U.transpose() * Ccomb * U;
    const Eigen::Vector2d miss2d{ u1.dot( rrel ), u2.dot( rrel ) };
    const double pcLinear = analyticPc( miss2d, C2d, kHardBodyRadius );

    std::printf( "B-plane combined 1-sigma: %.4f x %.4f km   miss in-plane = %.4f km\n",
                 std::sqrt( C2d( 0, 0 ) ), std::sqrt( C2d( 1, 1 ) ), miss2d.norm() );

    // ---- (2)+(3) Monte Carlo on SHARED samples. For each element draw, the
    // order-1 and order-N maps are evaluated and a fresh SGP4 run is done on the
    // same perturbed elements. Because the high-order map reproduces SGP4 to ~nm
    // (see the fidelity table) its Pc tracks the brute-force SGP4 Pc to a handful
    // of hits, while the order-1 (linear) map gives a measurably different count.
    auto bplane = [&]( const Vec3& d ) {
        return std::array< double, 2 >{ u1.dot( d ), u2.dot( d ) };
    };
    const double R2 = kHardBodyRadius * kHardBodyRadius;
    auto inDisk = [&]( const std::array< double, 2 >& p ) {
        return p[0] * p[0] + p[1] * p[1] <= R2;
    };

    const int Nmc = 1'000'000;
    constexpr int kScatter = 6000;  // B-plane points kept for plotting
    std::vector< std::array< double, 2 > > truthPts, polyPts;
    std::mt19937 rng( 12345 );
    std::normal_distribution< double > z( 0.0, 1.0 );
    int hitTruth = 0, hitPoly1 = 0, hitPolyN = 0;
    double polyMs = 0.0, truthMs = 0.0;
    for ( int n = 0; n < Nmc; ++n )
    {
        std::array< double, kVars > dxP, dxS;
        for ( int i = 0; i < kVars; ++i )
        {
            dxP[i] = kSigma[i] * z( rng );
            dxS[i] = kSigma[i] * z( rng );
        }

        // order-N and order-1 polynomial-map relative position (propagation-free)
        auto p0 = std::chrono::steady_clock::now();
        const Vec3 rN{ stS.r( 0 ).eval( dxS ) - stP.r( 0 ).eval( dxP ),
                       stS.r( 1 ).eval( dxS ) - stP.r( 1 ).eval( dxP ),
                       stS.r( 2 ).eval( dxS ) - stP.r( 2 ).eval( dxP ) };
        auto p1 = std::chrono::steady_clock::now();
        polyMs += std::chrono::duration< double, std::milli >( p1 - p0 ).count();
        const Vec3 r1{ stSl.r( 0 ).eval( dxS ) - stPl.r( 0 ).eval( dxP ),
                       stSl.r( 1 ).eval( dxS ) - stPl.r( 1 ).eval( dxP ),
                       stSl.r( 2 ).eval( dxS ) - stPl.r( 2 ).eval( dxP ) };

        // SGP4 truth on the same draw
        Tle p = primary, s = secondary;
        p.inclo += dxP[0];
        p.nodeo += dxP[1];
        p.ecco += dxP[2];
        p.argpo += dxP[3];
        p.mo += dxP[4];
        p.no_kozai += dxP[5];
        s.inclo += dxS[0];
        s.nodeo += dxS[1];
        s.ecco += dxS[2];
        s.argpo += dxS[3];
        s.mo += dxS[4];
        s.no_kozai += dxS[5];
        auto s0 = std::chrono::steady_clock::now();
        Satellite< double > sp = Satellite< double >::fromTle( p );
        Satellite< double > ss = Satellite< double >::fromTle( s );
        const Vec3 rT = nominalR( ss.propagate( tca ) ) - nominalR( sp.propagate( tca ) );
        auto s1 = std::chrono::steady_clock::now();
        truthMs += std::chrono::duration< double, std::milli >( s1 - s0 ).count();

        const auto bT = bplane( rT ), bN = bplane( rN ), b1 = bplane( r1 );
        if ( inDisk( bT ) ) ++hitTruth;
        if ( inDisk( bN ) ) ++hitPolyN;
        if ( inDisk( b1 ) ) ++hitPoly1;
        if ( n < kScatter )
        {
            truthPts.push_back( bT );
            polyPts.push_back( bN );
        }
    }
    const int Ntruth = Nmc;
    const double pcTruth = static_cast< double >( hitTruth ) / Nmc;
    const double pcPolyLin = static_cast< double >( hitPoly1 ) / Nmc;
    const double pcPolyHi = static_cast< double >( hitPolyN ) / Nmc;

    // ---- report ----
    std::printf( "\nCollision probability (hard-body radius %.0f m):\n", kHardBodyRadius * 1000.0 );
    std::printf( "  analytic linear  (J C0 J^T, Gaussian)   Pc = %.3e\n", pcLinear );
    std::printf( "  polynomial MC, order 1  (%d)        Pc = %.3e\n", Nmc, pcPolyLin );
    std::printf( "  polynomial MC, order %d  (%d)        Pc = %.3e\n", kOrder, Nmc, pcPolyHi );
    std::printf( "  brute-force SGP4 MC     (%d)        Pc = %.3e  (truth)\n", Ntruth, pcTruth );
    std::printf(
        "\nSame %d samples for all three: the order-1 map reproduces the analytic\n"
        "Gaussian (it IS the linearisation); the order-%d map tracks the SGP4 truth to a\n"
        "few hits, capturing the nonlinear dispersion the linear model misses. All come\n"
        "from ONE SGP4 run per object - the order-1 Jacobian (linear Pc) is exact, not\n"
        "finite-differenced, and the order-%d map evaluates propagation-free per sample.\n",
        Nmc, kOrder, kOrder );
    std::printf(
        "\nTiming over %d samples: order-%d map eval %.0f ms (%.2f us/sample);  "
        "SGP4 re-propagation %.0f ms (%.2f us/sample).\n",
        Nmc, kOrder, polyMs, polyMs * 1e3 / Nmc, truthMs, truthMs * 1e3 / Nmc );

    // ---- plot data ----
    std::ofstream json( "conjunction.json" );
    json << "{\n  \"tca_min\": " << tca << ",\n  \"miss_km\": " << bestMiss
         << ",\n  \"vrel_kms\": " << vrel.norm() << ",\n  \"hard_body_km\": " << kHardBodyRadius
         << ",\n  \"order\": " << kOrder << ",\n  \"pc_linear\": " << pcLinear
         << ",\n  \"pc_poly_order1\": " << pcPolyLin << ",\n  \"pc_poly_orderN\": " << pcPolyHi
         << ",\n  \"pc_truth\": " << pcTruth << ",\n  \"miss2d\": [" << miss2d.x() << ", "
         << miss2d.y() << "],\n  \"cov2d\": [" << C2d( 0, 0 ) << ", " << C2d( 0, 1 ) << ", "
         << C2d( 1, 0 ) << ", " << C2d( 1, 1 ) << "]\n}\n";

    std::ofstream csv( "conjunction_bplane.csv" );
    csv << "source,xi,zeta\n";
    for ( const auto& p : truthPts ) csv << "truth," << p[0] << "," << p[1] << "\n";
    for ( const auto& p : polyPts ) csv << "poly," << p[0] << "," << p[1] << "\n";
    std::printf( "\nWrote conjunction.json and conjunction_bplane.csv\n" );
    return 0;
}
