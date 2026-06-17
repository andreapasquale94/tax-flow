// benchmarks/bench_ode_cr3bp.cpp
//
// Cross-method, cross-controller benchmark on the shared CR3BP
// propagation fixture (Task 21's tests/ode/cr3bp_problem.hpp).
//
// Reference combination: Fehlberg78 + I-controller at 1e-12 (classical
// baseline). Each benchmarked combination is timed and its endpoint
// deviation from the reference is reported in the `err` counter.
//
// Taylor is swept across orders {8, 10, 12, 16, 20, 24, 30} under
// JorbaZou, with PI and H211b additionally exercised at N = 12 and
// N = 24 for controller comparison.
//
// Build:
//   cmake -DTAX_BUILD_BENCHMARK=ON … && cmake --build … --target bench_ode_cr3bp
// Run:
//   ./bench_ode_cr3bp --benchmark_format=console

#include <benchmark/benchmark.h>

#include <tax/la/types.hpp>

#include "../tests/ode/problems/cr3bp_problem.hpp"
#include <tax/ode.hpp>

using namespace tax::ode::test;

namespace
{

CR3BPState g_reference;
bool       g_reference_ready = false;

void ensure_reference()
{
    if ( g_reference_ready ) return;
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-14;

    using State = tax::la::VecNT< 4, double >;
    using Stepper = tax::ode::Fehlberg78Stepper<
        State, tax::ode::controllers::I< double > >;
    auto rhs = cr3bp_rhs();
    using F = decltype( rhs );
    tax::ode::Integrator< Stepper, F, /*Dense=*/false > integ{
        std::move( rhs ), cfg };
    auto sol = integ.integrate( cr3bp_transit_ic(), 0.0, kCR3BPTFinal );
    g_reference = sol.x.back();
    g_reference_ready = true;
}

double endpoint_error( const CR3BPState& x )
{
    ensure_reference();
    return ( x - g_reference ).norm();
}

template < int N, class Controller = tax::ode::controllers::JorbaZou< double > >
CR3BPState run_taylor( double tol )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = tol;
    using State = tax::la::VecNT< 4, double >;
    using Stepper = tax::ode::TaylorStepper< N, State, Controller >;
    auto rhs = cr3bp_rhs();
    using F = decltype( rhs );
    tax::ode::Integrator< Stepper, F, false > integ{ std::move( rhs ), cfg };
    return integ.integrate( cr3bp_transit_ic(), 0.0, kCR3BPTFinal ).x.back();
}

template < class Stepper >
CR3BPState run_rk( double tol )
{
    tax::ode::IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = tol;
    auto rhs = cr3bp_rhs();
    using F = decltype( rhs );
    tax::ode::Integrator< Stepper, F, false > integ{ std::move( rhs ), cfg };
    return integ.integrate( cr3bp_transit_ic(), 0.0, kCR3BPTFinal ).x.back();
}

}  // namespace

// -------- Reference: Fehlberg78 + I @ 1e-12 (one row of the table) --------
static void BM_RefFehlberg78_I_1e12( benchmark::State& s )
{
    using St = tax::ode::Fehlberg78Stepper<
        tax::la::VecNT< 4, double >, tax::ode::controllers::I< double > >;
    for ( auto _ : s )
    {
        auto x = run_rk< St >( 1e-12 );
        benchmark::DoNotOptimize( x );
    }
    s.counters["err"] = endpoint_error( run_rk< St >( 1e-12 ) );
}
BENCHMARK( BM_RefFehlberg78_I_1e12 );

// -------- RK methods × controllers at tol = 1e-12 --------
#define BENCH_RK( name, Stepper, ControllerT, tol )                              \
    static void name( benchmark::State& s )                                       \
    {                                                                             \
        using S = tax::la::VecNT< 4, double >;                                  \
        using St = Stepper< S, tax::ode::controllers::ControllerT< double > >;    \
        for ( auto _ : s )                                                        \
        {                                                                         \
            auto x = run_rk< St >( tol );                                         \
            benchmark::DoNotOptimize( x );                                        \
        }                                                                         \
        s.counters["err"] = endpoint_error( run_rk< St >( tol ) );                \
    }                                                                             \
    BENCHMARK( name )

BENCH_RK( BM_Verner78_PI_1e12,    tax::ode::Verner78Stepper,   PI,    1e-12 );
BENCH_RK( BM_Verner89_PI_1e12,    tax::ode::Verner89Stepper,   PI,    1e-12 );
BENCH_RK( BM_Feagin12_PI_1e12,    tax::ode::Feagin12Stepper,   PI,    1e-12 );
BENCH_RK( BM_Feagin14_PI_1e12,    tax::ode::Feagin14Stepper,   PI,    1e-12 );
BENCH_RK( BM_Verner89_H211b_1e12, tax::ode::Verner89Stepper,   H211b, 1e-12 );
BENCH_RK( BM_Feagin14_H211b_1e12, tax::ode::Feagin14Stepper,   H211b, 1e-12 );

// -------- Taylor order sweep with JorbaZou (default Taylor controller) --------
#define BENCH_TAYLOR_JZ( name, N )                                               \
    static void name( benchmark::State& s )                                       \
    {                                                                             \
        for ( auto _ : s )                                                        \
        {                                                                         \
            auto x = run_taylor< N >( 1e-12 );                                    \
            benchmark::DoNotOptimize( x );                                        \
        }                                                                         \
        s.counters["err"] = endpoint_error( run_taylor< N >( 1e-12 ) );           \
    }                                                                             \
    BENCHMARK( name )

BENCH_TAYLOR_JZ( BM_TaylorJZ_N08,  8 );
BENCH_TAYLOR_JZ( BM_TaylorJZ_N10, 10 );
BENCH_TAYLOR_JZ( BM_TaylorJZ_N12, 12 );
BENCH_TAYLOR_JZ( BM_TaylorJZ_N16, 16 );
BENCH_TAYLOR_JZ( BM_TaylorJZ_N20, 20 );
BENCH_TAYLOR_JZ( BM_TaylorJZ_N24, 24 );
BENCH_TAYLOR_JZ( BM_TaylorJZ_N30, 30 );

// -------- Taylor with PI / H211b at representative orders --------
static void BM_TaylorPI_N12( benchmark::State& s )
{
    for ( auto _ : s )
    {
        auto x = run_taylor< 12, tax::ode::controllers::PI< double > >( 1e-12 );
        benchmark::DoNotOptimize( x );
    }
    s.counters["err"] = endpoint_error(
        run_taylor< 12, tax::ode::controllers::PI< double > >( 1e-12 ) );
}
BENCHMARK( BM_TaylorPI_N12 );

static void BM_TaylorH211b_N24( benchmark::State& s )
{
    for ( auto _ : s )
    {
        auto x = run_taylor< 24, tax::ode::controllers::H211b< double > >( 1e-12 );
        benchmark::DoNotOptimize( x );
    }
    s.counters["err"] = endpoint_error(
        run_taylor< 24, tax::ode::controllers::H211b< double > >( 1e-12 ) );
}
BENCHMARK( BM_TaylorH211b_N24 );

BENCHMARK_MAIN();
