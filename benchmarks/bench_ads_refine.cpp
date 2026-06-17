// =============================================================================
// benchmarks/bench_ads_refine.cpp
//
// Runtime comparison of the two ADS strategies on the same task — propagate a
// box of Kepler initial conditions over one orbit and partition it:
//
//   * classic in-flight ADS   (tax::ads::propagate; Truncation / NLI criteria)
//   * propagate-then-assess    (tax::ads::refine; CoefficientMatch / Volume)
//
// Three questions are benchmarked:
//
//  1. Thread scaling (arg 0 = driver num_threads, 1..4): classic ADS vs.
//     binary refine vs. aggressive 4-way refine (split_dirs = 2).
//  2. Aggressive vs. binary refinement — does splitting the top two axes at
//     once (2^2 = 4 children) beat bisecting one axis?
//  3. Classic ADS vs. polynomial order: Truncation and NLI at P = 2, 4, 6
//     (one thread), showing the order-vs-leaf-count trade-off.
//
// The "leaves" counter reports the partition size, since the criteria do not
// split to the same resolution — divide time by leaves for a per-box cost.
//
// Build:  cmake -S . -B build -DTAX_BUILD_BENCHMARK=ON && cmake --build build -j
// Run:    ./build/benchmarks/bench_ads_refine
// =============================================================================

#include <benchmark/benchmark.h>

#include <cmath>
#include <tax/ads.hpp>
#include <tax/ode.hpp>

namespace
{
using namespace tax::ode::methods;

// Planar Kepler problem (GM = 1), generic over scalar / DA state.
auto rhs()
{
    return []( const auto& s, const auto& ) {
        using S = std::decay_t< decltype( s ) >;
        const auto x = s( 0 );
        const auto y = s( 1 );
        const auto r2 = x * x + y * y;
        const auto r3 = r2 * sqrt( r2 );
        S o;
        o( 0 ) = s( 2 );
        o( 1 ) = s( 3 );
        o( 2 ) = -x / r3;
        o( 3 ) = -y / r3;
        return o;
    };
}

tax::la::VecNT< 4, double > center() { return { 0.5, 0.0, 0.0, std::sqrt( 3.0 ) }; }

// IC box varying y and vy (×3 the small tutorial box, so the partition is big
// enough to exercise the thread pool).
tax::ads::Box< double, 4 > icBox()
{
    constexpr double s = 3.0;
    return { center(), tax::la::VecNT< 4, double >{ 0.0, 8e-3 * s, 0.0, 2e-2 * s } };
}

tax::ode::IntegratorConfig< double > cfg()
{
    tax::ode::IntegratorConfig< double > c;
    c.abstol = c.reltol = 1e-12;
    return c;
}

constexpr double kTfinal = 2.0 * M_PI;
}  // namespace

// ---- 1) thread scaling -----------------------------------------------------

static void BM_ClassicAds( benchmark::State& state )
{
    const int threads = static_cast< int >( state.range( 0 ) );
    std::size_t leaves = 0;
    for ( auto _ : state )
    {
        auto tree =
            tax::ads::propagate< 6 >( Verner89{}, tax::ads::TruncationCriterion{ 1e-4, 6 }, rhs(),
                                      icBox(), center(), 0.0, kTfinal, cfg(), threads );
        benchmark::DoNotOptimize( tree );
        leaves = tree.done().size();
    }
    state.counters["leaves"] = static_cast< double >( leaves );
}
BENCHMARK( BM_ClassicAds )->DenseRange( 1, 4 )->Unit( benchmark::kMillisecond )->UseRealTime();

static void BM_RefineCoeff( benchmark::State& state )
{
    const int threads = static_cast< int >( state.range( 0 ) );
    std::size_t leaves = 0;
    for ( auto _ : state )
    {
        auto tree =
            tax::ads::refine< 6 >( Verner89{}, tax::ads::CoefficientMatchCriterion{ 1e-6, 6 },
                                   rhs(), icBox(), center(), 0.0, kTfinal, cfg(), threads,
                                   /*split_dirs=*/1 );
        benchmark::DoNotOptimize( tree );
        leaves = tree.done().size();
    }
    state.counters["leaves"] = static_cast< double >( leaves );
}
BENCHMARK( BM_RefineCoeff )->DenseRange( 1, 4 )->Unit( benchmark::kMillisecond )->UseRealTime();

// Aggressive: split the top two axes at once (4 children per node).
static void BM_RefineCoeff4way( benchmark::State& state )
{
    const int threads = static_cast< int >( state.range( 0 ) );
    std::size_t leaves = 0;
    for ( auto _ : state )
    {
        auto tree =
            tax::ads::refine< 6 >( Verner89{}, tax::ads::CoefficientMatchCriterion{ 1e-6, 6 },
                                   rhs(), icBox(), center(), 0.0, kTfinal, cfg(), threads,
                                   /*split_dirs=*/2 );
        benchmark::DoNotOptimize( tree );
        leaves = tree.done().size();
    }
    state.counters["leaves"] = static_cast< double >( leaves );
}
BENCHMARK( BM_RefineCoeff4way )->DenseRange( 1, 4 )->Unit( benchmark::kMillisecond )->UseRealTime();

static void BM_RefineVolume( benchmark::State& state )
{
    const int threads = static_cast< int >( state.range( 0 ) );
    std::size_t leaves = 0;
    for ( auto _ : state )
    {
        auto tree = tax::ads::refine< 6 >( Verner89{},
                                           tax::ads::VolumeRatioCriterion{ 1e-6, 6, { 1, 3 }, 8 },
                                           rhs(), icBox(), center(), 0.0, kTfinal, cfg(), threads );
        benchmark::DoNotOptimize( tree );
        leaves = tree.done().size();
    }
    state.counters["leaves"] = static_cast< double >( leaves );
}
BENCHMARK( BM_RefineVolume )->DenseRange( 1, 4 )->Unit( benchmark::kMillisecond )->UseRealTime();

// ---- 3) classic ADS vs polynomial order (single thread) --------------------

template < int P >
static void BM_OrderTruncation( benchmark::State& state )
{
    std::size_t leaves = 0;
    for ( auto _ : state )
    {
        auto tree = tax::ads::propagate< P >( Verner89{}, tax::ads::TruncationCriterion{ 1e-4, 8 },
                                              rhs(), icBox(), center(), 0.0, kTfinal, cfg(), 1 );
        benchmark::DoNotOptimize( tree );
        leaves = tree.done().size();
    }
    state.counters["leaves"] = static_cast< double >( leaves );
}
BENCHMARK_TEMPLATE( BM_OrderTruncation, 2 )->Unit( benchmark::kMillisecond );
BENCHMARK_TEMPLATE( BM_OrderTruncation, 4 )->Unit( benchmark::kMillisecond );
BENCHMARK_TEMPLATE( BM_OrderTruncation, 6 )->Unit( benchmark::kMillisecond );

template < int P >
static void BM_OrderNli( benchmark::State& state )
{
    std::size_t leaves = 0;
    for ( auto _ : state )
    {
        auto tree = tax::ads::propagate< P >( Verner89{}, tax::ads::NliCriterion{ 0.1, 8 }, rhs(),
                                              icBox(), center(), 0.0, kTfinal, cfg(), 1 );
        benchmark::DoNotOptimize( tree );
        leaves = tree.done().size();
    }
    state.counters["leaves"] = static_cast< double >( leaves );
}
BENCHMARK_TEMPLATE( BM_OrderNli, 2 )->Unit( benchmark::kMillisecond );
BENCHMARK_TEMPLATE( BM_OrderNli, 4 )->Unit( benchmark::kMillisecond );
BENCHMARK_TEMPLATE( BM_OrderNli, 6 )->Unit( benchmark::kMillisecond );

BENCHMARK_MAIN();
