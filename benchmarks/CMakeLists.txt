include(FetchContent)
find_package(benchmark 1.9 QUIET)
if(NOT benchmark_FOUND AND NOT TARGET benchmark::benchmark)
    FetchContent_Declare(benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.4)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
endif()
function(tax_add_bench name)
    cmake_parse_arguments(B "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${B_SOURCES})
    target_link_libraries(${name} PRIVATE tax-flow benchmark::benchmark)
endfunction()

tax_add_bench(bench_ode_cr3bp SOURCES bench_ode_cr3bp.cpp)
# bench_ode_cr3bp.cpp includes "../tests/ode/problems/cr3bp_problem.hpp"
target_include_directories(bench_ode_cr3bp PRIVATE ${CMAKE_SOURCE_DIR})

tax_add_bench(bench_ads_refine SOURCES bench_ads_refine.cpp)
