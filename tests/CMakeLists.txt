include(FetchContent)
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# Each test executable links the tax-flow INTERFACE target (which pulls tax,
# Eigen3 and Threads transitively) and gtest_main.
function(tax_add_test name)
    cmake_parse_arguments(T "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${T_SOURCES})
    target_link_libraries(${name} PRIVATE tax-flow GTest::gtest GTest::gtest_main)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

# Domain module
tax_add_test(test_domain_box SOURCES domain/test_box.cpp)
tax_add_test(test_domain_zonotope SOURCES domain/test_zonotope.cpp)
tax_add_test(test_domain_reorient SOURCES domain/test_reorient.cpp)
tax_add_test(test_domain_polynomial_zonotope SOURCES domain/test_polynomial_zonotope.cpp)
tax_add_test(test_domain_enclosure SOURCES domain/test_enclosure.cpp)
tax_add_test(test_domain_localize_evaluate SOURCES domain/test_localize_evaluate.cpp)
tax_add_test(test_domain_soundness SOURCES domain/test_soundness.cpp)

# ADS
tax_add_test(test_ads_leaf_tree SOURCES ads/test_leaf_tree.cpp)
tax_add_test(test_ads_nonlinearity_index SOURCES ads/test_nonlinearity_index.cpp)
tax_add_test(test_ads_criteria SOURCES ads/test_criteria.cpp)
tax_add_test(test_ads_split_event SOURCES ads/test_split_event.cpp)
tax_add_test(test_ads_da_state SOURCES ads/test_da_state.cpp)
tax_add_test(test_ads_driver SOURCES ads/test_driver.cpp)
tax_add_test(test_ads_merge SOURCES ads/test_merge.cpp)
tax_add_test(test_ads_parallel SOURCES ads/test_parallel.cpp)
tax_add_test(test_ads_refine SOURCES ads/test_refine.cpp)
tax_add_test(test_ads_solution SOURCES ads/test_ads_solution.cpp)
tax_add_test(test_ads_snapshots SOURCES ads/test_ads_snapshots.cpp)

# ODE integrator
add_subdirectory(ode)
