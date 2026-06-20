cmake_minimum_required(VERSION 3.28)
project(tax-flow VERSION 0.1.0 LANGUAGES CXX)

option(TAXFLOW_BUILD_UNITTESTS "Build and enable unit tests"               ON)
option(TAXFLOW_BUILD_EXAMPLES  "Build example programs (under examples/)"  OFF)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wno-psabi)
endif()

# ---------------------------------------------------------------
# tax dependency
# ---------------------------------------------------------------
# Prefer an installed tax (find_package); otherwise fall back to the sibling
# source tree so a fresh checkout of both folders builds with no install step.
set(TAX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../tax"
    CACHE PATH "Path to tax source tree (used when tax is not installed)")
find_package(tax CONFIG QUIET)
if(tax_FOUND)
    message(STATUS "tax-flow: using installed tax ${tax_VERSION}")
else()
    if(NOT EXISTS "${TAX_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "tax was not found via find_package and no source tree exists at "
            "TAX_SOURCE_DIR='${TAX_SOURCE_DIR}'.\n"
            "Install tax, or point -DTAX_SOURCE_DIR=<path-to-tax> at its checkout.")
    endif()
    message(STATUS "tax-flow: building tax from source at ${TAX_SOURCE_DIR}")
    # Don't drag tax's own tests/benchmarks/examples into a tax-flow build.
    set(TAX_BUILD_UNITTESTS   OFF CACHE BOOL "" FORCE)
    set(TAX_BUILD_REGRESSIONS OFF CACHE BOOL "" FORCE)
    set(TAX_BUILD_BENCHMARK   OFF CACHE BOOL "" FORCE)
    set(TAX_BUILD_EXAMPLES    OFF CACHE BOOL "" FORCE)
    add_subdirectory("${TAX_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/_tax")
endif()

# ---------------------------------------------------------------
# tax-flow interface library
# ---------------------------------------------------------------
add_library(tax-flow INTERFACE)
add_library(tax::flow ALIAS tax-flow)
set_target_properties(tax-flow PROPERTIES EXPORT_NAME flow)
target_compile_features(tax-flow INTERFACE cxx_std_23)
target_include_directories(tax-flow INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
# tax::tax carries Eigen3 and Threads as transitive INTERFACE dependencies.
target_link_libraries(tax-flow INTERFACE tax::tax)

if(TAXFLOW_BUILD_UNITTESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if(TAXFLOW_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ---------------------------------------------------------------
# install / export
# ---------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(TAXFLOW_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/tax-flow")
install(TARGETS tax-flow EXPORT taxFlowTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT taxFlowTargets FILE taxFlowTargets.cmake NAMESPACE tax::
        DESTINATION ${TAXFLOW_INSTALL_CMAKEDIR})
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tax-flowConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/tax-flowConfig.cmake"
    INSTALL_DESTINATION ${TAXFLOW_INSTALL_CMAKEDIR})
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/tax-flowConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/tax-flowConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/tax-flowConfigVersion.cmake"
    DESTINATION ${TAXFLOW_INSTALL_CMAKEDIR})
