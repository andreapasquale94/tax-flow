# examples/CMakeLists.txt
#
# Each example is a single source file linking against the header-only
# tax-flow library. Run via `./two_body_taylor` etc. — they write CSV
# output into the current working directory.

function(tax_add_example name)
    cmake_parse_arguments(E "" "" "SOURCES" ${ARGN})
    add_executable(${name} ${E_SOURCES})
    target_link_libraries(${name} PRIVATE tax-flow)
endfunction()

tax_add_example(two_body_taylor     SOURCES two_body/taylor.cpp)
tax_add_example(two_body_ads        SOURCES two_body/ads.cpp)
tax_add_example(two_body_loads      SOURCES two_body/loads.cpp)
tax_add_example(two_body_refine     SOURCES two_body/refine.cpp)
tax_add_example(two_body_validation SOURCES two_body/validation.cpp)
tax_add_example(three_body_taylor SOURCES three_body/taylor.cpp)
tax_add_example(three_body_ads    SOURCES three_body/ads.cpp)
tax_add_example(three_body_loads  SOURCES three_body/loads.cpp)
tax_add_example(wsb_search        SOURCES wsb/wsb_search.cpp)
tax_add_example(wsb_taylor        SOURCES wsb/taylor.cpp)
tax_add_example(wsb_ads           SOURCES wsb/ads.cpp)
tax_add_example(reachability      SOURCES reachability/reachability.cpp)
tax_add_example(missed_thrust     SOURCES missed_thrust/missed_thrust.cpp)
