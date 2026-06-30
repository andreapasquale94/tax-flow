#tests / sgp4 / CMakeLists.txt
#
#SGP4 module unit tests.Each executable links the tax - flow INTERFACE target
#and gtest_main.Registered via tax_add_test() from tests / CMakeLists.txt.
#
#test_sgp4_tle      — TLE parsing( elements, units, epoch->Julian date )
#test_sgp4_vallado  — double propagator vs Vallado's tcppver.out reference
#test_sgp4_taylor   — Taylor compatibility( value identity + Jacobian vs FD )

tax_add_test( test_sgp4_tle SOURCES test_sgp4_tle.cpp )
    tax_add_test( test_sgp4_vallado SOURCES test_sgp4_vallado.cpp )
        tax_add_test( test_sgp4_taylor SOURCES test_sgp4_taylor.cpp )
