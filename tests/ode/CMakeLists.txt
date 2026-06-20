# tests/ode/CMakeLists.txt
#
# ODE integrator unit tests. Each test executable links the tax
# INTERFACE target and gtest_main. Registered via tax_add_test()
# from the parent tests/CMakeLists.txt.
#
# Layout mirrors include/tax/ode/:
#   ./                  — foundation (config, concepts, solution, controllers)
#                         plus cross-cutting features (FixedStep, VectorOps,
#                         DA-vector RK state, RK helpers).
#   integrator/         — driver-level tests (basic, dense, static-vs-dynamic).
#   events/             — event-system tests (zero-crossing, every-step).
#   steppers/           — one file per stepper (Taylor + 5 RK families).
#   problems/           — end-to-end correctness on real ODEs
#                         (planar Kepler, CR3BP propagation + events).

# ---- Foundation + cross-cutting -----------------------------------------
tax_add_test(test_ode_config         SOURCES test_config.cpp)
tax_add_test(test_ode_concepts       SOURCES test_concepts.cpp)
tax_add_test(test_ode_solution       SOURCES test_solution.cpp)
tax_add_test(test_ode_controllers    SOURCES test_controllers.cpp)
tax_add_test(test_ode_fixed_step     SOURCES test_fixed_step.cpp)
tax_add_test(test_ode_vector_ops     SOURCES test_vector_ops.cpp)
tax_add_test(test_ode_rk_da_state    SOURCES test_rk_da_state.cpp)
tax_add_test(test_ode_named_state    SOURCES test_named_state.cpp)
tax_add_test(test_ode_rk_helpers     SOURCES test_rk_helpers.cpp)

# ---- Integrator driver --------------------------------------------------
tax_add_test(test_ode_integrator_basic      SOURCES integrator/test_basic.cpp)
tax_add_test(test_ode_integrator_static     SOURCES integrator/test_static.cpp)
tax_add_test(test_ode_integrator_save_steps SOURCES integrator/test_save_steps.cpp)

# ---- Events -------------------------------------------------------------
tax_add_test(test_ode_events_zero_crossing SOURCES events/test_zero_crossing.cpp)
tax_add_test(test_ode_events_every_step    SOURCES events/test_every_step.cpp)
tax_add_test(test_ode_step_evaluator       SOURCES events/test_step_evaluator.cpp)
tax_add_test(test_ode_events_grid          SOURCES events/test_grid_event.cpp)

# ---- Steppers -----------------------------------------------------------
tax_add_test(test_ode_taylor_stepper      SOURCES steppers/test_taylor.cpp)
tax_add_test(test_ode_verner78_stepper    SOURCES steppers/test_verner78.cpp)
tax_add_test(test_ode_verner89_stepper    SOURCES steppers/test_verner89.cpp)
tax_add_test(test_ode_fehlberg78_stepper  SOURCES steppers/test_fehlberg78.cpp)
tax_add_test(test_ode_feagin12_stepper    SOURCES steppers/test_feagin12.cpp)
tax_add_test(test_ode_feagin14_stepper    SOURCES steppers/test_feagin14.cpp)

# ---- End-to-end problems ------------------------------------------------
tax_add_test(test_ode_two_body_kepler     SOURCES problems/test_two_body_kepler.cpp)
tax_add_test(test_ode_cr3bp_propagation   SOURCES problems/test_cr3bp_propagation.cpp)
tax_add_test(test_ode_cr3bp_events        SOURCES problems/test_cr3bp_events.cpp)
