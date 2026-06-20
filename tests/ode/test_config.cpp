// tests/ode/testConfig.cpp
//
// Validates IntegratorConfig defaults and constructor behaviour.

#include <gtest/gtest.h>

#include <tax/ode.hpp>

TEST( OdeConfig, DefaultsAreSane )
{
    tax::ode::IntegratorConfig< double > cfg;
    EXPECT_GT( cfg.abstol, 0.0 );
    EXPECT_GT( cfg.reltol, 0.0 );
    EXPECT_EQ( cfg.initial_step, 0.0 );  // 0 ⇒ stepper picks
    EXPECT_EQ( cfg.min_step, 0.0 );      // 0 ⇒ ~eps × span
    EXPECT_EQ( cfg.max_step, 0.0 );      // 0 ⇒ tmax - t0
    EXPECT_GT( cfg.max_steps, 0 );
    EXPECT_GT( cfg.max_rejects_per_step, 0 );
}

TEST( OdeConfig, AcceptsCustomValues )
{
    tax::ode::IntegratorConfig< double > cfg{
        .abstol = 1e-10,
        .reltol = 1e-8,
        .initial_step = 1e-3,
        .min_step = 1e-12,
        .max_step = 0.1,
        .max_steps = 1000,
        .max_rejects_per_step = 8,
    };
    EXPECT_DOUBLE_EQ( cfg.abstol, 1e-10 );
    EXPECT_DOUBLE_EQ( cfg.reltol, 1e-8 );
    EXPECT_EQ( cfg.max_steps, 1000 );
}
