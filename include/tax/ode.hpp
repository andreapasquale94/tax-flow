// include/tax/ode.hpp
//
// Opt-in ODE integration umbrella — include only this header to access
// the full ODE surface (config + concepts + step result + solution + …).

#pragma once

#include <tax/ode/aliases.hpp>
#include <tax/ode/concepts.hpp>
#include <tax/ode/config.hpp>
#include <tax/ode/controllers.hpp>
#include <tax/ode/event.hpp>
#include <tax/ode/events/grid_event.hpp>
#include <tax/ode/events/root_finding_event.hpp>
#include <tax/ode/events/step_event.hpp>
#include <tax/ode/integrator.hpp>
#include <tax/ode/named.hpp>
#include <tax/ode/propagate.hpp>
#include <tax/ode/solution.hpp>
#include <tax/ode/step_evaluator.hpp>
#include <tax/ode/step_result.hpp>
#include <tax/ode/steppers/dormand_prince45.hpp>
#include <tax/ode/steppers/feagin12.hpp>
#include <tax/ode/steppers/feagin14.hpp>
#include <tax/ode/steppers/fehlberg78.hpp>
#include <tax/ode/steppers/taylor.hpp>
#include <tax/ode/steppers/verner67.hpp>
#include <tax/ode/steppers/verner78.hpp>
#include <tax/ode/steppers/verner89.hpp>

// Validated Taylor-model integration (methods::Picard) — only when the tax
// core ships the tax::model module (tax PR "feat(model)").
#if __has_include( <tax/model.hpp>)
#include <tax/ode/steppers/taylor_model.hpp>
#endif
