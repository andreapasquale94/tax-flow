// include/tax/ode.hpp
//
// Opt-in ODE integration umbrella — include only this header to access
// the full ODE surface (config + concepts + step result + solution + …).

#pragma once

#include <tax/ode/config.hpp>
#include <tax/ode/step_result.hpp>
#include <tax/ode/concepts.hpp>
#include <tax/ode/solution.hpp>
#include <tax/ode/controllers.hpp>
#include <tax/ode/steppers/taylor.hpp>
#include <tax/ode/steppers/verner78.hpp>
#include <tax/ode/steppers/verner89.hpp>
#include <tax/ode/steppers/fehlberg78.hpp>
#include <tax/ode/steppers/feagin12.hpp>
#include <tax/ode/steppers/feagin14.hpp>
#include <tax/ode/event.hpp>
#include <tax/ode/actions.hpp>
#include <tax/ode/triggers.hpp>
#include <tax/ode/integrator.hpp>
#include <tax/ode/aliases.hpp>
#include <tax/ode/propagate.hpp>
#include <tax/ode/named.hpp>
