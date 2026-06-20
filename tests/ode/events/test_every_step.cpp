#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <tax/la/types.hpp>
#include <tax/ode.hpp>
#include <vector>

using tax::ode::IntegratorConfig;
using tax::ode::StepEvent;
using tax::ode::TaylorStepper;

TEST( OdeEventsEveryStep, FiresOncePerStep )
{
    constexpr int N = 12;
    using State = tax::la::VecNT< 1, double >;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const auto f = []( const auto& x, const auto& ) { return x; };

    tax::ode::Taylor< N, State, tax::ode::controllers::JorbaZou< double >, decltype( f ) > integ{
        f, cfg };
    integ.addStepEvent( "step" );

    State x0;
    x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    // One record per accepted step == sol.size() - 1.
    EXPECT_EQ( sol.events.size(), sol.size() - 1 );
    EXPECT_GE( sol.events.size(), 1u );
    for ( const auto& e : sol.events ) EXPECT_EQ( e.label, "step" );
}

namespace
{
// A user-defined event: terminate once the step boundary passes t = 0.3.
template < class State, class T >
class StopAfter final : public tax::ode::BaseEvent< StopAfter< State, T >, State, T >
{
   public:
    using Action = typename tax::ode::Event< State, T >::Action;
    explicit StopAfter( T t_stop ) : t_stop_( t_stop ) {}
    [[nodiscard]] std::string name() const override { return "stop_after"; }
    Action onStep( tax::ode::Recorder< State, T >& ) override
    {
        const T t_new = this->eval_->tOld() + this->eval_->hUsed();
        return t_new > t_stop_ ? Action::Terminate : Action::Continue;
    }

   private:
    T t_stop_;
};
}  // namespace

TEST( OdeEventsEveryStep, UserEventCanTerminate )
{
    constexpr int N = 12;
    using State = tax::la::VecNT< 1, double >;

    IntegratorConfig< double > cfg;
    cfg.abstol = cfg.reltol = 1e-12;

    const auto f = []( const auto& x, const auto& ) { return x; };

    tax::ode::Taylor< N, State, tax::ode::controllers::JorbaZou< double >, decltype( f ) > integ{
        f, cfg };
    integ.addEvent( std::make_shared< StopAfter< State, double > >( 0.3 ) );

    State x0;
    x0( 0 ) = 1.0;
    auto sol = integ.integrate( x0, 0.0, 1.0 );

    EXPECT_LT( sol.t.back(), 1.0 );
    EXPECT_GE( sol.t.back(), 0.3 );
}
