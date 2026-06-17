// include/tax/ode/solution.hpp
//
// Solution<Stepper, State, Dense> — partial-specialised on Dense.
// Dense=false:  step-boundary states + events only.
// Dense=true :  adds per-step continuous-extension payload and an
//               operator()(t_query) that interpolates via the
//               stepper's static eval_dense().

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace tax::ode
{

template < class State, class T >
struct EventRecord
{
    std::string label;          // "" if anonymous
    T           t;
    State       x;
};

template < class Stepper, class State, bool Dense >
class Solution;

// Discrete specialisation: step boundaries + events only.
template < class Stepper, class State >
class Solution< Stepper, State, /*Dense=*/false >
{
public:
    using T = typename Stepper::T;
    std::vector< T >                       t;
    std::vector< State >                   x;
    std::vector< EventRecord< State, T > > events;

    [[nodiscard]] std::size_t size() const noexcept { return t.size(); }
};

// Dense specialisation: adds per-step continuous-extension data and sol(t).
template < class Stepper, class State >
class Solution< Stepper, State, /*Dense=*/true >
{
public:
    using T         = typename Stepper::T;
    using DenseData = typename Stepper::DenseData;

    std::vector< T >                       t;        // size = nsteps + 1
    std::vector< State >                   x;        // size = nsteps + 1
    std::vector< DenseData >               dense;    // size = nsteps
    std::vector< EventRecord< State, T > > events;

    [[nodiscard]] std::size_t size() const noexcept { return t.size(); }

    [[nodiscard]] State operator()( const T& t_query ) const
    {
        if ( t.empty() )
            throw std::runtime_error( "Solution::operator(): empty solution" );
        if ( dense.empty() )
            throw std::runtime_error( "Solution::operator(): no dense data" );
        if ( t_query < t.front() || t_query > t.back() )
            throw std::out_of_range( "Solution::operator(): t_query out of [t0, tf]" );

        // Invariant: dense.size() == t.size() - 1 (one continuous extension
        // per step). Maintained by the Integrator; callers constructing a
        // Solution manually must preserve it.
        assert( dense.size() + 1 == t.size() );

        // Binary search: find i with t[i] <= t_query <= t[i+1].
        auto       it = std::upper_bound( t.begin(), t.end(), t_query );
        // upper_bound never returns begin() here because t_query >= t.front()
        // is already enforced; the clamp guards against any future relaxation.
        const auto i  = static_cast< std::size_t >( std::max< std::ptrdiff_t >(
            0, std::distance( t.begin(), it ) - 1 ) );
        const std::size_t idx = std::min( i, dense.size() - 1 );
        return Stepper::eval_dense( dense[idx], t[idx], t_query );
    }
};

}  // namespace tax::ode
