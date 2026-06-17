// include/tax/ode/io.hpp
//
// Snapshot helpers for tax::ode solutions. Opt-in: not pulled in by the
// umbrella <tax/ode.hpp>; users include this only when they want CSV
// output.
//
//   linspace(t0, t1, n)      — n evenly-spaced times in [t0, t1].
//   writeCsv(sol, ts, path)  — write a Dense=true Solution's snapshots
//                              at the given times as CSV
//                              (columns: t, x0, x1, …, x{D-1}).

#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <span>
#include <type_traits>
#include <vector>

namespace tax::ode
{

[[nodiscard]] inline std::vector< double > linspace( double t0, double t1, int n_points )
{
    std::vector< double > out;
    if ( n_points <= 0 ) return out;
    out.reserve( static_cast< std::size_t >( n_points ) );
    if ( n_points == 1 )
    {
        out.push_back( t0 );
        return out;
    }
    const double step = ( t1 - t0 ) / ( n_points - 1 );
    for ( int i = 0; i < n_points; ++i ) out.push_back( t0 + i * step );
    return out;
}

template < class Solution >
void writeCsv( const Solution& sol, std::span< const double > times,
               const std::filesystem::path& path )
{
    std::ofstream out( path );
    using State          = std::decay_t< decltype( sol.x.front() ) >;
    const Eigen::Index D = sol.x.front().size();
    out << 't';
    for ( Eigen::Index j = 0; j < D; ++j ) out << ",x" << j;
    out << '\n';
    for ( double t : times )
    {
        State x = sol( t );
        out << t;
        for ( Eigen::Index j = 0; j < x.size(); ++j ) out << ',' << x( j );
        out << '\n';
    }
}

}  // namespace tax::ode
