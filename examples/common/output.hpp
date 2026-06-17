// =============================================================================
// examples/common/output.hpp
//
// Shared I/O scaffolding for the example programs, so each example's
// main() contains only the problem setup and the tax calls.
//
//   * Stopwatch              — wall-clock timer (milliseconds)
//   * adsThreads()           — worker count (TAX_ADS_THREADS or hardware)
//   * unitSquareBoundary(n)  — closed loop tracing the boundary of [-1,1]^2
//   * evalPolygon(...)       — image of that loop under a DA flow map
//   * boxPolygon(...)        — the loop mapped onto the IC box itself (t = 0)
//   * sampleOrbit(...)       — dense solution sampled on a uniform grid
//   * writeRunJson(...)      — the one JSON schema shared by all examples
//   * printBanner(...)       — tidy per-example terminal summary
//
// JSON schema written by writeRunJson:
//
//   {
//     "method":  "<taylor|ads|loads>",
//     "params":  { "<key>": <raw json>, ... },
//     "timing":  { "elapsed_ms": <num> },
//     "reference_orbit": { "t": [...], "x0": [...], ..., "x<D-1>": [...] },
//     "snapshots": [
//       { "t": <num>, "leaves": [ { "id": <int>, "depth": <int>,
//                                   "x": [...], "y": [...] }, ... ] },
//       ...
//     ]
//   }
//
// The plotting scripts in examples/plot/ consume exactly this schema.
// =============================================================================

#pragma once

#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace example
{

// ---- Timing -----------------------------------------------------------------
class Stopwatch
{
   public:
    Stopwatch() : start_( std::chrono::steady_clock::now() ) {}
    [[nodiscard]] double ms() const
    {
        return std::chrono::duration< double, std::milli >( std::chrono::steady_clock::now()
                                                            - start_ )
            .count();
    }

   private:
    std::chrono::steady_clock::time_point start_;
};

// ---- Parallel ADS worker count ----------------------------------------------
inline int adsThreads()
{
    if ( const char* e = std::getenv( "TAX_ADS_THREADS" ) )
    {
        const int n = std::atoi( e );
        if ( n > 0 ) return n;
    }
    const unsigned hc = std::thread::hardware_concurrency();
    return hc > 0 ? static_cast< int >( hc ) : 1;
}

// ---- Boundary of [-1, 1]^2 ----------------------------------------------------
//
// 4 * n_per_edge + 1 samples tracing the perimeter counter-clockwise; the
// first vertex is repeated at the end so the polygon closes.
inline std::vector< std::array< double, 2 > > unitSquareBoundary( int n_per_edge )
{
    std::vector< std::array< double, 2 > > pts;
    pts.reserve( static_cast< std::size_t >( 4 * n_per_edge + 1 ) );
    for ( int edge = 0; edge < 4; ++edge )
    {
        for ( int i = 0; i < n_per_edge; ++i )
        {
            const double s = static_cast< double >( i ) / static_cast< double >( n_per_edge );
            double a = 0.0, b = 0.0;
            switch ( edge )
            {
                case 0: a = -1.0 + 2.0 * s; b = +1.0;            break;
                case 1: a = +1.0;           b = +1.0 - 2.0 * s;  break;
                case 2: a = +1.0 - 2.0 * s; b = -1.0;            break;
                case 3: a = -1.0;           b = -1.0 + 2.0 * s;  break;
            }
            pts.push_back( { a, b } );
        }
    }
    pts.push_back( pts.front() );
    return pts;
}

// ---- Polygons -----------------------------------------------------------------
struct Polygon
{
    int id = 0;
    int depth = 0;
    std::vector< double > x;
    std::vector< double > y;
};

struct Snapshot
{
    double t = 0.0;
    std::vector< Polygon > leaves;
};

/**
 * Image of the IC-box boundary under a DA flow map: evaluates the (x, y)
 * components of `state` (an Eigen vector of TaylorExpansions) at each
 * boundary sample. `toBox(a, b)` maps the two active boundary coordinates
 * to the M-dimensional normalised displacement (problem-specific).
 */
template < class StateVec, class ToBox >
Polygon evalPolygon( const StateVec& state,
                     const std::vector< std::array< double, 2 > >& boundary, ToBox&& toBox,
                     int id = 0, int depth = 0 )
{
    Polygon p;
    p.id = id;
    p.depth = depth;
    p.x.reserve( boundary.size() );
    p.y.reserve( boundary.size() );
    for ( const auto& ab : boundary )
    {
        const auto d = toBox( ab[0], ab[1] );
        p.x.push_back( state( 0 ).eval( d ) );
        p.y.push_back( state( 1 ).eval( d ) );
    }
    return p;
}

/**
 * The boundary loop mapped onto the IC box itself — the exact t = 0
 * "polygon" (the box's (x, y) face), bypassing any flow map.
 */
template < class BoxT, class ToBox >
Polygon boxPolygon( const BoxT& box, const std::vector< std::array< double, 2 > >& boundary,
                    ToBox&& toBox )
{
    Polygon p;
    p.x.reserve( boundary.size() );
    p.y.reserve( boundary.size() );
    for ( const auto& ab : boundary )
    {
        const auto d = toBox( ab[0], ab[1] );
        p.x.push_back( box.center( 0 ) + box.halfWidth( 0 ) * d[0] );
        p.y.push_back( box.center( 1 ) + box.halfWidth( 1 ) * d[1] );
    }
    return p;
}

// ---- Reference orbit sampling -------------------------------------------------
struct OrbitSamples
{
    std::vector< double > t;
    std::vector< std::vector< double > > cols;  // cols[j][i] = x_j(t_i)
};

/// Sample a Dense solution on `times` into per-component columns.
template < class Sol >
OrbitSamples sampleOrbit( const Sol& sol, const std::vector< double >& times, int dim )
{
    OrbitSamples s;
    s.t = times;
    s.cols.assign( static_cast< std::size_t >( dim ),
                   std::vector< double >( times.size() ) );
    for ( std::size_t i = 0; i < times.size(); ++i )
    {
        const auto x = sol( times[i] );
        for ( int j = 0; j < dim; ++j ) s.cols[std::size_t( j )][i] = x( j );
    }
    return s;
}

// ---- JSON helpers ---------------------------------------------------------------
template < class Range >
inline void writeJsonArray( std::ostream& out, const Range& v )
{
    out << '[';
    bool first = true;
    for ( auto x : v )
    {
        if ( !first ) out << ", ";
        out << x;
        first = false;
    }
    out << ']';
}

/// Render a numeric range as inline JSON text (for `params` values).
template < class Range >
inline std::string jsonArray( const Range& v )
{
    std::ostringstream os;
    os << std::setprecision( 14 );
    writeJsonArray( os, v );
    return os.str();
}

/// Render a number as JSON text (for `params` values).
inline std::string jsonNumber( double v )
{
    std::ostringstream os;
    os << std::setprecision( 14 ) << v;
    return os.str();
}

/// Ordered key -> raw-JSON-value pairs placed under "params".
using JsonParams = std::vector< std::pair< std::string, std::string > >;

/// Write a complete example run in the shared schema (see header comment).
inline void writeRunJson( const std::string& path, std::string_view method,
                          const JsonParams& params, const OrbitSamples& reference,
                          const std::vector< Snapshot >& snapshots, double elapsed_ms )
{
    std::ofstream out( path );
    out << std::setprecision( 14 );

    out << "{\n  \"method\": \"" << method << "\",\n";

    out << "  \"params\": {\n";
    for ( std::size_t i = 0; i < params.size(); ++i )
        out << "    \"" << params[i].first << "\": " << params[i].second
            << ( i + 1 < params.size() ? "," : "" ) << "\n";
    out << "  },\n";

    out << "  \"timing\": { \"elapsed_ms\": " << elapsed_ms << " },\n";

    out << "  \"reference_orbit\": {\n    \"t\": ";
    writeJsonArray( out, reference.t );
    for ( std::size_t j = 0; j < reference.cols.size(); ++j )
    {
        out << ",\n    \"x" << j << "\": ";
        writeJsonArray( out, reference.cols[j] );
    }
    out << "\n  },\n";

    out << "  \"snapshots\": [\n";
    for ( std::size_t s = 0; s < snapshots.size(); ++s )
    {
        out << "    { \"t\": " << snapshots[s].t << ", \"leaves\": [";
        const auto& leaves = snapshots[s].leaves;
        for ( std::size_t l = 0; l < leaves.size(); ++l )
        {
            out << "\n      { \"id\": " << leaves[l].id << ", \"depth\": " << leaves[l].depth
                << ", \"x\": ";
            writeJsonArray( out, leaves[l].x );
            out << ", \"y\": ";
            writeJsonArray( out, leaves[l].y );
            out << " }" << ( l + 1 < leaves.size() ? "," : "" );
        }
        out << "\n    ] }" << ( s + 1 < snapshots.size() ? "," : "" ) << "\n";
    }
    out << "  ]\n}\n";
}

// ---- Terminal banner --------------------------------------------------------------
using BannerRows = std::vector< std::pair< std::string, std::string > >;

inline void printBanner( std::string_view title, const BannerRows& rows )
{
    constexpr std::size_t label_w = 18;
    std::cout << "\n=== " << title << " ===\n";
    for ( const auto& [ label, value ] : rows )
    {
        const std::size_t pad = label.size() < label_w ? label_w - label.size() : 0;
        std::cout << "  " << std::string( pad, ' ' ) << label << " : " << value << '\n';
    }
    std::cout << '\n';
}

}  // namespace example
