#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <tclap/CmdLine.h>

#include "GpuUtility.h"
#include "Utility.h"
#include "MatrixMultiply.h"

// From: https://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c/
bool is_power_of_two( size_t x )
{
    return ((x != 0) && ((x & (~x + 1)) == x));
}

int main( int argc, char ** argv )
{
    try
    {
        // Parse command line.
        TCLAP::CmdLine cmd( "EE382V Algorithms Final - Matrix Multiply" );

        TCLAP::ValueArg<double> lower( "l",
            "lower",
            "Lower bound on the random number generator.",
            false,
            -50.0,
            "lower bound" );

        TCLAP::ValueArg<double> upper( "u",
            "upper",
            "Upper bound on the random number generator.",
            false,
            50.0,
            "upper bound" );

        TCLAP::ValueArg<size_t> precision( "p",
            "precision",
            "Number of digits to display for each matrix entry.",
            false,
            6,
            "number digits" );

        TCLAP::ValueArg<size_t> seed( "s",
            "seed",
            "Seed to use for random number generation.",
            false,
            1,
            "RNG seed" );

        TCLAP::SwitchArg integer_entries( "i",
            "int",
            "Use integer entries instead of real entries.",
            false );

        TCLAP::ValueArg<size_t> cutoff( "c",
            "cutoff",
            "Power of 2 cutoff value used by Strassen multiplication.",
            false,
            2,
            "cutoff" );

        TCLAP::ValueArg<size_t> trials( "t",
            "trials",
            "Number of trials to run.",
            false,
            3,
            "number trials" );

        TCLAP::ValueArg<std::string> algorithm_arg( "a",
            "algorithm",
            "Matrix multiply algorithm to use.",
            false,
            "serial_dp",
            "algorithm" );

        TCLAP::UnlabeledValueArg<size_t> matrix_size( "matrix_size",
            "Specifies the size of the N-by-N matrix to compute. Must be power of 2.",
            true, // required
            2, // default value
            "matrix size" );

        cmd.add( upper );
        cmd.add( trials );
        cmd.add( seed );
        cmd.add( precision );
        cmd.add( lower );
        cmd.add( integer_entries );
        cmd.add( cutoff );
        cmd.add( algorithm_arg );
        cmd.add( matrix_size );
        cmd.parse( argc, argv );

        // Extract command line values.
        const size_t N = matrix_size.getValue();
        const size_t strassen_cutoff = cutoff.getValue();
        const double lower_bound = lower.getValue();
        const double upper_bound = upper.getValue();
        const size_t number_trials = trials.getValue();
        const std::string algorithm( algorithm_arg.getValue() );
        const std::vector<std::string> valid_algorithms
        {
            "serial_dp",
            "serial_sp",
            "serial_op",
            "strass_serial",
            "gpu_dp",
            "gpu_dp_shared"
        };

        // Make sure size and cutoff values are powers of two.
        if( !(is_power_of_two( N ) && is_power_of_two( strassen_cutoff )) )
            throw std::invalid_argument( "Size and cutoff (if given) must be powers of 2." );

        // Make sure algorithm is valid.
        if( std::none_of( valid_algorithms.begin(), valid_algorithms.end(),
            [&]( const std::string & s ) { return s == algorithm; } ) )
            throw std::invalid_argument( "Unrecognized algorithm given." );

        auto trial_durations = std::make_unique<double[]>( number_trials );

        for( size_t trial = 0; trial < number_trials; ++trial )
        {
            // Allocate matrices.
            auto A = make_matrix_gpu( N );
            auto B = make_matrix_gpu( N );
            auto C = make_matrix_gpu( N );

            // Initialize matrices.
            init_matrix( C.get(), N, zero_gen );

            std::default_random_engine random_engine;
            if( seed.isSet() )
            {
                random_engine.seed( seed.getValue() );
            }
            else
            {
                std::random_device rd;
                random_engine.seed( rd() );
            }

            if( integer_entries.getValue() )
            {
                std::uniform_int_distribution<int> random_distribution( static_cast<int>(lower_bound),
                    static_cast<int>(upper_bound) );
                auto next_random = std::bind( random_distribution, random_engine );
                init_matrix( A.get(), N, next_random );
                init_matrix( B.get(), N, next_random );
            }
            else
            {
                std::uniform_real_distribution<double> random_distribution( lower_bound, upper_bound );
                auto next_random = std::bind( random_distribution, random_engine );
                init_matrix( A.get(), N, next_random );
                init_matrix( B.get(), N, next_random );
            }

            // Run matrix multiply.
            if( algorithm == "serial_dp" )
                multiply_dp( A.get(), B.get(), C.get(), N, trial_durations[trial] );
            else if( algorithm == "serial_sp" )
                multiply_sp( A.get(), B.get(), C.get(), N, trial_durations[trial] );
            else if( algorithm == "serial_op" )
                multiply_op( A.get(), B.get(), C.get(), N, trial_durations[trial] );
            else if( algorithm == "strass_serial" )
                strass_serial( A.get(), B.get(), C.get(), N, strassen_cutoff, trial_durations[trial] );
            else if( algorithm == "gpu_dp" )
                multiply_dp_gpu( A.get(), B.get(), C.get(), N, trial_durations[trial] );
            else if( algorithm == "gpu_dp_shared" )
                multiply_dp_gpu_shared( A.get(), B.get(), C.get(), N, trial_durations[trial] );
        }

        std::stringstream timing_file_name;
        timing_file_name << "timing-" << N << "-" << algorithm << ".txt";
        std::FILE * timing_file;
        if( (timing_file = std::fopen( timing_file_name.str().c_str(), "w" )) == nullptr )
        {
            std::cerr << "Failed to open timing file for writing." << std::endl;
            return 1;
        }

        fprintf( timing_file, "%zu ", N );
        for( size_t trial = 0; trial < number_trials; ++trial )
        {
            fprintf( timing_file, "%.15lf ", trial_durations[trial] );
        }

        fprintf( timing_file, "\n" );
        std::fclose( timing_file );

        return 0;
    }
    catch( const TCLAP::ArgException & e )
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }
    catch( const std::exception & e )
    {
        std::cerr << "error: " << e.what() << std::endl;
    }

    return 1;
}
