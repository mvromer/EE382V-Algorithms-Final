#include <algorithm>
#include <functional>
#include <ios>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <stdexcept>
#include <tclap/CmdLine.h>

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

        TCLAP::UnlabeledValueArg<size_t> matrix_size( "matrix_size",
            "Specifies the size of the N-by-N matrix to compute. Must be power of 2.",
            true, // required
            2, // default value
            "matrix size" );

        cmd.add( upper );
        cmd.add( seed );
        cmd.add( precision );
        cmd.add( lower );
        cmd.add( integer_entries );
        cmd.add( cutoff );
        cmd.add( matrix_size );
        cmd.parse( argc, argv );

        // Extract command line values.
        const size_t N = matrix_size.getValue();
        const size_t strassen_cutoff = cutoff.getValue();
        const double lower_bound = lower.getValue();
        const double upper_bound = upper.getValue();

        // Make sure size and cutoff values are powers of two.
        if( !(is_power_of_two( N ) && is_power_of_two( strassen_cutoff )) )
            throw std::invalid_argument( "Size and cutoff (if given) must be powers of 2." );

        // Allocate matrices.
        auto A = make_matrix( N );
        auto B = make_matrix( N );
        auto C = make_matrix( N );

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
        //multiply_op( A.get(), B.get(), C.get(), N );
        strass_serial( A.get(), B.get(), C.get(), N, strassen_cutoff );

        // Configure output precision.
        std::cout.precision( precision.getValue() );

        // Print results.
        print_matrix( A.get(), N, "A" );
        print_matrix( B.get(), N, "B" );
        print_matrix( C.get(), N, "C" );

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
