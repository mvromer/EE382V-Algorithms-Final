#include <functional>
#include <ios>
#include <iostream>
#include <memory>
#include <random>
#include <tclap/CmdLine.h>
#include <algorithm>
#include <iterator>

#include "Utility.h"
#include "MatrixMultiply.h"

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

        TCLAP::UnlabeledValueArg<size_t> matrix_size( "matrix_size",
            "Specifies the size of the N-by-N matrix to compute.",
            true, // required
            2, // default value
            "matrix size" );

        cmd.add( upper );
        cmd.add( seed );
        cmd.add( precision );
        cmd.add( lower );
        cmd.add( integer_entries );
        cmd.add( matrix_size );
        cmd.parse( argc, argv );

        // Allocate matrices.
        const size_t N = matrix_size.getValue();
        const double lowerBound = lower.getValue();
        const double upperBound = upper.getValue();

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
            std::uniform_int_distribution<int> random_distribution( static_cast<int>(lowerBound),
                static_cast<int>(upperBound) );
            auto next_random = std::bind( random_distribution, random_engine );
            init_matrix( A.get(), N, next_random );
            init_matrix( B.get(), N, next_random );
        }
        else
        {
            std::uniform_real_distribution<double> random_distribution( lowerBound, upperBound );
            auto next_random = std::bind( random_distribution, random_engine );
            init_matrix( A.get(), N, next_random );
            init_matrix( B.get(), N, next_random );
        }

        // Run matrix multiply.
        multiply_op( A.get(), B.get(), C.get(), matrix_size.getValue() );
        //strass_serial( A.get(), B.get(), C.get(), matrix_size.getValue(), 2 );

        // Configure output precision.
        std::cout.precision( precision.getValue() );

        // Print results.
        print_matrix( A.get(), matrix_size.getValue(), "A" );
        print_matrix( B.get(), matrix_size.getValue(), "B" );
        print_matrix( C.get(), matrix_size.getValue(), "C" );

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
