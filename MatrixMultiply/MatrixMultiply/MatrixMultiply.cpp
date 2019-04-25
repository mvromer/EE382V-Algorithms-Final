#include <functional>
#include <ios>
#include <iostream>
#include <memory>
#include <random>
#include <tclap/CmdLine.h>
#include <algorithm>
#include <iterator>

// Forward declarations.
template<typename TGenerator>
void init_array( double * M, size_t N, TGenerator & next_random );

template<typename TGenerator>
std::unique_ptr<double[]> make_matrix( size_t N, TGenerator & next_random );

void print_matrix( double * M, size_t N, const std::string & name );

double zero_gen();

// Matrix multiply: Dot Product (ijk variant)
void multiply_dp( double * A, double * B, double * C, size_t N )
{
    for ( size_t row = 0; row < N; ++row )
    {
        for ( size_t col = 0; col < N; ++col )
        {
            for (size_t dataIndex = 0; dataIndex < N; ++dataIndex)
            {
                C[row + col * N] = A[row + dataIndex * N] * B[dataIndex + col * N] + C[row + col * N];
            }
        }
    }
}

// Matrix multiply: Saxpy (ikj variant)
void multiply_sp( double * A, double * B, double * C, size_t N )
{
    for ( size_t row = 0; row < N; ++row )
    {
        for (size_t dataIndex = 0; dataIndex < N; ++dataIndex)
        {
            for ( size_t col = 0; col < N; ++col )
            {
                C[row + col * N] = A[row + dataIndex * N] * B[dataIndex + col * N] + C[row + col * N];
            }
        }
    }
}

// Matrix multiply: outer Product (kij)
void multiply_op( double * A, double * B, double * C, size_t N )
{
    for (size_t dataIndex = 0; dataIndex < N; ++dataIndex)
    {
        for ( size_t row = 0; row < N; ++row )
        {
            for ( size_t col = 0; col < N; ++col )
            {
                C[row + col * N] = A[row + dataIndex * N] * B[dataIndex + col * N] + C[row + col * N];
            }
        }
    }
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

        TCLAP::UnlabeledValueArg<size_t> matrix_size( "matrix_size",
            "Specifies the size of the N-by-N matrix to compute.",
            true, // required
            2, // default value
            "matrix size" );

        cmd.add( upper );
        cmd.add( seed );
        cmd.add( precision );
        cmd.add( lower );
        cmd.add( matrix_size );
        cmd.parse( argc, argv );

        // Configure output precision.
        std::cout.precision( precision.getValue() );

        // Setup random number generator.
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

        std::uniform_real_distribution<double> random_distribution( lower.getValue(), upper.getValue() );
        auto next_random = std::bind( random_distribution, random_engine );

        // Create matrices.
        auto A = make_matrix( matrix_size.getValue(), next_random );
        auto B = make_matrix( matrix_size.getValue(), next_random );
        auto C = make_matrix( matrix_size.getValue(), zero_gen );

        // Print results.
        print_matrix( C.get(), matrix_size.getValue(), "ori C" );

        // Run matrix multiply.
        multiply_op( A.get(), B.get(), C.get(), matrix_size.getValue() );

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

// Utility functions.
template<typename TGenerator>
void init_array( double * M, size_t N, TGenerator & num_gen )
{

    for( size_t col = 0; col < N; ++col )
    {
        for( size_t row = 0; row < N; ++row )
        {
            M[row + col * N] = num_gen();
        }
    }
}

template<typename TGenerator>
std::unique_ptr<double[]> make_matrix( size_t N, TGenerator & num_gen )
{
    auto M = std::make_unique<double[]>( N * N );
    init_array( M.get(), N, num_gen );
    return M;
}

void print_matrix( double * M, size_t N, const std::string & name )
{
    std::cout << name << " = [" << std::endl;

    for( size_t row = 0; row < N; ++row )
    {
        std::cout << "  ";

        for( size_t col = 0; col < N; ++col )
        {
            std::cout << M[row + col * N];

            if( col != N - 1 || row != N - 1 )
                std::cout << ", ";
        }

        std::cout << std::endl;
    }

    std::cout << "]" << std::endl;
}

double zero_gen()
{
    return 0;
}
