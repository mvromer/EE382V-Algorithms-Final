#include <iostream>

#include "Utility.h"

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

std::unique_ptr<double[]> make_matrix( size_t N )
{
    return std::make_unique<double[]>( N * N );
}

std::unique_ptr<double[]> make_workspace( size_t num_rows, size_t num_cols )
{
    return std::make_unique<double[]>( num_rows * num_cols );
}
