#include <iostream>
#include "GpuUtility.h"

std::unique_ptr<double[], GpuDeleter> make_matrix_gpu( size_t N )
{
    double * p;
    if( cudaMallocManaged( &p, sizeof( double ) * N * N ) != cudaSuccess )
    {
        std::cout << "Failed to allocate CUDA memory." << std::endl;
        return std::unique_ptr<double[], GpuDeleter>();
    }
    return std::unique_ptr<double[], GpuDeleter>( p );
}
