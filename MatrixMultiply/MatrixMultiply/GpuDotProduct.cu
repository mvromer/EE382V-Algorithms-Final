#include <algorithm>
#include <cstddef>
#include <cuda_runtime.h>

namespace
{

__global__ void multiply_dp_gpu_kernel( double * A, double * B, double * C, size_t N )
{
    double dot = 0.0;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if( row < N && col < N )
    {
        for( size_t idx = 0; idx < N; ++idx )
            dot += A[row + idx * N] * B[idx + col * N];

        C[row + col * N] = dot;
    }
}

}

void multiply_dp_gpu( double * A, double * B, double * C, size_t N )
{
    const size_t min_number_blocks = 1;
    dim3 block_size( 16, 16 );
    dim3 grid_size( std::max( N / block_size.x, min_number_blocks ), std::max( N / block_size.y, min_number_blocks ) );
    multiply_dp_gpu_kernel<<< block_size, grid_size >>>( A, B, C, N );
    cudaDeviceSynchronize();
}