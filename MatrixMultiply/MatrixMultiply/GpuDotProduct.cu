#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <cuda_runtime.h>

#define THREADS_PER_DIM 32

namespace
{

__global__ void multiply_dp_gpu_kernel( double * A, double * B, double * C, size_t N )
{
    double dot = 0.0;
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if( row < N && col < N )
    {
        for( size_t idx = 0; idx < N; ++idx )
            dot += A[row + idx * N] * B[idx + col * N];

        C[row + col * N] = dot;
    }
}

__global__ void multiply_dp_gpu_shared_kernel( double * A, double * B, double * C, size_t N, size_t dp_length )
{
    const size_t block_row = blockIdx.y;
    const size_t block_col = blockIdx.x;

    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if( row < N && col < N )
    {
        double dot = 0.0;
        const size_t number_subblocks = ceil( static_cast<double>(N) / static_cast<double>(blockDim.x) );

        __shared__ double Asub[THREADS_PER_DIM * THREADS_PER_DIM];
        __shared__ double Bsub[THREADS_PER_DIM * THREADS_PER_DIM];

        for( size_t subblock = 0; subblock < number_subblocks; ++subblock )
        {
            size_t Arow = block_row * blockDim.y + threadIdx.y;
            size_t Acol = subblock * blockDim.x + threadIdx.x;
            size_t Brow = subblock * blockDim.y + threadIdx.y;
            size_t Bcol = block_col * blockDim.x + threadIdx.x;

            Asub[threadIdx.x * THREADS_PER_DIM + threadIdx.y] = A[Arow + Acol * N];
            Bsub[threadIdx.x * THREADS_PER_DIM + threadIdx.y] = B[Brow + Bcol * N];

            __syncthreads();

            for( size_t idx = 0; idx < dp_length; ++idx )
                dot += Asub[idx * THREADS_PER_DIM + threadIdx.y] * Bsub[threadIdx.x * THREADS_PER_DIM + idx];

            __syncthreads();
        }

        C[row + col * N] = dot;
    }
}

}

void multiply_dp_gpu( double * A, double * B, double * C, size_t N, double & duration )
{
    const size_t threads_per_dim = THREADS_PER_DIM;
    const size_t min_blocks_per_dim = 1;
    const size_t blocks_per_dim = std::max( N / threads_per_dim, min_blocks_per_dim );

    dim3 block_size( threads_per_dim, threads_per_dim );
    dim3 grid_size( blocks_per_dim, blocks_per_dim );

    cudaGetLastError();

    const auto start = std::chrono::steady_clock::now();
    multiply_dp_gpu_kernel<<< grid_size, block_size >>>( A, B, C, N );
    cudaDeviceSynchronize();
    const auto end = std::chrono::steady_clock::now();
    duration = std::chrono::duration<double>( end - start ).count();

    std::cout << "N = " << N << ": Last CUDA error: " << cudaGetLastError() << std::endl;
}

void multiply_dp_gpu_shared( double * A, double * B, double * C, size_t N, double & duration )
{
    const size_t threads_per_dim = THREADS_PER_DIM;
    const size_t min_blocks_per_dim = 1;
    const size_t blocks_per_dim = std::max( N / threads_per_dim, min_blocks_per_dim );
    const size_t dp_length = std::min( N, threads_per_dim );

    dim3 block_size( threads_per_dim, threads_per_dim );
    dim3 grid_size( blocks_per_dim, blocks_per_dim );

    cudaGetLastError();

    const auto start = std::chrono::steady_clock::now();
    multiply_dp_gpu_shared_kernel<<< grid_size, block_size >>>( A, B, C, N, dp_length );
    cudaDeviceSynchronize();
    const auto end = std::chrono::steady_clock::now();
    duration = std::chrono::duration<double>( end - start ).count();

    std::cout << "N = " << N << ": Last CUDA error: " << cudaGetLastError() << std::endl;
}
