#ifndef __GPU_UTILITY_H__
#define __GPU_UTILITY_H__

#include <memory>
#include <cuda_runtime.h>

struct GpuDeleter
{
    void operator()( double * p )
    {
        cudaFree( p );
    }
};

std::unique_ptr<double[], GpuDeleter> make_matrix_gpu( size_t N );

#endif
