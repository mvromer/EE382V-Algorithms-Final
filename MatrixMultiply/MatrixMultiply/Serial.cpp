#include <chrono>

// Matrix multiply: Dot Product (ijk variant)
void multiply_dp( double * A, double * B, double * C, size_t N, double & duration )
{
    const auto start = std::chrono::steady_clock::now();
    for( size_t row = 0; row < N; ++row )
    {
        for( size_t col = 0; col < N; ++col )
        {
            for( size_t dataIndex = 0; dataIndex < N; ++dataIndex )
            {
                C[row + col * N] = A[row + dataIndex * N] * B[dataIndex + col * N] + C[row + col * N];
            }
        }
    }
    const auto end = std::chrono::steady_clock::now();
    duration = std::chrono::duration<double>( end - start ).count();
}

// Matrix multiply: Saxpy (ikj variant)
void multiply_sp( double * A, double * B, double * C, size_t N, double & duration )
{
    const auto start = std::chrono::steady_clock::now();
    for( size_t row = 0; row < N; ++row )
    {
        for( size_t dataIndex = 0; dataIndex < N; ++dataIndex )
        {
            for( size_t col = 0; col < N; ++col )
            {
                C[row + col * N] = A[row + dataIndex * N] * B[dataIndex + col * N] + C[row + col * N];
            }
        }
    }
    const auto end = std::chrono::steady_clock::now();
    duration = std::chrono::duration<double>( end - start ).count();
}

// Matrix multiply: outer Product (kij)
void multiply_op( double * A, double * B, double * C, size_t N, double & duration )
{
    const auto start = std::chrono::steady_clock::now();
    for( size_t dataIndex = 0; dataIndex < N; ++dataIndex )
    {
        for( size_t row = 0; row < N; ++row )
        {
            for( size_t col = 0; col < N; ++col )
            {
                C[row + col * N] = A[row + dataIndex * N] * B[dataIndex + col * N] + C[row + col * N];
            }
        }
    }
    const auto end = std::chrono::steady_clock::now();
    duration = std::chrono::duration<double>( end - start ).count();
}
