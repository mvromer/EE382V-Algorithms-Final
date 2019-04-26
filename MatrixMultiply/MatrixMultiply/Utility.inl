#include "Utility.h"

template<typename TGenerator>
void init_matrix( double * M, size_t N, TGenerator & next_value )
{

    for( size_t col = 0; col < N; ++col )
    {
        for( size_t row = 0; row < N; ++row )
        {
            M[row + col * N] = next_value();
        }
    }
}
