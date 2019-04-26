#ifndef __UTILITY_H__
#define __UTILITY_H__

#include <memory>
#include <string>

template<typename TGenerator>
void init_matrix( double * M, size_t N, TGenerator & next_value );

std::unique_ptr<double[]> make_matrix( size_t N );

std::unique_ptr<double[]> make_workspace( size_t num_rows, size_t num_cols );

void print_matrix( double * M, size_t N, const std::string & name );

double zero_gen();

#include "Utility.inl"
#endif
