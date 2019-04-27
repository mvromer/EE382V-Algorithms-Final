#ifndef __MATRIX_MULTIPLY_H__
#define __MATRIX_MULTIPLY_H__

void multiply_dp( double * A, double * B, double * C, size_t N );
void multiply_sp( double * A, double * B, double * C, size_t N );
void multiply_op( double * A, double * B, double * C, size_t N );
void strass_serial( double * A, double * B, double * C, size_t N, size_t cutoff );

void multiply_dp_gpu( double * A, double * B, double * C, size_t N );
void multiply_dp_gpu_shared( double * A, double * B, double * C, size_t N );

#endif
