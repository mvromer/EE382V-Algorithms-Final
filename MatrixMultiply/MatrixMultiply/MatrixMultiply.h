#ifndef __MATRIX_MULTIPLY_H__
#define __MATRIX_MULTIPLY_H__

void multiply_dp( double * A, double * B, double * C, size_t N, double & duration );
void multiply_sp( double * A, double * B, double * C, size_t N, double & duration );
void multiply_op( double * A, double * B, double * C, size_t N, double & duration );
void strass_serial( double * A, double * B, double * C, size_t N, size_t cutoff, double & duration );

void multiply_dp_gpu( double * A, double * B, double * C, size_t N, double & duration );
void multiply_dp_gpu_shared( double * A, double * B, double * C, size_t N, double & duration );

#endif
