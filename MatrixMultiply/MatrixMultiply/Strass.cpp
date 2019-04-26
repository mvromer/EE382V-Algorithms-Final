#include <vector>

#include "Utility.h"

namespace
{

using WorkspaceVector = std::vector<std::unique_ptr<double[]>>;

void add( double * A, size_t lda, double * B, size_t ldb, double * C, size_t ldc, size_t N )
{
    for( size_t col = 0; col < N; ++col )
    {
        for( size_t row = 0; row < N; ++row )
        {
            size_t ai = row + col * lda;
            size_t bi = row + col * ldb;
            size_t ci = row + col * ldc;
            C[ci] = A[ai] + B[bi];
        }
    }
}

void subtract( double * A, size_t lda, double * B, size_t ldb, double * C, size_t ldc, size_t N )
{
    for( size_t col = 0; col < N; ++col )
    {
        for( size_t row = 0; row < N; ++row )
        {
            size_t ai = row + col * lda;
            size_t bi = row + col * ldb;
            size_t ci = row + col * ldc;
            C[ci] = A[ai] - B[bi];
        }
    }
}

void copy( double * A, size_t lda, double * C, size_t ldc, size_t N )
{
    for( size_t col = 0; col < N; ++col )
    {
        for( size_t row = 0; row < N; ++row )
        {
            size_t ai = row + col * lda;
            size_t ci = row + col * ldc;
            C[ci] = A[ai];
        }
    }
}

void strass_serial_helper( double * A, size_t lda, double * B, size_t ldb, double * C, size_t ldc,
    size_t N, size_t cutoff,
    WorkspaceVector & workspaces, size_t work_idx, size_t ldwork )
{
    if( N <= cutoff )
    {
        // Do normal matrix multiply.
        for( size_t col = 0; col < N; ++col )
        {
            for( size_t row = 0; row < N; ++row )
            {
                double dot = 0.0;

                for( size_t idx = 0; idx < N; ++idx )
                    dot += A[row + idx * N] * B[idx + col * N];

                C[row + col * N] = dot;
            }
        }
    }
    else
    {
        size_t No2 = N / 2;

        double * workspace = workspaces[work_idx].get();
        double * Pn = workspace;
        double * Asub = workspace + No2 * ldwork;
        double * Bsub = workspace + 2 * No2 * ldwork;

        double * Auu = A;
        double * Avu = A + No2;
        double * Auv = A + No2 * lda;
        double * Avv = A + No2 + No2 * lda;

        double * Buu = B;
        double * Bvu = B + No2;
        double * Buv = B + No2 * ldb;
        double * Bvv = B + No2 + No2 * ldb;

        double * Cuu = C;
        double * Cvu = C + No2;
        double * Cuv = C + No2 * ldc;
        double * Cvv = C + No2 + No2 * ldc;

        // Compute P1.
        add( Auu, lda, Avv, lda, Asub, ldwork, No2 );
        add( Buu, ldb, Bvv, ldb, Bsub, ldwork, No2 );
        strass_serial_helper( Asub, ldwork, Bsub, ldwork, Pn, ldwork, No2, cutoff, workspaces, work_idx + 1, ldwork / 2 );
        copy( Pn, ldwork, Cuu, ldc, No2 ); // Cuu = P1
        copy( Pn, ldwork, Cvv, ldc, No2 ); // Cvv = P1

        // Compute P2.
        add( Avu, lda, Avv, lda, Asub, ldwork, No2 );
        copy( Buu, ldb, Bsub, ldwork, No2 );
        strass_serial_helper( Asub, ldwork, Bsub, ldwork, Pn, ldwork, No2, cutoff, workspaces, work_idx + 1, ldwork / 2 );
        copy( Pn, ldwork, Cvu, ldc, No2 ); // Cvu = P2
        subtract( Cvv, ldc, Pn, ldwork, Cvv, ldc, No2 ); // Cvv = P1 - P2

        // Compute P3.
        copy( Auu, lda, Asub, ldwork, No2 );
        subtract( Buv, ldb, Bvv, ldb, Bsub, ldwork, No2 );
        strass_serial_helper( Asub, ldwork, Bsub, ldwork, Pn, ldwork, No2, cutoff, workspaces, work_idx + 1, ldwork / 2 );
        copy( Pn, ldwork, Cuv, ldc, No2 ); // Cuv = P3
        add( Cvv, ldc, Pn, ldwork, Cvv, ldc, No2 ); // Cvv = P1 - P2 + P3

        // Compute P4.
        copy( Avv, lda, Asub, ldwork, No2 );
        subtract( Bvu, ldb, Buu, ldb, Bsub, ldwork, No2 );
        strass_serial_helper( Asub, ldwork, Bsub, ldwork, Pn, ldwork, No2, cutoff, workspaces, work_idx + 1, ldwork / 2 );
        add( Cuu, ldc, Pn, ldwork, Cuu, ldc, No2 ); // Cuu = P1 + P4
        add( Cvu, ldc, Pn, ldwork, Cvu, ldc, No2 ); // Cvu = P2 + P4

        // Compute P5.
        add( Auu, lda, Auv, lda, Asub, ldwork, No2 );
        copy( Bvv, ldb, Bsub, ldwork, No2 );
        strass_serial_helper( Asub, ldwork, Bsub, ldwork, Pn, ldwork, No2, cutoff, workspaces, work_idx + 1, ldwork / 2 );
        subtract( Cuu, ldc, Pn, ldwork, Cuu, ldc, No2 ); // Cuu = P1 + P4 - P5
        add( Cuv, ldc, Pn, ldwork, Cuv, ldc, No2 ); // Cuv = P3 + P5

        // Compute P6.
        subtract( Avu, lda, Auu, lda, Asub, ldwork, No2 );
        add( Buu, ldb, Buv, ldb, Bsub, ldwork, No2 );
        strass_serial_helper( Asub, ldwork, Bsub, ldwork, Pn, ldwork, No2, cutoff, workspaces, work_idx + 1, ldwork / 2 );
        add( Cvv, ldc, Pn, ldwork, Cvv, ldc, No2 ); // Cvv = P1 - P2 + P3 + P6

        // Compute P7.
        subtract( Auv, lda, Avv, lda, Asub, ldwork, No2 );
        add( Bvu, ldb, Bvv, ldb, Bsub, ldwork, No2 );
        strass_serial_helper( Asub, ldwork, Bsub, ldwork, Pn, ldwork, No2, cutoff, workspaces, work_idx + 1, ldwork / 2 );
        add( Cuu, ldc, Pn, ldwork, Cuu, ldc, No2 ); // Cuu = P1 + P4 - P5 + P7
    }
}

}

void strass_serial( double * A, double * B, double * C, size_t N, size_t cutoff )
{
    WorkspaceVector workspaces;

    size_t workspace_size = N / 2;
    while( workspace_size >= cutoff )
    {
        size_t num_rows = workspace_size;
        size_t num_cols = 3 * workspace_size;
        workspaces.push_back( std::move( make_workspace( num_rows, num_cols ) ) );
        workspace_size /= 2;
    }

    strass_serial_helper( A, N, B, N, C, N, N, cutoff, workspaces, 0, N / 2 );
}