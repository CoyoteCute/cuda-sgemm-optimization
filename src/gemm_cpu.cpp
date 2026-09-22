#include "gemm.h"

// i-k-j order: the inner loop walks a row of B and a row of C, both contiguous,
// where i-j-k walked a column of B at a stride of N floats and missed cache on
// every step. Each C[i][j] still sums its k terms in increasing k, so the result
// is bit-identical to the i-j-k version. Rows of C are independent, so they
// split across threads without changing any sum.
void gemm_cpu(const float* A, const float* B, float* C, int M, int K, int N)
{
    #pragma omp parallel for schedule(static)
    for (int i=0; i<M; i++)
    {
        float* c = C + (long)i*N;
        for (int j=0; j<N; j++) c[j] = 0;

        for (int k=0; k<K; k++)
        {
            const float  a = A[(long)i*K + k];
            const float* b = B + (long)k*N;
            for (int j=0; j<N; j++) c[j] += a*b[j];
        }
    }
}
