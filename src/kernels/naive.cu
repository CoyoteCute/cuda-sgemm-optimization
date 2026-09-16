#include <cuda_runtime.h>

#include "gemm.h"
#include "kernels/kernels.h"

// Stage 0 of the ladder. One thread per output element, one full K-pass each.
// Column index is the fast axis so consecutive threads read consecutive B.
__global__ void naive_gemm_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row<M and col<N)
    {
        float acc = 0.0f;
        for (int k=0; k<K; k++)
        {
            acc += A[row*K+k]*B[k*N+col];
        }
        C[row*N+col] = acc;
    }
}

void launch_naive(const float* dA, const float* dB, float* dC, int M, int K, int N)
{
    constexpr unsigned BM = 32;
    constexpr unsigned BN = 32;

    dim3 blocksize{BN, BM};
    dim3 gridsize{(N+BN-1)/BN, (M+BM-1)/BM};
    naive_gemm_kernel<<<gridsize, blocksize>>>(dA, dB, dC, M, K, N);
}
