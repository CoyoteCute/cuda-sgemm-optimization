#include <cuda_runtime.h>

constexpr unsigned BM = 32;
constexpr unsigned BN = 32;
constexpr unsigned BK = 32;

__global__ void smem_gemm_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, int M, int K, int N)
{
    __shared__ float As[BM*BK];
    __shared__ float Bs[BK*BN];
    int rs = threadIdx.y;
    int cs = threadIdx.x;
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;
    float acc=0.0f;
         
    for (int k=0; k<K; k+=BK)
    {
        As[rs*BK+cs] = (row<M && k+cs<K)? A[row*K + k + cs]: 0.0f;
        Bs[rs*BN+cs] = (k+rs<K && col<N)? B[(k+rs)*N + col]: 0.0f;
        __syncthreads();
        for (int kk=0; kk<BK; kk++)
        {
            acc += As[rs*BK+kk]*Bs[kk*BN+cs];
            
        }
            __syncthreads();
    }
    if (row<M and col<N ) C[row*N+col] = acc;
    
}

void launch_smem(const float* dA, const float* dB, float* dC, int M, int K, int N)
{
    
    dim3 blocksize{BN, BM};
    dim3 gridsize{(N+BN-1)/BN, (M+BM-1)/BM};
    smem_gemm_kernel<<<gridsize, blocksize>>>(dA, dB, dC, M, K, N);
}