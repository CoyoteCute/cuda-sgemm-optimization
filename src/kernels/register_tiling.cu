#include <cuda_runtime.h>

#include "gemm.h"
#include "kernels/kernels.h"

// With dim3 grid{(N+BN-1)/BN, (M+BM-1)/BM} and dim3 block{256}  a block is responsible for a
// 128x128 tile in C. 128 is 16x8 and it is because each thread is responsible for a 
// 8x8 tile of that 128x128 tile.
// In contrary to k-tiling where rows of As get multiplied to cols of Bs, in register tiling
// cols of As are multiplied by rows of Bs.
// Note loading data to shared memory has nothing to do with this 8x8 tiles of C. For A we load
// chunks of 128x8 and for B chunks of 8x128.
//                     BK=8
//                   ├──────┤
//         ┌─────────┬──────┐            ┌──────────────────────┐
//         │         │      │      BK=8  │        Bs            │  8×128
//    128  │    A    │  As  │ 128        └──────────────────────┘
//         │         │      │                     BN=128
//         └─────────┴──────┘
//              K              all 256 threads cooperate to fill both

//         As: 128×8 = 1024 floats  →  4 per thread (see the loading part)
//         Bs:   8×128 = 1024 floats  →  4 per thread

constexpr unsigned BLOCKSIZE = 256;
constexpr unsigned BM = 128; 
constexpr unsigned BN = 128;
constexpr unsigned BK = 8;
constexpr unsigned TM = 8;
constexpr unsigned TN = 8;

__global__ void register_tiling_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, int M, int K, int N)
{
    constexpr unsigned NUM_THREADS = (BM/TM) * (BN/TN);        // 256
    constexpr unsigned LOADS_A = (BM*BK) / NUM_THREADS;        // 4
    constexpr unsigned LOADS_B = (BK*BN) / NUM_THREADS;        // 4
    constexpr unsigned AS_STRIDE = NUM_THREADS/BK;               // 32
    constexpr unsigned BS_STRIDE = NUM_THREADS/BN;               // 2

    // coordinates in the 16x16 block from 256 threads
    unsigned threadRow = threadIdx.x/16;
    unsigned threadCol = threadIdx.x%16;
    
    unsigned cRow = blockIdx.y; // Because grid is organized in 128x128
    unsigned cCol = blockIdx.x;
    
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    float acc[TM][TN]={};

    int row_b = threadIdx.x / BN;        //threads 0:127 row 0, threads 128:255 row 2
    int col_b = threadIdx.x % BN;
    
    int row_a = threadIdx.x / BK;        //threads 0:31 row 0, threads 32:56 row 1, ... to threads 256-32:255 row 31
    int col_a = threadIdx.x % BK;

    for (int k=0; k<K; k+=BK)  // step through K with BK steps.
    {
        // loading phase, we load chunks of BK from the K direction of A and B.
        // loading B first as it is easy and coalesed
        for (int i=0; i<LOADS_B; i++)             // 4 comes from the fact that we need 8 rows and each thread block produces 2 rows
        {
            Bs[i*BS_STRIDE+row_b][col_b] = B[(k+i*BS_STRIDE+row_b)*N + cCol*BN + col_b];
        }

        // loading As tile
        for (int i=0; i<LOADS_A; i++)           // 4 comes from the fact that we need 128 rows and each thread block produces 32 rows
        {
            As[i*AS_STRIDE+row_a][col_a] = A[(cRow*BM + i*AS_STRIDE + row_a)*K + k + col_a]; 
        }
        __syncthreads();
        
        // compute
        float a[TM]={};
        float b[TN]={};
        for (int kk = 0; kk < BK; ++kk)
        {
            for (int i=0; i<TM; i++)
            {
                a[i] = As[threadRow*TM+i][kk];
            }
            for (int i=0; i<TN; i++)
            {
                b[i] = Bs[kk][threadCol*TN+i];
            }
            
            for (int i=0; i<TM; i++)
                for (int j=0; j<TN; j++)
                    acc[i][j] += a[i]*b[j];
        }

        __syncthreads();

    }
    const unsigned cellRow = cRow*BM + threadRow*TM;
    const unsigned cellCol = cCol*BN + threadCol*TN;
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            C[(cellRow + i)*N + cellCol + j] = acc[i][j];

}


void launch_registerT(const float* dA, const float* dB, float* dC, int M, int K, int N)
{
    dim3 blocksize{BLOCKSIZE};
    dim3 gridsize{(N+BN-1)/BN, (M+BM-1)/BM};
    register_tiling_kernel<<<gridsize, blocksize>>>( dA,  dB,  dC,  M,  K,  N);
}   