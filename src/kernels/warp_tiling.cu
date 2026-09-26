// ===========================================================================
// PLACEHOLDER -- DO NOT USE. NOT A WARP-TILED KERNEL YET.
//
// wt_kernel below is still a verbatim copy of rt_kernel from
// register_tiling_vectorized.cu, renamed so the two do not collide at link
// time. Warp tiling is not implemented: there is no warp-level decomposition
// here at all, and the WARPS_PER_BLOCK / LANES_M constants are declared but
// unused. It measures exactly what `rt` measures, because it IS `rt`.
//
// Its registry entry in src/registry.cu is commented out on purpose, so `wt`
// does not show up in test_gemm runs and cannot be mistaken for a result.
// Uncomment that line only once this file actually differs from rt_kernel.
// ===========================================================================

#include <cuda_runtime.h>
#include <cassert>
#include "gemm.h"
#include "kernels/kernels.h"

// With dim3 grid{(N+BN-1)/BN, (M+BM-1)/BM} and dim3 block{256} a block is 128x128 in C. 
// The warp tiling approach organizes computation at the warp level rather than the thread level.
// The 128x128 tile in C is broken into 2x4 sub-tiles each handled by a warp. 
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

//         As: 128×8 = 1024 floats  →  4 per thread, as one float4 (see the loading part)
//         Bs:   8×128 = 1024 floats  →  4 per thread, as one float4
// Both tiles are filled with 128-bit loads, so each of the 256 threads issues
// exactly one LDG.128 per tile per matrix, and each loading loop runs once.
// In rt_vectorized_Bs, As is held transposed as [BK][BM]: the store into it is
// strided, but the compute loop then reads As[kk][row..row+7] as two float4s.

constexpr unsigned BLOCKSIZE = 256;
constexpr unsigned BM = 128; 
constexpr unsigned BN = 128;
constexpr unsigned BK = 16;
constexpr unsigned TM = 8;
constexpr unsigned TN = 8;

// this is basically the same as register_tiling_kernel. added some controls for ragged matrices.
__global__ void wt_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, int M, int K, int N)
{
    constexpr unsigned NUM_THREADS = (BM/TM) * (BN/TN);        // 256
    constexpr unsigned stride_A = NUM_THREADS/BK;               // 32
    constexpr unsigned stride_B = NUM_THREADS/BN;               // 2
    const unsigned num_tiles = (K+BK-1)/BK; // number of tiles in K direction
    // warp parameters
    constexpr unsigned WARPS_PER_BLOCK = BLOCKSIZE / 32; // 8 warps per block
    constexpr unsigned WARPS_M = 2;           // 8 warps, 2x4
    constexpr unsigned WARPS_N = 4;           
    constexpr unsigned WM = BM / WARPS_M;             // 64
    constexpr unsigned WN = BN / WARPS_N;             // 32
    constexpr unsigned LANES_M = WM / TM;             // 8
    constexpr unsigned LANES_N = WN / TN;             // 4   (8×4 = 32 ✓)
    const unsigned warpId = threadIdx.x / 32;
    const unsigned laneId = threadIdx.x % 32;  
    // C coordinates
    const unsigned row = (warpId / WARPS_N) * WM + (laneId / LANES_N) * TM;
    const unsigned col = (warpId % WARPS_N) * WN + (laneId % LANES_N) * TN;

    unsigned b_y = blockIdx.y; // Because grid is organized in 128x128
    unsigned b_x = blockIdx.x;
    // Loading from dram to shared memory params and registers
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int B_view_ty = threadIdx.x / BN;        //threads 0:127 row 0, threads 128:255 row 2
    int B_view_tx = threadIdx.x % BN;
    int A_view_ty = threadIdx.x / BK;        //threads 0:31 row 0, threads 32:56 row 1, ... to threads 256-32:255 row 31
    int A_view_tx = threadIdx.x % BK;
    // accumulator and register storage for this thread
    float acc[TM][TN]={};
    float register_A[TM]={};
    float register_B[TN]={};
    
    // loading phase. the same as register tiling.
    for (int tile=0; tile<num_tiles; tile++)  // step through K with BK steps. this is reduction over K.
    {
        for (int load_offset=0; load_offset<BK; load_offset+=stride_B)  
        {   if (((tile * BK + load_offset + B_view_ty) < K) && (b_x * BN + B_view_tx < N))
            {
                Bs[load_offset + B_view_ty][B_view_tx] = B[(tile*BK+load_offset + B_view_ty)*N + b_x*BN + B_view_tx];
            }
            else
            {
                Bs[load_offset + B_view_ty][B_view_tx] = 0.0f;
            }
        }

        
        for (int load_offset=0; load_offset<BM; load_offset+=stride_A)           // 4 comes from the fact that we need 128 rows and each thread block produces 32 rows
        {
            if (((b_y * BM + load_offset + A_view_ty) < M) && ((tile * BK + A_view_tx) < K))
            {
                As[load_offset + A_view_ty][A_view_tx] = A[(b_y*BM + load_offset + A_view_ty)*K + tile*BK + A_view_tx];
            }
            else
            {
                As[load_offset + A_view_ty][A_view_tx] = 0.0f;
            }
        }
        __syncthreads();
        
        // compute
        
        for (int kk = 0; kk < BK; ++kk)
        {
            for (int i=0; i<TM; i++)
            {
                register_A[i] = As[row+i][kk];
            }
            // const float4 a0 = reinterpret_cast<float4*>(&As[threadRow*TM][kk])[0];    //a[0..3]
            // const float4 a1 = reinterpret_cast<float4*>(&As[threadRow*TM+4][kk])[0];  //a[4..7]
            // a[0]=a0.x; a[1]=a0.y; a[2]=a0.z; a[3]=a0.w;
            // a[4]=a1.x; a[5]=a1.y; a[6]=a1.z; a[7]=a1.w;
            for (int i=0; i<TN; i++)
            {
                register_B[i] = Bs[kk][col+i];
            }
            // const float4 b0 = reinterpret_cast<float4*>(&Bs[kk][threadCol*TN])[0];    //register_B[0..3]
            // const float4 b1 = reinterpret_cast<float4*>(&Bs[kk][threadCol*TN+4])[0];  //register_B[4..7]
            // register_B[0]=b0.x; register_B[1]=b0.y; register_B[2]=b0.z; register_B[3]=b0.w;
            // register_B[4]=b1.x; register_B[5]=b1.y; register_B[6]=b1.z; register_B[7]=b1.w;
            for (int i=0; i<TM; i++)
                for (int j=0; j<TN; j++)
                    acc[i][j] += register_A[i]*register_B[j];
        }

        __syncthreads();

    }
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            if ((b_y * BM + row + i < M) && (b_x * BN + col + j < N))
                C[(b_y*BM + row + i)*N + b_x*BN + col + j] = acc[i][j];

}


void launch_wt(const float* dA, const float* dB, float* dC, int M, int K, int N)
{
    dim3 blocksize{BLOCKSIZE};
    dim3 gridsize{(N+BN-1)/BN, (M+BM-1)/BM};
    wt_kernel<<<gridsize, blocksize>>>( dA,  dB,  dC,  M,  K,  N);
}   
 





