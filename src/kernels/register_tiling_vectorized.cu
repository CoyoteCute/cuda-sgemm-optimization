#include <cuda_runtime.h>
#include <cassert>
#include "gemm.h"
#include "kernels/kernels.h"

// With dim3 grid{(N+BN-1)/BN, (M+BM-1)/BM} and dim3 block{256}  a block of 16x16 threads is responsible for a
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
__global__ void rt_kernel(const float *__restrict__ A, const float *__restrict__ B,
                          float *__restrict__ C, int M, int K, int N)
{
    constexpr unsigned NUM_THREADS = (BM / TM) * (BN / TN); // 256
    constexpr unsigned stride_A = NUM_THREADS / BK;         // 32
    constexpr unsigned stride_B = NUM_THREADS / BN;         // 2
    const unsigned num_tiles = (K + BK - 1) / BK;           // number of tiles in K direction

    // The 256 threads form a 16x16 grid (BN/TN = 16 per row, 256/16 = 16 rows).
    // Scaled by TM and TN, these give the top-left
    // element of this thread's 8x8 piece of the block's 128x128 tile of C.
    unsigned row = TM * (threadIdx.x / (BN / TN)); // 0, 8, 16, ... 120
    unsigned col = TN * (threadIdx.x % (BN / TN)); // 0, 8, 16, ... 120

    unsigned b_y = blockIdx.y; // Because grid is organized in 128x128
    unsigned b_x = blockIdx.x;
    // padding to avoid bank confilicts in As writes
    constexpr unsigned PAD = 0;
    __shared__ float As[BM + PAD][BK];
    __shared__ float Bs[BK][BN];
    float acc[TM][TN] = {};
    // B_view_ty and B_view_tx determine which element of the BKxBN tile this thread will load.
    int B_view_ty = threadIdx.x / BN; // threads 0:127 row 0, threads 128:255 row 2
    int B_view_tx = threadIdx.x % BN;
    // A_view_ty and A_view_tx determine which element of the BMxBK tile this thread will load.
    int A_view_ty = threadIdx.x / BK; // threads 0:31 row 0, threads 32:56 row 1, ... to threads 256-32:255 row 31
    int A_view_tx = threadIdx.x % BK;
    float register_A[TM] = {};
    float register_B[TN] = {};

    for (int tile = 0; tile < num_tiles; tile++) // step through K with BK steps.
    {
        // loading phase, we load chunks of BK from the K direction of A and B.
        // loading B first as it is easy and coalesed
        // load_offset represents the starting point within the BK chunk for this thread to load.
        for (int load_offset = 0; load_offset < BK; load_offset += stride_B) 
        {
            // B_view_ty and B_view_tx determine which element of the BKxBN tile this thread will load.
            if (((tile * BK + load_offset + B_view_ty) < K) && (b_x * BN + B_view_tx < N))
            {
                Bs[load_offset + B_view_ty][B_view_tx] = B[(tile * BK + load_offset + B_view_ty) * N + b_x * BN + B_view_tx];
            }
            else
            {
                Bs[load_offset + B_view_ty][B_view_tx] = 0.0f;
            }
        }

        for (int load_offset = 0; load_offset < BM; load_offset += stride_A) // 4 comes from the fact that we need 128 rows and each thread block produces 32 rows
        {
            if (((b_y * BM + load_offset + A_view_ty) < M) && ((tile * BK + A_view_tx) < K))
            {
                As[load_offset + A_view_ty][A_view_tx] = A[(b_y * BM + load_offset + A_view_ty) * K + tile * BK + A_view_tx];
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
            for (int i = 0; i < TM; i++)
            {
                register_A[i] = As[row + i][kk];
            }
            // const float4 a0 = reinterpret_cast<float4*>(&As[threadRow*TM][kk])[0];    //a[0..3]
            // const float4 a1 = reinterpret_cast<float4*>(&As[threadRow*TM+4][kk])[0];  //a[4..7]
            // a[0]=a0.x; a[1]=a0.y; a[2]=a0.z; a[3]=a0.w;
            // a[4]=a1.x; a[5]=a1.y; a[6]=a1.z; a[7]=a1.w;
            for (int i = 0; i < TN; i++)
            {
                register_B[i] = Bs[kk][col + i];
            }
            // const float4 b0 = reinterpret_cast<float4*>(&Bs[kk][threadCol*TN])[0];    //register_B[0..3]
            // const float4 b1 = reinterpret_cast<float4*>(&Bs[kk][threadCol*TN+4])[0];  //register_B[4..7]
            // register_B[0]=b0.x; register_B[1]=b0.y; register_B[2]=b0.z; register_B[3]=b0.w;
            // register_B[4]=b1.x; register_B[5]=b1.y; register_B[6]=b1.z; register_B[7]=b1.w;
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i][j] += register_A[i] * register_B[j];
        }

        __syncthreads();
    }
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j)
            if ((b_y * BM + row + i < M) && (b_x * BN + col + j < N))
                C[(b_y * BM + row + i) * N + b_x * BN + col + j] = acc[i][j];
}

void launch_rt(const float *dA, const float *dB, float *dC, int M, int K, int N)
{
    dim3 blocksize{BLOCKSIZE};
    dim3 gridsize{(N + BN - 1) / BN, (M + BM - 1) / BM};
    rt_kernel<<<gridsize, blocksize>>>(dA, dB, dC, M, K, N);
}

// rt stands for register tiling
__global__ void rt_vectorized_AsBs(const float *__restrict__ A, const float *__restrict__ B,
                                   float *__restrict__ C, int M, int K, int N)
{
    constexpr unsigned NUM_THREADS = (BM / TM) * (BN / TN); // 256
    static_assert(NUM_THREADS % BK == 0);
    static_assert(NUM_THREADS % BN == 0);
    static_assert(BK % 4 == 0);
    static_assert(BN % 4 == 0);

    // Rows of the tile covered by one pass of the loading loop. With float4
    // loads, BK/4 = 2 threads fill a row of As and BN/4 = 32 fill a row of Bs,
    // so 256 threads cover 128 rows of As and 8 rows of Bs: one pass each.
    constexpr unsigned stride_A = NUM_THREADS / (BK / 4); // 128 = BM
    constexpr unsigned stride_B = NUM_THREADS / (BN / 4); // 8   = BK
    const unsigned num_tiles = (K + BK - 1) / BK;         // number of tiles in K direction

    // Top-left corner of this thread's 8x8 piece of C, inside the 128x128 tile.
    // BN/TN = 16 threads per row, and 256/16 = 16 such rows.
    unsigned row = TM * (threadIdx.x / (BN / TN)); // 0, 8, 16, ... 120
    unsigned col = TN * (threadIdx.x % (BN / TN)); // 0, 8, 16, ... 120

    unsigned b_y = blockIdx.y; // Because grid is organized in 128x128
    unsigned b_x = blockIdx.x;
    // As is transposed: [k][row], so a row of it is 128 consecutive rows of A.
    // PAD spreads the strided stores across banks; keep it a multiple of 4 so
    // the float4 reads in the compute loop stay 16-byte aligned.
    constexpr unsigned PAD = 0;
    __shared__ float As[BK][BM + PAD];
    __shared__ float Bs[BK][BN];
    float acc[TM][TN] = {};

    // Loading coordinates, in units of float4 along the row. Bs is 8x128, so
    // 32 threads cover one row: threads 0:31 row 0, 32:63 row 1, ... 224:255 row 7.
    int B_view_ty = threadIdx.x / (BN / 4); // 0..7   , the row of Bs
    int B_view_tx = threadIdx.x % (BN / 4); // 0..31  , which float4 in it

    // A's tile is 128 rows x 8 k, and 2 threads cover one row of it: threads
    // 0,1 row 0, 2,3 row 1, ... 254,255 row 127. These name the position in A,
    // not in As -- As holds the transpose, so the two subscripts swap on store.
    int A_view_ty = threadIdx.x / (BK / 4); // 0..127 , the row of A's tile
    int A_view_tx = threadIdx.x % (BK / 4); // 0 or 1 , which float4 of k
    float register_A[TM] = {};
    float register_B[TN] = {};

    for (int tile = 0; tile < num_tiles; tile++) // step through K with BK steps.
    {
        // loading phase, we load chunks of BK from the K direction of A and B.
        // loading B first as it is easy and coalesed
        for (int load_offset = 0; load_offset < BK; load_offset += stride_B) // stride_B == BK, so this runs once
        {
            // if (((tile * BK + load_offset + B_view_ty) < K) && (b_x * BN + B_view_tx < N))
            // {
            //     Bs[load_offset + B_view_ty][B_view_tx] = B[(tile*BK+load_offset + B_view_ty)*N + b_x*BN + B_view_tx];
            // }
            // else
            // {
            //     Bs[load_offset + B_view_ty][B_view_tx] = 0.0f;
            // }
            if (((tile * BK + B_view_ty + load_offset) < K) && (((b_x * BN + B_view_tx * 4)) < N))
            {
                const float4 temp_B = reinterpret_cast<const float4 *>(&B[(tile * BK + B_view_ty + load_offset) * N + ((b_x * BN + B_view_tx * 4))])[0];
                reinterpret_cast<float4 *>(&Bs[B_view_ty + load_offset][B_view_tx * 4])[0] = temp_B;
            }
            else
            {
                reinterpret_cast<float4 *>(&Bs[B_view_ty + load_offset][B_view_tx * 4])[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }

        // As is stored transposed, [k][row] instead of [row][k]. The global read
        // is still one float4 of four consecutive k values from one row of A,
        // but in shared memory those four now land in four different rows of As,
        // BM+PAD floats apart, so the store is four scalars instead of a float4.
        // That is the trade: a strided store here buys a contiguous read in the
        // compute loop below, where it is paid for on every one of the BK steps.
        for (int load_offset = 0; load_offset < BM; load_offset += stride_A) // stride_A == BM, so this runs once
        {
            // The K guard covers all four floats at once: K % 4 == 0 is needed
            // for the float4 load to be aligned anyway, so once the first element
            // is in range the other three are too.
            if (((b_y * BM + load_offset + A_view_ty) < M) && ((tile * BK + A_view_tx * 4) < K))
            {
                const float4 temp_A = reinterpret_cast<const float4 *>(&A[(b_y * BM + load_offset + A_view_ty) * K + tile * BK + A_view_tx * 4])[0];
                As[A_view_tx * 4 + 0][load_offset + A_view_ty] = temp_A.x;
                As[A_view_tx * 4 + 1][load_offset + A_view_ty] = temp_A.y;
                As[A_view_tx * 4 + 2][load_offset + A_view_ty] = temp_A.z;
                As[A_view_tx * 4 + 3][load_offset + A_view_ty] = temp_A.w;
            }
            else
            {
                As[A_view_tx * 4 + 0][load_offset + A_view_ty] = 0.0f;
                As[A_view_tx * 4 + 1][load_offset + A_view_ty] = 0.0f;
                As[A_view_tx * 4 + 2][load_offset + A_view_ty] = 0.0f;
                As[A_view_tx * 4 + 3][load_offset + A_view_ty] = 0.0f;
            }
        }
        __syncthreads();

        // compute

        for (int kk = 0; kk < BK; ++kk)
        {
            // The payoff of the transpose: As[kk][row .. row+7] is contiguous,
            // so the eight values come back as two float4 reads. PAD must stay a
            // multiple of 4 or &As[kk][row] is no longer 16-byte aligned.
            static_assert(PAD % 4 == 0);
            const float4 a0 = reinterpret_cast<const float4 *>(&As[kk][row])[0];     // register_A[0..3]
            const float4 a1 = reinterpret_cast<const float4 *>(&As[kk][row + 4])[0]; // register_A[4..7]
            register_A[0] = a0.x;
            register_A[1] = a0.y;
            register_A[2] = a0.z;
            register_A[3] = a0.w;
            register_A[4] = a1.x;
            register_A[5] = a1.y;
            register_A[6] = a1.z;
            register_A[7] = a1.w;
            for (int i = 0; i < TN; i++)
            {
                register_B[i] = Bs[kk][col + i];
            }
            // const float4 b0 = reinterpret_cast<float4*>(&Bs[kk][threadCol*TN])[0];    //register_B[0..3]
            // const float4 b1 = reinterpret_cast<float4*>(&Bs[kk][threadCol*TN+4])[0];  //register_B[4..7]
            // register_B[0]=b0.x; register_B[1]=b0.y; register_B[2]=b0.z; register_B[3]=b0.w;
            // register_B[4]=b1.x; register_B[5]=b1.y; register_B[6]=b1.z; register_B[7]=b1.w;
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i][j] += register_A[i] * register_B[j];
        }

        __syncthreads();
    }
    // for (int i = 0; i < TM; ++i)
    //     for (int j = 0; j < TN; ++j)
    //         if ((b_y * BM + row + i < M) && (b_x * BN + col + j < N))
    //             C[(b_y*BM + row + i)*N + b_x*BN + col + j] = acc[i][j];

    // float4 C_vec = reinterpret_cast<float4*>(&acc[0][0])[0];
    for (int i = 0; i < TM; ++i)
    {
        if ((b_y * BM + row + i < M) && (b_x * BN + col < N)) // when N%4=0 the rest of elements in float4 are in bound.
            reinterpret_cast<float4 *>(&C[(b_y * BM + row + i) * N + b_x * BN + col])[0] = reinterpret_cast<float4 *>(&acc[i][0])[0];
        if ((b_y * BM + row + i < M) && (b_x * BN + col + 4 < N))
            reinterpret_cast<float4 *>(&C[(b_y * BM + row + i) * N + b_x * BN + col + 4])[0] = reinterpret_cast<float4 *>(&acc[i][4])[0];
    }
}

void launch_rt_V_AsBs(const float *dA, const float *dB, float *dC, int M, int K, int N)
{
    assert(M % 4 == 0);
    assert(N % 4 == 0);
    assert(K % 4 == 0);
    dim3 blocksize{BLOCKSIZE};
    dim3 gridsize{(N + BN - 1) / BN, (M + BM - 1) / BM};
    rt_vectorized_AsBs<<<gridsize, blocksize>>>(dA, dB, dC, M, K, N);
}
