#include <cuda_runtime.h>
#include <cassert>
#include <cuda_pipeline.h>
#include "gemm.h"
#include "kernels/kernels.h"


constexpr unsigned BLOCKSIZE = 256;
constexpr unsigned BM = 128;
constexpr unsigned BN = 128;
constexpr unsigned BK = 16;
constexpr unsigned TM = 8;
constexpr unsigned TN = 8;


// rt stands for register tiling
__global__ void rt_async(const float *__restrict__ A, const float *__restrict__ B,
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
    __shared__ float As[2][BK][BM + PAD];
    // __align__(16): a 16-byte __pipeline_memcpy_async needs its shared
    // destination 16-byte aligned, which the language does not promise for a
    // plain float array.
    __shared__ __align__(16) float Bs[2][BK][BN];
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

    assert(num_tiles > 0);

    // One tile's worth of B, global -> shared, into buffer `buf`. Pulled out so
    // the prologue and the in-loop prefetch are the same code and cannot drift
    // apart: the only difference between the two calls is the tile index.
    auto prefetch_B = [&](int t, int buf)
    {
        for (int load_offset = 0; load_offset < BK; load_offset += stride_B) // stride_B == BK, so this runs once
        {
            const int k_row = t * BK + B_view_ty + load_offset;
            if ((k_row < K) && ((b_x * BN + B_view_tx * 4) < N))
            {
                __pipeline_memcpy_async(&Bs[buf][B_view_ty + load_offset][B_view_tx * 4],
                                        &B[k_row * N + (b_x * BN + B_view_tx * 4)],
                                        sizeof(float4));
            }
            else
            {
                // Out of range: zero it synchronously. This thread contributes
                // nothing to the batch, which is fine -- an empty batch still
                // counts, and __syncthreads() below makes the zeros visible.
                reinterpret_cast<float4 *>(&Bs[buf][B_view_ty + load_offset][B_view_tx * 4])[0] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
    };

    // As is synchronous and single-buffered, so it is always loaded for the tile
    // about to be computed. Stored transposed, [k][row] instead of [row][k]: the
    // global read is still one float4 of four consecutive k from one row of A,
    // but those four land in four different rows of As, BM+PAD floats apart, so
    // the store is four scalars. That strided store buys a contiguous read in
    // the compute loop, which is paid on every one of the BK steps.
    auto prefetch_A = [&](int t, int buf)
    {
        for (int load_offset = 0; load_offset < BM; load_offset += stride_A) // stride_A == BM, so this runs once
        {
            // The K guard covers all four floats at once: K % 4 == 0 is needed
            // for the float4 load to be aligned anyway, so once the first element
            // is in range the other three are too.
            if (((b_y * BM + load_offset + A_view_ty) < M) && ((t * BK + A_view_tx * 4) < K))
            {
                const float4 temp_A = reinterpret_cast<const float4 *>(&A[(b_y * BM + load_offset + A_view_ty) * K + t * BK + A_view_tx * 4])[0];
                As[buf][A_view_tx * 4 + 0][load_offset + A_view_ty] = temp_A.x;
                As[buf][A_view_tx * 4 + 1][load_offset + A_view_ty] = temp_A.y;
                As[buf][A_view_tx * 4 + 2][load_offset + A_view_ty] = temp_A.z;
                As[buf][A_view_tx * 4 + 3][load_offset + A_view_ty] = temp_A.w;
            }
            else
            {
                As[buf][A_view_tx * 4 + 0][load_offset + A_view_ty] = 0.0f;
                As[buf][A_view_tx * 4 + 1][load_offset + A_view_ty] = 0.0f;
                As[buf][A_view_tx * 4 + 2][load_offset + A_view_ty] = 0.0f;
                As[buf][A_view_tx * 4 + 3][load_offset + A_view_ty] = 0.0f;
            }
        }
    };

    // Prologue: get tile 0's B moving before the loop body ever runs.
    prefetch_B(0, 0);
    prefetch_A(0, 0);
    __pipeline_commit();

    for (int tile = 0; tile < num_tiles; tile++) // step through K with BK steps.
    {
        // Issue the NEXT tile's B into the other buffer. This is the whole point
        // of the double buffer: its copy overlaps this tile's A load and compute.
        // Writing Bs[(tile+1)%2] is safe because the previous iteration's readers
        // of that buffer passed the __syncthreads() at the bottom of the loop.
        if (tile + 1 < num_tiles)
        {
            prefetch_B(tile + 1, (tile + 1) % 2);
            prefetch_A(tile + 1, (tile + 1) % 2);
        }
        // Committed unconditionally, including on the last iteration where the
        // batch is empty: wait_prior(1) counts batches, so skipping this commit
        // would shift the count and make the wait below retire the wrong one.
        __pipeline_commit();

        // Two batches are in flight here -- this tile's and the next one's -- so
        // "leave at most 1 outstanding" retires exactly the one about to be read.
        // wait_prior only covers this thread's own copies; __syncthreads() is what
        // makes every other thread's copies visible to this one.
        __pipeline_wait_prior(1);
        __syncthreads();

        // compute

        for (int kk = 0; kk < BK; ++kk)
        {
            // The payoff of the transpose: As[kk][row .. row+7] is contiguous,
            // so the eight values come back as two float4 reads. PAD must stay a
            // multiple of 4 or &As[kk][row] is no longer 16-byte aligned.
            static_assert(PAD % 4 == 0);
            const float4 a0 = reinterpret_cast<const float4 *>(&As[tile%2][kk][row])[0];     // register_A[0..3]
            const float4 a1 = reinterpret_cast<const float4 *>(&As[tile%2][kk][row + 4])[0]; // register_A[4..7]
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
                register_B[i] = Bs[tile%2][kk][col + i];
            }
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i][j] += register_A[i] * register_B[j];
        }

        __syncthreads();
    }
    // float4 C_vec = reinterpret_cast<float4*>(&acc[0][0])[0];
    for (int i = 0; i < TM; ++i)
    {
        if ((b_y * BM + row + i < M) && (b_x * BN + col < N)) // when N%4=0 the rest of elements in float4 are in bound.
            reinterpret_cast<float4 *>(&C[(b_y * BM + row + i) * N + b_x * BN + col])[0] = reinterpret_cast<float4 *>(&acc[i][0])[0];
        if ((b_y * BM + row + i < M) && (b_x * BN + col + 4 < N))
            reinterpret_cast<float4 *>(&C[(b_y * BM + row + i) * N + b_x * BN + col + 4])[0] = reinterpret_cast<float4 *>(&acc[i][4])[0];
    }
}

void launch_rt_async(const float *dA, const float *dB, float *dC, int M, int K, int N)
{
    assert(M % 4 == 0);
    assert(N % 4 == 0);
    assert(K % 4 == 0);
    dim3 blocksize{BLOCKSIZE};
    dim3 gridsize{(N + BN - 1) / BN, (M + BM - 1) / BM};
    rt_async<<<gridsize, blocksize>>>(dA, dB, dC, M, K, N);
}
