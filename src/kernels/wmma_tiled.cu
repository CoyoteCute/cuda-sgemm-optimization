// Templated tensor-core kernel: the sgemm optimizations that carry over to WMMA.
//
// Three nested levels, which is the whole idea:
//
//   block tile   BM x BN   staged in shared memory, BK deep
//   warp tile    WM x WN   one warp's share of the block tile
//   fragment     16 x 16   what mma_sync does, K=8 at a time for tf32
//
// Each warp keeps (WM/16)*(WN/16) accumulator fragments live across the entire K
// loop -- the same trick as the 8x8 register tile in rt_async, one level up. The
// reuse comes from that: one `a` fragment feeds WN/16 mma calls and one `b`
// fragment feeds WM/16, so fragment loads from shared are amortized over the
// warp tile instead of being one-shot like in wmma_tf32.cu.
//
// Carried over from the sgemm ladder:
//   - shared-memory staging, so fragments load from smem rather than global
//   - float4 global loads (LDG.128), one per thread per row chunk
//   - PAD as a knob on the shared row stride, kept a multiple of 4 so the
//     float4 stores stay 16-byte aligned. Note PAD=4 measurably HURT the sgemm
//     kernels (see "PAD does not pay here" in the README), so it defaults to 0
//     and is a template parameter to be swept rather than assumed.
//
// Deliberately NOT here yet: __pipeline_memcpy_async and double buffering. Those
// are a separate axis worth ~7-9% on rt_async, and changing the tiling structure
// and the load mechanism in one step makes the result unattributable.
//
// No partial-tile handling: load_matrix_sync cannot mask. The launcher asserts
// the shape divides evenly, which is what keeps these out of the ragged case.
#include <cuda_runtime.h>
#include <cassert>
#include <mma.h>

#include "gemm.h"
#include "kernels/kernels.h"

using namespace nvcuda;

namespace
{
    // tf32 fragments are m16n16k8. 16x16x16 does not exist for tf32, so the
    // fragment K is 8 and BK must be a multiple of it.
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 8;

    template <int BM, int BN, int BK, int WM, int WN, int PAD = 0>
    __global__ void wmma_tiled_kernel(const float *__restrict__ A, const float *__restrict__ B,
                                      float *__restrict__ C, int M, int K, int N)
    {
        // ---- shape arithmetic, all compile-time ----
        constexpr int WARPS_M = BM / WM;
        constexpr int WARPS_N = BN / WN;
        constexpr int NUM_WARPS = WARPS_M * WARPS_N;
        constexpr int NUM_THREADS = NUM_WARPS * 32;

        constexpr int FRAG_M = WM / WMMA_M;   // accumulator fragments down the warp tile
        constexpr int FRAG_N = WN / WMMA_N;   // ... and across it

        constexpr int LDA = BK + PAD;         // shared row strides
        constexpr int LDB = BN + PAD;

        static_assert(BM % WM == 0 && BN % WN == 0, "block tile must divide into warp tiles");
        static_assert(WM % WMMA_M == 0 && WN % WMMA_N == 0, "warp tile must divide into 16x16 fragments");
        static_assert(BK % WMMA_K == 0, "BK must be a multiple of the tf32 fragment K (8)");
        static_assert(BK % 4 == 0 && BN % 4 == 0, "float4 staging needs BK and BN divisible by 4");
        static_assert(PAD % 4 == 0, "PAD must keep the float4 shared stores 16-byte aligned");
        static_assert((BM * BK) % (NUM_THREADS * 4) == 0, "A tile must divide evenly into float4 per thread");
        static_assert((BK * BN) % (NUM_THREADS * 4) == 0, "B tile must divide evenly into float4 per thread");
        static_assert((BM * LDA + BK * LDB) * sizeof(float) <= 48 * 1024,
                      "static __shared__ is capped at 48 KB on sm_86");

        __shared__ __align__(16) float As[BM * LDA];   // [BM][BK+PAD], row-major
        __shared__ __align__(16) float Bs[BK * LDB];   // [BK][BN+PAD], row-major

        const int tid = threadIdx.x;
        const int warp = tid / 32;
        const int warp_m = warp / WARPS_N;   // this warp's position in the block tile
        const int warp_n = warp % WARPS_N;

        const int block_row = blockIdx.y * BM;
        const int block_col = blockIdx.x * BN;

        // ---- staging coordinates, in units of float4 along the row ----
        constexpr int A_VEC_PER_ROW = BK / 4;                  // threads to fill one row of As
        constexpr int A_ROW_STRIDE = NUM_THREADS / A_VEC_PER_ROW;
        const int a_row = tid / A_VEC_PER_ROW;
        const int a_col = (tid % A_VEC_PER_ROW) * 4;

        constexpr int B_VEC_PER_ROW = BN / 4;
        constexpr int B_ROW_STRIDE = NUM_THREADS / B_VEC_PER_ROW;
        const int b_row = tid / B_VEC_PER_ROW;
        const int b_col = (tid % B_VEC_PER_ROW) * 4;

        // Accumulators stay in registers for the whole K loop. This is the
        // register budget: 8 floats per lane per fragment, FRAG_M*FRAG_N of them.
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[FRAG_M][FRAG_N];
        for (int m = 0; m < FRAG_M; ++m)
            for (int n = 0; n < FRAG_N; ++n)
                wmma::fill_fragment(acc[m][n], 0.0f);

        for (int tile = 0; tile < K; tile += BK)
        {
            // A: BM x BK, one float4 per thread per pass.
            for (int r = 0; r < BM; r += A_ROW_STRIDE)
            {
                const float4 v = reinterpret_cast<const float4 *>(
                    &A[(block_row + r + a_row) * K + tile + a_col])[0];
                reinterpret_cast<float4 *>(&As[(r + a_row) * LDA + a_col])[0] = v;
            }

            // B: BK x BN, same shape of loop.
            for (int r = 0; r < BK; r += B_ROW_STRIDE)
            {
                const float4 v = reinterpret_cast<const float4 *>(
                    &B[(tile + r + b_row) * N + block_col + b_col])[0];
                reinterpret_cast<float4 *>(&Bs[(r + b_row) * LDB + b_col])[0] = v;
            }

            __syncthreads();

            for (int kk = 0; kk < BK; kk += WMMA_K)
            {
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> a[FRAG_M];
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> b[FRAG_N];

                // ldm is the SHARED tile's row stride, not K or N. Every lane of
                // the warp passes the same address -- these are warp-collective.
                for (int m = 0; m < FRAG_M; ++m)
                {
                    wmma::load_matrix_sync(a[m], &As[(warp_m * WM + m * WMMA_M) * LDA + kk], LDA);
                    for (int i = 0; i < a[m].num_elements; ++i)
                        a[m].x[i] = wmma::__float_to_tf32(a[m].x[i]);
                }
                for (int n = 0; n < FRAG_N; ++n)
                {
                    wmma::load_matrix_sync(b[n], &Bs[kk * LDB + warp_n * WN + n * WMMA_N], LDB);
                    for (int i = 0; i < b[n].num_elements; ++i)
                        b[n].x[i] = wmma::__float_to_tf32(b[n].x[i]);
                }

                // FRAG_M*FRAG_N mma calls against FRAG_M+FRAG_N fragment loads:
                // that ratio is the reuse the warp tile buys.
                for (int m = 0; m < FRAG_M; ++m)
                    for (int n = 0; n < FRAG_N; ++n)
                        wmma::mma_sync(acc[m][n], a[m], b[n], acc[m][n]);
            }

            __syncthreads();
        }

        for (int m = 0; m < FRAG_M; ++m)
            for (int n = 0; n < FRAG_N; ++n)
                wmma::store_matrix_sync(
                    &C[(block_row + warp_m * WM + m * WMMA_M) * N + block_col + warp_n * WN + n * WMMA_N],
                    acc[m][n], N, wmma::mem_row_major);
    }

    // Every variant launches through here, so the asserts cannot drift per entry.
    template <int BM, int BN, int BK, int WM, int WN, int PAD = 0>
    inline void launch_tiled(const float *dA, const float *dB, float *dC, int M, int K, int N)
    {
        assert(M % BM == 0);
        assert(N % BN == 0);
        assert(K % BK == 0);

        constexpr int NUM_THREADS = (BM / WM) * (BN / WN) * 32;
        dim3 blocksize{NUM_THREADS};
        dim3 gridsize{(unsigned)(N / BN), (unsigned)(M / BM)};
        wmma_tiled_kernel<BM, BN, BK, WM, WN, PAD><<<gridsize, blocksize>>>(dA, dB, dC, M, K, N);
    }

} // namespace

// ---- the sweep. Each entry is one point in (BM, BN, BK, WM, WN) space. ----
// acc regs/lane = 8*(WM/16)*(WN/16); shared = (BM*(BK+PAD) + BK*(BN+PAD))*4.

// 4 warps, 32 acc regs, 8 KB shared -- the conservative starting point.
void launch_wmma_64x64(const float *dA, const float *dB, float *dC, int M, int K, int N)
{ launch_tiled<64, 64, 16, 32, 32>(dA, dB, dC, M, K, N); }

// 4 warps, 64 acc regs, 12 KB: taller warp tile, more reuse per a-fragment.
void launch_wmma_128x64(const float *dA, const float *dB, float *dC, int M, int K, int N)
{ launch_tiled<128, 64, 16, 64, 32>(dA, dB, dC, M, K, N); }

// 8 warps, 64 acc regs, 16 KB.
void launch_wmma_128x128(const float *dA, const float *dB, float *dC, int M, int K, int N)
{ launch_tiled<128, 128, 16, 64, 32>(dA, dB, dC, M, K, N); }

// Same block tile, warp tile transposed: isolates warp-shape from block-shape.
void launch_wmma_128x128_w32x64(const float *dA, const float *dB, float *dC, int M, int K, int N)
{ launch_tiled<128, 128, 16, 32, 64>(dA, dB, dC, M, K, N); }

// Deeper K step: fewer __syncthreads per unit work, 32 KB shared.
void launch_wmma_128x128_bk32(const float *dA, const float *dB, float *dC, int M, int K, int N)
{ launch_tiled<128, 128, 32, 64, 32>(dA, dB, dC, M, K, N); }

// The PAD knob, on the configuration that wins without it.
void launch_wmma_128x128_pad4(const float *dA, const float *dB, float *dC, int M, int K, int N)
{ launch_tiled<128, 128, 16, 64, 32, 4>(dA, dB, dC, M, K, N); }
