// First tensor-core kernel. One warp computes one 16x16 tile of C.
//
// Deliberately the simplest thing that works: no shared memory at all.
// load_matrix_sync reads straight from global, so the smem staging the earlier
// kernels need is an optimization, not a requirement. What changes versus every
// kernel before this one is who owns what: no thread owns an element any more,
// the warp collectively owns a tile, and every lane must pass the SAME address
// to each wmma call.
//
// The TF32 fragment shape is m16n16k8 -- 16x16x16 does not exist for tf32, so
// the K loop steps by 8, not 16.
//
// No bounds handling: load_matrix_sync cannot mask, so a partial tile reads out
// of range. Fine for 4096, which divides by 16; the ragged case (513/1025) will
// read out of bounds, which is why this kernel is FP32-shaped only for now.
#include <cuda_runtime.h>
#include <cassert>
#include <mma.h>

#include "gemm.h"
#include "kernels/kernels.h"

using namespace nvcuda;

// WMMA's template parameters are int, so these are int rather than unsigned.
constexpr int BM = 16;
constexpr int BN = 16;
constexpr int BK = 8;

__global__ void wmma_tf32_kernel(const float *__restrict__ A, const float *__restrict__ B,
                                 float *__restrict__ C, int M, int K, int N)
{
    // The tile this warp owns, in elements of C. No threadIdx anywhere: all 32
    // lanes cooperate on the same tile and must agree on these.
    const int tile_row = blockIdx.y * BM;
    const int tile_col = blockIdx.x * BN;

    // matrix_a and matrix_b need the element type AND a layout. The accumulator
    // takes neither a layout nor tf32 -- it accumulates in real FP32, which is
    // why a tf32 kernel is far more accurate than its 10-bit inputs suggest.
    wmma::fragment<wmma::matrix_a, BM, BN, BK, wmma::precision::tf32, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, BM, BN, BK, wmma::precision::tf32, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, BM, BN, BK, float> c;

    wmma::fill_fragment(c, 0.0f);

    for (int k = 0; k < K; k += BK)
    {
        // ldm is the row length of the source matrix, not of the tile: K for A,
        // N for B. Getting this wrong reads a valid-looking but wrong rectangle.
        wmma::load_matrix_sync(a, A + tile_row * K + k, K);
        wmma::load_matrix_sync(b, B + k * N + tile_col, N);

        // load_matrix_sync brought in full-precision floats; the fragments must
        // be rounded to tf32 before mma_sync. This is per element, not per
        // fragment, and nvcc does not complain if you forget it.
        for (int i = 0; i < a.num_elements; i++) a.x[i] = wmma::__float_to_tf32(a.x[i]);
        for (int i = 0; i < b.num_elements; i++) b.x[i] = wmma::__float_to_tf32(b.x[i]);

        wmma::mma_sync(c, a, b, c); // accumulates in place
    }

    wmma::store_matrix_sync(C + tile_row * N + tile_col, c, N, wmma::mem_row_major);
}

void launch_wmma_tf32(const float *dA, const float *dB, float *dC, int M, int K, int N)
{
    // No partial-tile handling yet, so fail loudly instead of reading out of
    // bounds. This is what excludes the kernel from the ragged case for now.
    assert(M % BM == 0);
    assert(N % BN == 0);
    assert(K % BK == 0);

    dim3 blocksize{32}; // exactly one warp: the unit wmma works on
    dim3 gridsize{(unsigned)(N + BN - 1) / BN, (unsigned)(M + BM - 1) / BM};
    wmma_tf32_kernel<<<gridsize, blocksize>>>(dA, dB, dC, M, K, N);
}
