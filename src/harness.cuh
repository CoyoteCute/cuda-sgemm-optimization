#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstring>

#include "cuda_check.cuh"
#include "gemm.h"

// Times a launch: warmups discarded, then median of `iters` timed runs.
// Returns milliseconds. Templated on the callable so the lambda inlines and
// the timed loop holds nothing but the launch.
template <typename F>
float time_kernel(F launch, int warmup = 5, int iters = 20)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmups already pay for a sync, so check launch + async errors here and
    // keep cudaDeviceSynchronize out of the timed loop.
    for (int i = 0; i < warmup; ++i) launch();
    CUDA_CHECK_KERNEL();

    std::vector<float> times;
    times.reserve(iters);

    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

inline void fill_mat(float* A, int M, int K, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (int i=0; i<M; i++)
    {
        for (int j=0; j<K; j++)
        {
            A[i*K+j] = dist(gen);
        }
    }
}

// Rounds src into dst at `p`'s precision, on the host.
//
// This is the whole trick behind comparing a reduced-precision kernel against an
// FP32 reference: hand the kernel inputs it can already represent exactly, and
// hand the reference the identical values. The input-rounding error then cancels
// out of the comparison instead of showing up as a difference, so the tolerance
// only has to cover accumulation order -- and a structurally broken kernel still
// stands out by orders of magnitude.
//
// FP16 and BF16 go through the intrinsics, which get the narrower exponent range
// and subnormals right. TF32 has no host intrinsic, so it is done by hand:
// round-to-nearest-even on the 13 mantissa bits it drops, keeping 10 of 23.
// Verified against __float2half: for values inside FP16's exponent range the two
// agree exactly, as they must, both keeping 10 bits.
inline void round_to_prec(float* dst, const float* src, size_t n, Prec p)
{
    switch (p) {
    case Prec::FP32:
        if (dst != src) memcpy(dst, src, n * sizeof(float));
        break;
    case Prec::TF32:
        for (size_t i = 0; i < n; ++i) {
            uint32_t u;
            memcpy(&u, &src[i], 4);
            u += 0x1000u + ((u >> 13) & 1u);   // round to nearest even
            u &= 0xFFFFE000u;                  // drop the low 13 mantissa bits
            memcpy(&dst[i], &u, 4);
        }
        break;
    case Prec::FP16:
        for (size_t i = 0; i < n; ++i) dst[i] = __half2float(__float2half(src[i]));
        break;
    case Prec::BF16:
        for (size_t i = 0; i < n; ++i) dst[i] = __bfloat162float(__float2bfloat16(src[i]));
        break;
    }
}

template<typename T>
double compare_mat(T A, T B, int M, int N, int K, double tol)
{
    double rel = 0;
    int bi = -1, bj = -1;
    for (int i=0; i<M; i++)
    {
        for (int j=0; j<N; j++)
        {
            double new_rel = fabs( A[i*N+j] - B[i*N+j]) / (fabs(B[i*N+j])+1e-2 * sqrt((double)K) );
            if (rel < new_rel) { rel = new_rel; bi = i; bj = j; }
        }
    }
    printf("  worst at (%d,%d): gpu=%.6f ref=%.6f\n", bi, bj, A[bi*N+bj], B[bi*N+bj]);
    return rel;
}
