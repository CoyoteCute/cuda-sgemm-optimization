#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

#include "cuda_check.cuh"

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
