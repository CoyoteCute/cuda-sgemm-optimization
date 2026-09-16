#pragma once
#include<cuda_runtime.h>
#include<cstdio>
#include<cstdlib>

inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line)
{
    if (err != cudaSuccess){
        fprintf(stderr, "CUDA error at %s:%d\n %s\n %s: %s\n",
            file, line, expr, cudaGetErrorName(err), cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)
// After a kernel launch. Two distinct failure modes, two calls.
#define CUDA_CHECK_KERNEL()                       \
    do {                                          \
        CUDA_CHECK(cudaGetLastError());           \
        CUDA_CHECK(cudaDeviceSynchronize());      \
    } while (0)
