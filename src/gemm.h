#pragma once

// A kernel, seen from the outside: device pointers in, C = A*B out, row-major.
// The launch config is the kernel's own business and stays in its .cu.
// Asynchronous -- the caller decides when to synchronize.
using GemmFn = void (*)(const float* dA, const float* dB, float* dC,
                        int M, int K, int N);

struct Kernel {
    const char* name;
    GemmFn      launch;
};

// The ladder from docs/PLAN.md, in order. Defined in src/registry.cu.
extern const Kernel kernels[];
extern const int    num_kernels;

// Host reference implementation.
void gemm_cpu(const float* A, const float* B, float* C, int M, int K, int N);
