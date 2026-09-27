#pragma once

// A kernel, seen from the outside: device pointers in, C = A*B out, row-major.
// The launch config is the kernel's own business and stays in its .cu.
// Asynchronous -- the caller decides when to synchronize.
using GemmFn = void (*)(const float* dA, const float* dB, float* dC,
                        int M, int K, int N);

// What the kernel's inputs are rounded to before it runs. The kernel still
// receives float* -- this says what precision its arithmetic actually uses, so
// the harness can hand it inputs that are exactly representable there.
enum class Prec { FP32, TF32, FP16, BF16 };

inline const char* prec_name(Prec p)
{
    switch (p) {
        case Prec::TF32: return "tf32";
        case Prec::FP16: return "fp16";
        case Prec::BF16: return "bf16";
        default:         return "fp32";
    }
}

struct Kernel {
    const char* name;
    GemmFn      launch;

    // Both default, so existing registry entries need no change.
    Prec   prec      = Prec::FP32;  // A and B are pre-rounded to this
    double tol_scale = 1.0;         // multiplies the shape's tolerance
};

// The ladder from docs/PLAN.md, in order. Defined in src/registry.cu.
extern const Kernel kernels[];
extern const int    num_kernels;

// Host reference implementation.
void gemm_cpu(const float* A, const float* B, float* C, int M, int K, int N);
