#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

#include "gemm.h"
#include "kernels/kernels.h"

// cuBLAS is not a rung on the ladder, it is the ceiling: the useful number is
// what fraction of it our own kernels reach on the same shape and the same data.
//
// No timing or synchronization in here. cublasSgemm is asynchronous on the
// default stream, exactly like a <<<>>> launch, so time_kernel in
// src/harness.cuh brackets it with events the same way it does every kernel.

#define CUBLAS_CHECK(expr)                                                      \
    do                                                                          \
    {                                                                           \
        cublasStatus_t status_ = (expr);                                         \
        if (status_ != CUBLAS_STATUS_SUCCESS)                                    \
        {                                                                        \
            fprintf(stderr, "cuBLAS error at %s:%d\n %s\n %s: %s\n",            \
                    __FILE__, __LINE__, #expr,                                   \
                    cublasGetStatusName(status_), cublasGetStatusString(status_));\
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

namespace
{

    // One handle for the process, built on first use. Creating it costs
    // milliseconds and allocates workspace, which would wreck the first timed
    // iteration; the harness runs 5 warmup launches first, so that cost lands
    // there instead.
    cublasHandle_t handle()
    {
        struct Handle
        {
            cublasHandle_t h{};
            Handle()
            {
                CUBLAS_CHECK(cublasCreate(&h));
                // PEDANTIC_MATH keeps SGEMM in true FP32. Left at the default,
                // cuBLAS may drop to TF32 on the tensor cores: faster, but not
                // the arithmetic our kernels do, so the comparison would be
                // measuring two different computations.
                CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_PEDANTIC_MATH));
            }
            ~Handle() { cublasDestroy(h); }
        };
        static Handle instance;
        return instance.h;
    }

} // namespace

void launch_cublas(const float *dA, const float *dB, float *dC, int M, int K, int N)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS is column-major, our matrices are row-major. Rather than transpose
    // anything, use the fact that a row-major MxN matrix is bit-identical to a
    // column-major NxM one: ask cuBLAS for C^T = B^T * A^T. That means swapping
    // the operands, swapping m and n, and passing each matrix's row length as its
    // leading dimension. Both ops stay _N -- nothing is actually transposed.
    CUBLAS_CHECK(cublasSgemm(handle(),
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             N,
                             M,
                             K,
                             &alpha,
                             dB,
                             N,
                             dA,
                             K,
                             &beta,
                             dC,
                             N));
}
