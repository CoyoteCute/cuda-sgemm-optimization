#include "gemm.h"
#include "kernels/kernels.h"

const Kernel kernels[] = {
    {"naive", launch_naive},
    {"smem", launch_smem},
    {"registerT", launch_registerT},
    {"rt", launch_rt},
    {"rt_V_AsBs", launch_rt_V_AsBs},
    {"rt_async", launch_rt_async},
    // tol_scale 10: measured, not guessed. A single mma_sync is 2.4e-6 from an
    // exact double reference, so the tensor core is not the problem -- but the
    // accumulator sums 8 terms in hardware and then 512 mma results in sequence,
    // an order gemm_cpu's strictly sequential sum cannot match, and that gap
    // grows with K: 6e-6 at K=64, 1.4e-4 at K=512, 3.9e-4 at K=4096 against
    // exact. Measured 9.1e-4 against our FP32 reference at 4096, so 10x the
    // shape tolerance (2e-3 there) leaves margin while staying three orders
    // below anything structurally broken.
    {"wmma_tf32", launch_wmma_tf32, Prec::TF32, 10.0},
    {"wmma_64x64",          launch_wmma_64x64,          Prec::TF32, 10.0},
    {"wmma_128x64",         launch_wmma_128x64,         Prec::TF32, 10.0},
    {"wmma_128x128",        launch_wmma_128x128,        Prec::TF32, 10.0},
    {"wmma_128x128_w32x64", launch_wmma_128x128_w32x64, Prec::TF32, 10.0},
    {"wmma_128x128_bk32",   launch_wmma_128x128_bk32,   Prec::TF32, 10.0},
    {"wmma_128x128_pad4",   launch_wmma_128x128_pad4,   Prec::TF32, 10.0},
    {"cublas", launch_cublas}
    //{"wt", launch_wt}
};

const int num_kernels = sizeof(kernels) / sizeof(kernels[0]);
