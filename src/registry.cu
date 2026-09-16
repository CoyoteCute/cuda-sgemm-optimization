#include "gemm.h"
#include "kernels/kernels.h"

const Kernel kernels[] = {
    {"naive", launch_naive},
    {"smem", launch_smem}
};

const int num_kernels = sizeof(kernels) / sizeof(kernels[0]);
