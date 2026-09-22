#include "gemm.h"
#include "kernels/kernels.h"

const Kernel kernels[] = {
    {"naive", launch_naive},
    {"smem", launch_smem},
    {"registerT", launch_registerT},
    {"rt_V_Bs", launch_rt_V_Bs},
    {"rt_V_BsAs", launch_rt_V_BsAs}
};

const int num_kernels = sizeof(kernels) / sizeof(kernels[0]);
