#include "gemm.h"
#include "kernels/kernels.h"

const Kernel kernels[] = {
    {"naive", launch_naive},
    {"smem", launch_smem},
    {"registerT", launch_registerT},
    {"rt", launch_rt},
    {"rt_V_AsBs", launch_rt_V_AsBs},
    {"rt_async", launch_rt_async}
    //{"wt", launch_wt}
};

const int num_kernels = sizeof(kernels) / sizeof(kernels[0]);
