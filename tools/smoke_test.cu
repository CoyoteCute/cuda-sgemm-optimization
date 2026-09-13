// Verifies toolchain + profiler counter access.
//   nvcc -arch=sm_86 tools/smoke_test.cu -o build/smoke && ncu --set basic build/smoke
#include <cstdio>
__global__ void k(float* p) { p[threadIdx.x] = threadIdx.x; }
int main() {
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("%s  sm_%d%d  %d SMs  %.0f GB/s\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    float* p; cudaMalloc(&p, 1024);
    k<<<1, 32>>>(p);
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : 1;
}
