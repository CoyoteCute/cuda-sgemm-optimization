// Every registered GPU kernel against the CPU reference, on two shapes:
// one tile-aligned shape carried through the sweep, and one ragged shape whose
// only job is to catch kernels that mishandle partial tiles.
//   build/test_gemm                 all kernels, both cases
//   build/test_gemm naive           only kernels whose name contains "naive"
//   build/test_gemm -c aligned      only the case named exactly "aligned"
//   build/test_gemm -c aligned smem both filters at once
//   build/test_gemm -n ...          no check: skip the CPU reference, still time
//   build/test_gemm -p ...          profile mode: one launch per kernel, no CPU
//                                   reference, no check, no timing -- for ncu
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "gemm.h"
#include "harness.cuh"

namespace {

struct Case {
    const char* name;
    int  M, K, N;
    bool timed;      // false: correctness only, the shape is too small to mean anything
};

const Case cases[] = {
    // Every dim a multiple of 32, so no kernel in the ladder ever runs a partial
    // tile here. This is the shape the GFLOP/s numbers in docs/PLAN.md refer to;
    // don't change it without restating the baseline.
    //{"aligned", 4096, 4096, 4096, true},
    {"aligned", 512, 1024, 1024, true},

    // Deliberately awkward: M and N are 32k+1, so the last tile in each axis has
    // exactly one valid row and one valid column, and K is not a multiple of
    // anything. A kernel whose bounds guard is wrong fails here and nowhere else.
    {"ragged",  513,   1025,  1025, false},
};
const int num_cases = sizeof(cases) / sizeof(cases[0]);

// FP32 accumulation over K=1024 lands near 2e-5 against the CPU reference, which
// also sums in float but in a different order; see the acceptance note in docs/PLAN.md. This catches a broken
// kernel, not a differently-rounded one.
constexpr double kTol = 1e-4;

constexpr double kPeakGflops = 16200.0;   // RTX 3060 Ti, FP32

// Runs every kernel matching `filter` on one shape. Adds the number of kernels
// run to *ran, returns how many of them failed. With `check` off, the CPU
// reference and the comparison are skipped and every kernel counts as passed.
// With `profile` set, each kernel is launched exactly once and nothing else
// runs: under ncu the CPU reference is pure waiting, and the timing launches
// only get skipped by -c 1 anyway.
int run_case(const Case& cs, const char* filter, bool check, bool profile, int* ran)
{
    const int M = cs.M, K = cs.K, N = cs.N;

    printf("\n=== %s   M=%d K=%d N=%d ===\n", cs.name, M, K, N);

    float* A     = new float[M*K];
    float* B     = new float[K*N];
    float* C     = new float[M*N];
    float* C_cpu = new float[M*N];

    fill_mat(A, M, K, 42);
    fill_mat(B, K, N, 1337);
    if (check) gemm_cpu(A, B, C_cpu, M, K, N);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc((void**)&dA, M*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dB, K*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&dC, M*N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, A, M*K*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, K*N*sizeof(float), cudaMemcpyHostToDevice));

    int failures = 0;

    for (int i = 0; i < num_kernels; ++i) {
        const Kernel& k = kernels[i];
        if (filter && !strstr(k.name, filter)) continue;
        ++*ran;

        printf("[%s]\n", k.name);

        // Zeroed first, so an element the kernel never writes reads back as 0
        // and shows up as an error instead of as leftover state from last run.
        CUDA_CHECK(cudaMemset(dC, 0, M*N*sizeof(float)));
        k.launch(dA, dB, dC, M, K, N);
        CUDA_CHECK_KERNEL();
        if (profile) { printf("  launched once (profile mode, not checked)\n"); continue; }

        if (check) {
            CUDA_CHECK(cudaMemcpy(C, dC, M*N*sizeof(float), cudaMemcpyDeviceToHost));
            double max_rel = compare_mat(C, C_cpu, M, N, K, kTol);
            const bool pass = max_rel < kTol;
            printf("  max rel err: %.3e  [%s]\n", max_rel, pass ? "pass" : "FAIL");
            if (!pass) { ++failures; continue; }
        } else {
            printf("  not checked (-n)\n");
        }

        // A wrong kernel is not worth timing, and neither is the ragged shape.
        if (!cs.timed) continue;

        float ms = time_kernel([&]{ k.launch(dA, dB, dC, M, K, N); });
        double gflops = 2.0 * M * N * K / (ms * 1e6);
        printf("  %8.3f ms   %8.2f GFLOP/s   %5.2f%% of peak\n",
               ms, gflops, 100.0 * gflops / kPeakGflops);
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    delete[] A; delete[] B; delete[] C; delete[] C_cpu;

    return failures;
}

}  // namespace

int main(int argc, char** argv)
{
    const char* filter    = nullptr;   // kernel name, substring
    const char* only_case = nullptr;   // case name, exact: "a" would match both
    bool        check     = true;
    bool        profile   = false;
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "-c") && i + 1 < argc) only_case = argv[++i];
        else if (!strcmp(argv[i], "-n"))                  check     = false;
        else if (!strcmp(argv[i], "-p"))                  profile   = true;
        else                                               filter    = argv[i];
    }
    if (profile) check = false;

    if (only_case) {
        bool known = false;
        for (int i = 0; i < num_cases; ++i) known |= !strcmp(cases[i].name, only_case);
        if (!known) {
            fprintf(stderr, "no case named \"%s\"; have:", only_case);
            for (int i = 0; i < num_cases; ++i) fprintf(stderr, " %s", cases[i].name);
            fprintf(stderr, "\n");
            return 2;
        }
    }

    if      (profile) printf("profile mode: no reference, no check, no timing");
    else if (!check)  printf("no check: no reference, timing only");
    else              printf("reference: CPU   tol %.0e", kTol);
    if (filter)    printf("   filter \"%s\"", filter);
    if (only_case) printf("   case \"%s\"", only_case);
    printf("\n");

    int ran = 0, failures = 0;
    for (int i = 0; i < num_cases; ++i) {
        if (only_case && strcmp(cases[i].name, only_case)) continue;
        failures += run_case(cases[i], filter, check, profile, &ran);
    }

    if (ran == 0) {
        fprintf(stderr, "\nno kernel matched \"%s\"\n", filter);
        return 2;
    }
    if (!check) return 0;   // nothing was checked, so there is no pass count to report
    printf("\n%d/%d passed\n", ran - failures, ran);
    return failures == 0 ? 0 : 1;
}
