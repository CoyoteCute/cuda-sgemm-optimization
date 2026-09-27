#pragma once

// One declaration per kernel. Each kernel's .cu includes this header too, so a
// launcher whose signature drifts from GemmFn fails to compile rather than
// failing to link.

void launch_naive(const float* dA, const float* dB, float* dC, int M, int K, int N);
void launch_smem(const float* dA, const float* dB, float* dC, int M, int K, int N);
void launch_registerT(const float* dA, const float* dB, float* dC, int M, int K, int N);
void launch_rt(const float* dA, const float* dB, float* dC, int M, int K, int N);
void launch_rt_V_AsBs(const float* dA, const float* dB, float* dC, int M, int K, int N);
void launch_wt(const float* dA, const float* dB, float* dC, int M, int K, int N);
void launch_rt_async(const float* dA, const float* dB, float* dC, int M, int K, int N);

// First tensor-core kernel: inputs are pre-rounded to tf32 by the harness.
void launch_wmma_tf32(const float* dA, const float* dB, float* dC, int M, int K, int N);

// Templated tensor-core sweep, all from wmma_tiled.cu:
// (BM, BN, BK, WM, WN, PAD)
void launch_wmma_64x64(const float* dA, const float* dB, float* dC, int M, int K, int N);
void launch_wmma_128x64(const float* dA, const float* dB, float* dC, int M, int K, int N);
void launch_wmma_128x128(const float* dA, const float* dB, float* dC, int M, int K, int N);
void launch_wmma_128x128_w32x64(const float* dA, const float* dB, float* dC, int M, int K, int N);
void launch_wmma_128x128_bk32(const float* dA, const float* dB, float* dC, int M, int K, int N);
void launch_wmma_128x128_pad4(const float* dA, const float* dB, float* dC, int M, int K, int N);

// Reference ceiling, not part of the ladder.
void launch_cublas(const float* dA, const float* dB, float* dC, int M, int K, int N);
