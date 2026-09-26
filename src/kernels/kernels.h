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

// Reference ceiling, not part of the ladder.
void launch_cublas(const float* dA, const float* dB, float* dC, int M, int K, int N);
