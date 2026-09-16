#pragma once

// One declaration per kernel. Each kernel's .cu includes this header too, so a
// launcher whose signature drifts from GemmFn fails to compile rather than
// failing to link.

void launch_naive(const float* dA, const float* dB, float* dC, int M, int K, int N);
void launch_smem(const float* dA, const float* dB, float* dC, int M, int K, int N);
