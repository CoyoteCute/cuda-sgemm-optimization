#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

__global__ void naive_gemm_kernel(const float* __restrict__  A, const  float* __restrict__ B, float* __restrict__ C, int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row<M and col<N)
    {
        float acc = 0.0f;
        for (int k=0; k<K; k++)
        {
            acc += A[row*K+k]*B[k*N+col];
        }
        C[row*N+col] = acc;
    }
};

void gemm_cpu(const float*  A, const  float*  B, float*  C, int M, int K, int N)
{   
    
    for (int i=0; i<M; i++)
    {
        for (int j=0; j<N; j++)
        {
            C[i*N+j] = 0;
            for (int k=0; k<K; k++)
            {
                C[i*N+j] += A[i*K+k]*B[k*N+j];
    
            }
        }
    }
    
};

void fill_mat(float * A, int M, int K, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (int i=0; i<M; i++)
    {
        for (int j=0; j<K; j++)
        {
            A[i*K+j] = dist(gen);
        }
    }
};

template<typename T>
double compare_mat(T A, T B, int M, int N, int K, double tol)
{
    double rel = 0;
    int bi = -1, bj = -1;
    for (int i=0; i<M; i++)
    {
        for (int j=0; j<N; j++)
        {
            double new_rel = fabs( A[i*N+j] - B[i*N+j]) / (fabs(B[i*N+j])+1e-2 * sqrt((double)K) );
            if (rel < new_rel) { rel = new_rel; bi = i; bj = j; }
        }
    }
    printf("worst at (%d,%d): gpu=%.6f ref=%.6f\n", bi, bj, A[bi*N+bj], B[bi*N+bj]);
    return rel;
};

int main()
{
    int M = 512;
    int K = 1024;
    int N = 1024;
    unsigned int BM = 32;
    unsigned int BN = 32;
    double tol = 1e-6;                                  // ~10 ULP relative
    
    float* A = new float[M*K];
    float* B = new float[K*N];
    float* C = new float[M*N];
    float* C_cpu =  new float[M*N];

    fill_mat(A, M, K, 42);
    fill_mat(B, K, N, 1337);
    gemm_cpu(A, B, C_cpu, M, K, N);

    float* dA;
    cudaMalloc((void**)&dA, M*K*sizeof(float));
    float* dB;
    cudaMalloc((void**)&dB, N*K*sizeof(float));
    float* dC;
    cudaMalloc((void**)&dC, M*N*sizeof(float));
    cudaMemcpy(dA, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N*K*sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blocksize{BM, BN};
    dim3 gridsize{(N+BN-1)/BN, (M+BM-1)/BM};
    naive_gemm_kernel<<<gridsize, blocksize>>>(dA, dB, dC, M,K,N);
    cudaMemcpy(C, dC, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    
    double max_rel = compare_mat(C, C_cpu, M, N, K, tol);
    printf("max rel err: %.3e  [%s]\n", max_rel, max_rel < tol ? "pass" : "fail");

}