#define TN 8
#define TM 8
#define BM 128
#define BK 16
#define BN 128

__global__ void coarse2D_mat_mul_kernel(float *d_A, float *d_B, float *d_C,
                                        int M, int N, int K)
{
    // Getting number of threads per block
    const int NUM_THREADS = BM * BN / (TN * TM);
    static_assert(NUM_THREADS % BK == 0);
    static_assert(NUM_THREADS % BN == 0);

    const int b_x = blockIdx.x;
    const int b_y = blockIdx.y;
    const int t_x = threadIdx.x;

    // 1D -> 2D
    const int A_view_ty = t_x / BK;
    const int A_view_tx = t_x % BK;
    const int B_view_ty = t_x / BN;
    const int B_view_tx = t_x % BN;
    // Adding strides to load A and B
    const int stride_A = NUM_THREADS / BK;
    const int stride_B = NUM_THREADS / BN;

    // Defining rows and cols for C[row, col] and tiles
    const int row = TM * (t_x / (BN / TN));
    const int col = TN * (t_x % (BN / TN));
    const int num_tiles = ceil((float)K / BK);

    // Saving in SMEM
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Mat-Mul Parallelize
    float acc[TM][TN] = {0.0f};
    float register_A[TN] = {0.0f};
    float register_B[TM] = {0.0f};

    for (int tile = 0; tile < num_tiles; tile++)
    {
        for (int load_offset = 0; load_offset < BM; load_offset += stride_A)
        {
            if (((b_y * BM + load_offset + A_view_ty) < M) && ((tile * BK + A_view_tx) < K))
            {
                As[load_offset + A_view_ty][A_view_tx] = d_A[(b_y * BM + load_offset + A_view_ty) * K + (tile * BK + A_view_tx)];
            }
            else
            {
                As[load_offset + A_view_ty][A_view_tx] = 0.0f;
            }
        }

        for (int load_offset = 0; load_offset < BK; load_offset += stride_B)
        {
            if (((tile * BK + load_offset + B_view_ty) < K) && (b_x * BN + B_view_tx < N))
            {
                Bs[load_offset + B_view_ty][B_view_tx] = d_B[(tile * BK + B_view_ty + load_offset) * N + (b_x * BN + B_view_tx)];
            }
            else
            {
                Bs[load_offset + B_view_ty][B_view_tx] = 0.0f;
            }
        }
        __syncthreads();

        // per-thread results
        for (int k = 0; k < BK; ++k)
        {
            // into registers
            for (int i = 0; i < TM; ++i)
            {
                register_A[i] = As[row + i][k];
            }
            for (int i = 0; i < TN; ++i)
            {
                register_B[i] = Bs[k][col + i];
            }

            for (int cy = 0; cy < TM; ++cy)
            {
                for (int cx = 0; cx < TN; ++cx)
                {
                    acc[cy][cx] += register_A[cy] * register_B[cx];
                }
            }
        }
        __syncthreads();
    }
    // assign calculated value
    for (int cy = 0; cy < TM; ++cy)
    {
        for (int cx = 0; cx < TN; cx++)
        {
            if ((b_y * BM + row + cy < M) && (b_x * BN + col + cx < N))
            {
                d_C[(b_y * BM + row + cy) * N + (b_x * BN + col + cx)] = 1 * acc[cy][cx] + 0 * d_C[(b_y * BM + row + cy) * N + (b_x * BN + col + cx)];
            }
        }
    }
}