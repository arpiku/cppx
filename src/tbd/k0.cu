
#include <cstdio>
#include <cassert>
#include <cstdint>
#include <random>

#include <cuda_runtime.h>
#include <mma.h>

#include "funcs.h"

#define MMA_M 16
#define MMA_N 8
#define MMA_K 8

#define CUDA_CHECK(call)                                                           \
do {                                                                               \
        cudaError_t error = call;                                                  \
        if (error != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error),\
                    __FILE__, __LINE__);                                           \
            exit(error);                                                           \
        }                                                                          \
} while (0)
#undef NDEBUG


void launch_kernel();


using TensorFloat32 = float;

__global__ void k_mmaTensorMatMulM16N8k8TF32(TensorFloat32* rowMajorA_d,
                                             TensorFloat32* rowMajorB_d,
                                             float* resultMatrixC_d,
                                             int mDim, int nDim,
                                             int kDim);

// CPU Function to compute valid reference result
// Inputs => ( Row Major -> A, Row Major -> B)
__host__ void cpuMatMulReference(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float aVal = static_cast<float>(A[i*K + k]);
                float bVal = static_cast<float>(B[k*N + j]);
                sum += aVal * bVal;
            }
            C[i*N + j] = sum;
        }
    }
}

void launch_kernel() {
    const int TEST_CASE_COUNT = 7;
    const int PROBLEM_DIMS = 3;

    //Test case dimensions {M, N, K}
    const int MAX_M = 512;
    const int MAX_N = 512;
    const int MAX_K = 512;

    // Test dimensions (all must be multiples of MMA tile sizes)
    const int TEST_CASES_DIMS[TEST_CASE_COUNT][PROBLEM_DIMS] = {{16,8,8}, {512,512,512}, {32,16,32}, {256, 256, 256}, {64, 64, 64} , {64, 32, 32}, {128, 128, 128}};

    int deviceId = 0;
    cudaDeviceProp prop;
    cudaError_t status = cudaGetDeviceProperties(&prop, deviceId);
    const int WARP_SIZE = prop.warpSize;


    // Kernel configuration parameters
    const int WARPS_PER_BLOCK = 4;
    const int BLOCK_SIZE = WARP_SIZE * WARPS_PER_BLOCK;

    //Tolerance for validation, set to 1% due to nature of half precision operations
    const float TOLERANCE  = 0.01;

    //Set up random number generation
    std::mt19937 randEngine(time(nullptr));
    // Bounded random distribution for test case initialization
    std::uniform_real_distribution<float> randDist(1.0f, 100.0f);

    // Use a CUDA stream for asynchronous operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    //Pointers for Host Memory
    TensorFloat32* A_h =(TensorFloat32*)malloc(MAX_M * MAX_K * sizeof(TensorFloat32));
    TensorFloat32* B_h =(TensorFloat32*)malloc(MAX_K * MAX_N * sizeof(TensorFloat32));

    TensorFloat32* cpuC_h =(TensorFloat32*)malloc(MAX_M * MAX_N * sizeof(TensorFloat32)); // Reference Matrix space allocation on host
    TensorFloat32* gpuC_h = (TensorFloat32*)malloc(MAX_M * MAX_N * sizeof(TensorFloat32));// GPU result Matrix space allocation on host

    // Allocate the memory on the device
    TensorFloat32 *A_d;
    TensorFloat32 *B_d;
    float *C_d;

    CUDA_CHECK(cudaMallocAsync(&A_d, MAX_M * MAX_K * sizeof(TensorFloat32), stream));
    CUDA_CHECK(cudaMallocAsync(&B_d, MAX_K * MAX_N * sizeof(TensorFloat32), stream));
    CUDA_CHECK(cudaMallocAsync(&C_d, MAX_M * MAX_N * sizeof(float), stream));

    for (int t = 0; t < TEST_CASE_COUNT; ++t) {
        int M = TEST_CASES_DIMS[t][0];
        int N = TEST_CASES_DIMS[t][1];
        int K = TEST_CASES_DIMS[t][2];

        // Fill with random values
        for (int r = 0; r < M; ++r) {
            for (int c = 0; c < K ; ++c) {
                TensorFloat32 val = randDist(randEngine);
                A_h[r * K + c] = val;
            } // Filling A Matrix in Row Major Way
       }

       for (int r = 0; r < K; ++r) {
            for (int c = 0; c < N; ++c) {
                TensorFloat32 val = randDist(randEngine);
                B_h[r * N + c] = val;
            } // Filling B Matrix in Row Major Way
        }

        // Device memory
        CUDA_CHECK(cudaMemcpyAsync(A_d, A_h, M*K*sizeof(TensorFloat32), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(B_d, B_h, K*N*sizeof(TensorFloat32), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemsetAsync(C_d, 0, M*N*sizeof(float), stream));

        // Launch config
        int numTilesM     = (M + MMA_M - 1) / MMA_M;
        int numTilesN     = (N + MMA_N - 1) / MMA_N;
        dim3 gridDim(numTilesN, (numTilesM + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
        dim3 blockDim(BLOCK_SIZE);

        size_t shmemBytes = WARPS_PER_BLOCK * (MMA_M*MMA_K + MMA_K*MMA_N) * sizeof(TensorFloat32);

        // Kernel launch
        void *args[] = {&A_d,
                        &B_d,
                        &C_d,
                        &M,
                        &N,
                        &K};

        CUDA_CHECK(cudaLaunchKernel((void*)k_mmaTensorMatMulM16N8k8TF32,
                                            gridDim,
                                            blockDim,
                                            args,
                                            shmemBytes,
                                            stream
                                    ));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Get result
        CUDA_CHECK(cudaMemcpyAsync(gpuC_h, C_d, M*N*sizeof(float), cudaMemcpyDeviceToHost));

        // CPU reference
        cpuMatMulReference(A_h, B_h, cpuC_h, M, N, K);

        //Validate the result, with in 1% tolerance
        for(int t = 0; t < M*N; ++t) {
            assert(std::fabs((gpuC_h[t] - cpuC_h[t]) / std::fabs(cpuC_h[t])) <= TOLERANCE);
        }
   }
   CUDA_CHECK(cudaFreeAsync(A_d, stream));
   CUDA_CHECK(cudaFreeAsync(B_d, stream));
   CUDA_CHECK(cudaFreeAsync(C_d, stream));
   CUDA_CHECK(cudaStreamDestroy(stream));

   free(A_h);
   free(B_h);
   free(cpuC_h);
   free(gpuC_h);
}

__global__ void k_mmaTensorMatMulM16N8k8TF32(TensorFloat32* rowMajorA_d,
                                             TensorFloat32* rowMajorB_d,
                                             float* resultMatrixC_d,
                                             int mDim,
                                             int nDim,
                                             int kDim) {
    /*
    @solution
    */
    /*
    @cuda_toolkit_begin 12.2
    */

    const int ACC_FRAG_SIZE = 4;
    const int A_FRAG_SIZE = 4;
    const int B_FRAG_SIZE = 2;

    int warpsPerBlock = blockDim.x / warpSize;
    int warpId        = threadIdx.x / warpSize;   // which warp in the block
    int lane          = threadIdx.x % warpSize;   // lane within the warp

    // Guards
    if ( (blockIdx.y * warpsPerBlock + warpId)*MMA_M >= mDim )
        return;
    if ( blockIdx.x * MMA_N >= nDim )
        return;

    // Carve up shared memory per-warp
    extern __shared__ float shmem[];
    size_t aTileSize = MMA_M * MMA_K;
    size_t bTileSize = MMA_K * MMA_N;

    // Current tile offsets
    float* aTile = shmem + warpId * (aTileSize + bTileSize);
    float* bTile = aTile  + aTileSize;

    int numTilesK = kDim / MMA_K;
    int rowOffSetA = MMA_M/2; // 8
    int colOffSetA = MMA_K/2; // 4
    int rowOffsetB = 4;
    int rowOffsetC = 8;
    int lanesPerSubRow = 4;


    float accReg[ACC_FRAG_SIZE] = {0.f, 0.f, 0.f, 0.f};
    for (int tileK = 0; tileK < numTilesK; ++tileK) {
        // Pointer to 16x8 A tile for loading
        TensorFloat32* aG = rowMajorA_d + (blockIdx.y * warpsPerBlock + warpId)*MMA_M * kDim
                                              + tileK * MMA_K;

        // Pointer to 8x8 B tile for loading
        TensorFloat32* bG = rowMajorB_d + tileK * MMA_K * nDim
                                              + blockIdx.x * MMA_N;

        // Warp-strided load of a tile into shared memory:
        // Each lane loads it's own data
       for (int i = lane; i < aTileSize; i += warpSize) {
            int r = i / MMA_K;
            int c = i % MMA_K;
            aTile[i] = aG[r * kDim + c];
        }
        for (int i = lane; i < bTileSize; i += warpSize) {
            int r = i / MMA_N;
            int c = i % MMA_N;
            bTile[i] = bG[r * nDim + c];
        }
        __syncthreads();

        // Each lane loads its fragment from shared memory
        float fragA[A_FRAG_SIZE], fragB[B_FRAG_SIZE];

        // Gathering TF32 fragment for A
        {
            int row0 = lane / lanesPerSubRow;
            int col0 = lane % lanesPerSubRow;
            int row1 = row0 + rowOffSetA;

            fragA[0] = aTile[row0 * MMA_K + col0];
            fragA[1] = aTile[row1 * MMA_K + col0];
            fragA[2] = aTile[row0 * MMA_K + col0 + colOffSetA];
            fragA[3] = aTile[row1 * MMA_K + col0 + colOffSetA];
        }

        // Gathering TF32 fragment for B
        {
            int col = lane / lanesPerSubRow;
            int row = lane % lanesPerSubRow;

            fragB[0] = bTile[row * MMA_N + col];
            fragB[1] = bTile[(row + rowOffsetB) * MMA_N + col];
        }


        // MMA call
        // bit-cast the 32-bit TF32 floats into integer registers
        uint32_t aBits0 = __float_as_uint(fragA[0]);
        uint32_t aBits1 = __float_as_uint(fragA[1]);
        uint32_t aBits2 = __float_as_uint(fragA[2]);
        uint32_t aBits3 = __float_as_uint(fragA[3]);

        uint32_t bBits0 = __float_as_uint(fragB[0]);
        uint32_t bBits1 = __float_as_uint(fragB[1]);

        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0, %1, %2, %3}, "    /* D -> accReg[%0..3] (written) and C -> accReg[%0..3] (read) */
            "{%4, %5, %6, %7}, "    /* A -> aBits0..3 (r) */
            "{%8, %9}, "            /* B -> bBits0..1 (r) */
            "{%0, %1, %2, %3};"     /* C -> reuse accReg[%0..3] */
            : "+f"(accReg[0]), "+f"(accReg[1]), "+f"(accReg[2]), "+f"(accReg[3])
            : "r"(aBits0), "r"(aBits1), "r"(aBits2), "r"(aBits3),
              "r"(bBits0), "r"(bBits1)
        );

        __syncthreads();
    }

    // Write back to global memory (row major)
    float* cTile = resultMatrixC_d + ((blockIdx.y * warpsPerBlock + warpId) * MMA_M) * nDim + blockIdx.x * MMA_N;

    int rowOut = lane / lanesPerSubRow;
    int colOut = (lane % lanesPerSubRow) * 2;
    cTile[rowOut * nDim + colOut] = accReg[0];
    cTile[rowOut * nDim + colOut + 1] = accReg[1];
    cTile[(rowOut + rowOffsetC) * nDim + colOut] = accReg[2];
    cTile[(rowOut + rowOffsetC) * nDim + colOut + 1] = accReg[3];
}
