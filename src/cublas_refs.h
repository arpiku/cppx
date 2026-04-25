#include "bencher.h"

Bencher::BenchmarkResult get_bf16_cublas_ref();


inline void gemm_bf16_cublas_tensor(cublasHandle_t handle,
                                    int m,
                                    int n,
                                    int k,
                                    const __nv_bfloat16* a,
                                    const __nv_bfloat16* b,
                                    float alpha = 1.0f,
                                    float beta = 0.0f,
                                    float* c,
                                    cudaStream_t stream) {
    CUDA_CHECK(cudaGetLastError());
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              m,
                              n,
                              k,
                              &alpha,
                              a,
                              CUDA_R_16BF,
                              m,
                              b,
                              CUDA_R_16BF,
                              k,
                              &beta,
                              c,
                              CUDA_R_32F,
                              m,
                              CUBLAS_COMPUTE_32F_FAST_16BF,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
