#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>


namespace Bencher {

struct GemmShape {
    uint m;
    uint n;
    uint k;
};

struct ErrorMetrics {
    double l2_abs;
    double l2_rel;
    double max_abs;
};

struct BenchmarkResult {
    GemmShape shape{};
    int warmup_iters = 0;
    int bench_iters = 0;

    float total_ms = 0.0f;
    float avg_ms = 0.0f;

    double tflops = 0.0;
    double gflops = 0.0;
    double perf_pct = 0.0;

    ErrorMetrics error{};
};


struct TestCase {
    GemmShape shape{};
    std::vector<float> a_f32;
    std::vector<float> b_f32;
    std::vector<float> c_f32;
    float alpha = 1.0f;
    float beta = 0.0f;
};


inline void cuda_check_impl(cudaError_t status,
                            const char* expr,
                            const char* file,
                            int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr,
                     "CUDA_CHECK failed: %s\n  status: %s\n  at %s:%d\n",
                     expr,
                     cudaGetErrorString(status),
                     file,
                     line);
        std::fflush(stderr);
        std::abort();
    }
}
inline void cublas_check_impl(cublasStatus_t status,
                              const char* expr,
                              const char* file,
                              int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr,
                     "CUBLAS_CHECK failed: %s\n  status: %d\n  at %s:%d\n",
                     expr,
                     static_cast<int>(status),
                     file,
                     line);
        std::fflush(stderr);
        std::abort();
    }
}

#define CUDA_CHECK(expr) ::Bencher::cuda_check_impl((expr), #expr, __FILE__, __LINE__)
#define CUBLAS_CHECK(expr) ::Bencher::cublas_check_impl((expr), #expr, __FILE__, __LINE__)


inline double gemm_flops(const GemmShape& shape) {
    return 2.0 * static_cast<double>(shape.m) *
           static_cast<double>(shape.n) *
           static_cast<double>(shape.k);
}

inline double gemm_tflops(const GemmShape& shape, double elapsed_ms) {
    if (elapsed_ms <= 0.0) {
        return 0.0;
    }
    return gemm_flops(shape) / (elapsed_ms * 1.0e9);
}

template <typename T>
inline ErrorMetrics compute_error_metrics(const T* reference,
                                          const T* actual,
                                          std::size_t count) {
    long double sum_sq = 0.0L;
    long double ref_sq = 0.0L;
    long double max_abs = 0.0L;

    for (std::size_t i = 0; i < count; ++i) {
        const long double r = static_cast<long double>(to_float(reference[i]));
        const long double a = static_cast<long double>(to_float(actual[i]));
        const long double diff = a - r;
        const long double abs_diff = std::fabs(diff);

        sum_sq += diff * diff;
        ref_sq += r * r;
        if (abs_diff > max_abs) {
            max_abs = abs_diff;
        }
    }

    ErrorMetrics metrics{};
    metrics.l2_abs = static_cast<double>(std::sqrt(static_cast<long double>(sum_sq)));
    metrics.l2_rel = (ref_sq > 0.0L)
                          ? static_cast<double>(std::sqrt(sum_sq / ref_sq))
                          : 0.0;
    metrics.max_abs = static_cast<double>(max_abs);
    return metrics;
}

template <typename LaunchFn>
BenchmarkResult benchmark_kernel(const GemmShape& shape,
                                 LaunchFn&& launch,
                                 int warmup_iters = 10,
                                 int bench_iters = 100,
                                 cudaStream_t stream = 0);

TestCase generate_square_case(uint n, std::uint32_t seed);

}  // namespace Bencher
