#include "bencher.h"

namespace Bencher {

template <typename LaunchFn>
BenchmarkResult benchmark_kernel(const GemmShape& shape,
                                 LaunchFn&& launch,
                                 int warmup_iters,
                                 int bench_iters,
                                 cudaStream_t stream) {
    BenchmarkResult result{};

    result.shape = shape;
    result.warmup_iters = warmup_iters;
    result.bench_iters = bench_iters;

    auto&& kernel = launch;

    for (int i = 0; i < warmup_iters; ++i) {
        kernel(shape, stream);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < bench_iters; ++i) {
        kernel(shape, stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    result.total_ms = total_ms;
    result.avg_ms = (bench_iters > 0) ? (total_ms / static_cast<float>(bench_iters)) : 0.0f;
    result.tflops = gemm_tflops(shape, static_cast<double>(result.avg_ms));
    result.gflops = result.tflops * 1000.0;

    return result;
}



TestCase generate_square_case(uint n, std::uint32_t seed) {
    TestCase test_case{};
    test_case.shape = {n, n, n};
    std::mt19937 rng(seed);
    std::vector<float> a_matrix(n * n);
    std::vector<float> b_matrix(n * n);

    auto random_float = [&rng]() {
        return std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
    };

    for (auto& value : a_matrix) {
        value = random_float();
    }

    for (auto& value : b_matrix) {
            value = random_float();
        }
    test_case.a_f32 = a_matrix;
    test_case.b_f32 = b_matrix;
    return test_case;
}

}  // namespace Bencher
