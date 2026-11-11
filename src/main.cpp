#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

#include <omp.h>
#include <onnxruntime_cxx_api.h>

#ifndef DEFAULT_MODEL_PATH
#define DEFAULT_MODEL_PATH "models/fraud_model.onnx"
#endif

namespace {

// Reference output produced by the original PyTorch/ONNX pipeline in Python
// for the input vector below.
constexpr float kExpectedProbability = 0.814864f;
constexpr float kTolerance = 1e-5f;

constexpr std::size_t kRows = 10000;
constexpr std::size_t kCols = 10000;
constexpr int kWarmupRuns = 1;
constexpr int kTimedRuns = 5;

// Keeps the compiler from optimising the benchmark loops away without
// putting a memory store inside the hot loop (which `volatile` would do).
volatile double g_sink = 0.0;

double sum_row_major(const std::vector<float>& m, std::size_t rows, std::size_t cols) {
    double sum = 0.0;
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            sum += m[i * cols + j];
    return sum;
}

double sum_col_major(const std::vector<float>& m, std::size_t rows, std::size_t cols) {
    double sum = 0.0;
    for (std::size_t j = 0; j < cols; ++j)
        for (std::size_t i = 0; i < rows; ++i)
            sum += m[i * cols + j];
    return sum;
}

double sum_row_major_omp(const std::vector<float>& m, std::size_t rows, std::size_t cols) {
    double sum = 0.0;
    const long long n_rows = static_cast<long long>(rows);
#pragma omp parallel for reduction(+ : sum) schedule(static)
    for (long long i = 0; i < n_rows; ++i)
        for (std::size_t j = 0; j < cols; ++j)
            sum += m[static_cast<std::size_t>(i) * cols + j];
    return sum;
}

using SumFn = std::function<double(const std::vector<float>&, std::size_t, std::size_t)>;

// Returns the median wall-clock time in milliseconds over kTimedRuns runs.
double time_median_ms(const SumFn& fn, const std::vector<float>& m) {
    for (int r = 0; r < kWarmupRuns; ++r) g_sink = fn(m, kRows, kCols);

    std::vector<double> times;
    times.reserve(kTimedRuns);
    for (int r = 0; r < kTimedRuns; ++r) {
        const auto start = std::chrono::steady_clock::now();
        g_sink = fn(m, kRows, kCols);
        const auto end = std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

bool run_inference_check(const std::filesystem::path& model_path) {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "InferenceEngine");
    Ort::SessionOptions session_options;
    // path::c_str() is wchar_t* on Windows and char* elsewhere, matching ORTCHAR_T.
    Ort::Session session(env, model_path.c_str(), session_options);

    std::vector<float> input_features = {0.5f, -1.2f, 3.14f, 0.0f, 1.1f, -0.5f, 2.2f, 0.9f, -3.0f, 4.2f};
    std::vector<int64_t> input_shape = {1, 10};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_features.data(), input_features.size(), input_shape.data(), input_shape.size());

    const char* input_names[] = {"input_features"};
    const char* output_names[] = {"fraud_probability"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    const float prob = outputs.front().GetTensorMutableData<float>()[0];

    const bool ok = std::fabs(prob - kExpectedProbability) <= kTolerance;
    std::cout << std::fixed << std::setprecision(6)
              << "Model inference: " << prob << " (expected " << kExpectedProbability << ") "
              << (ok ? "[OK]" : "[MISMATCH]") << std::endl;
    return ok;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const std::filesystem::path model_path = argc > 1 ? argv[1] : DEFAULT_MODEL_PATH;
        if (!run_inference_check(model_path)) return EXIT_FAILURE;

        std::cout << "\n--- Memory access benchmark (" << kRows << "x" << kCols << " floats, "
                  << (kRows * kCols * sizeof(float)) / (1024 * 1024) << " MiB, median of "
                  << kTimedRuns << " runs) ---" << std::endl;

        const std::vector<float> matrix(kRows * kCols, 1.1f);

        const double row_ms = time_median_ms(sum_row_major, matrix);
        const double col_ms = time_median_ms(sum_col_major, matrix);
        const double omp_ms = time_median_ms(sum_row_major_omp, matrix);

        std::cout << std::setprecision(1);
        std::cout << "Row-major, 1 thread (cache-friendly): " << row_ms << " ms" << std::endl;
        std::cout << "Column-major, 1 thread (strided):     " << col_ms << " ms" << std::endl;
        std::cout << "Row-major, " << omp_get_max_threads() << " OpenMP threads:          " << omp_ms << " ms"
                  << std::endl;

        std::cout << std::setprecision(2);
        std::cout << "\n[RESULT] Column-major is " << col_ms / row_ms << "x slower than row-major." << std::endl;
        std::cout << "[RESULT] OpenMP row-major is " << row_ms / omp_ms << "x faster than single-threaded."
                  << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
