#include <iostream>
#include <vector>
#include <chrono>
#include <onnxruntime_cxx_api.h>


double run_row_major_benchmark(const std::vector<float>& matrix, int rows, int cols) {
    auto start = std::chrono::high_resolution_clock::now();
    volatile float sum = 0; 
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            sum += matrix[i * cols + j];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}


double run_col_major_benchmark(const std::vector<float>& matrix, int rows, int cols) {
    auto start = std::chrono::high_resolution_clock::now();
    volatile float sum = 0;
    for (int j = 0; j < cols; ++j) { 
        for (int i = 0; i < rows; ++i) {
            sum += matrix[i * cols + j];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    try {
        
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "InferenceEngine");
        Ort::SessionOptions session_options;
        const wchar_t* model_path = L"models/fraud_model.onnx";
        Ort::Session session(env, model_path, session_options);

        std::vector<float> input_features = {0.5, -1.2, 3.14, 0.0, 1.1, -0.5, 2.2, 0.9, -3.0, 4.2};
        std::vector<int64_t> input_shape = {1, 10};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_features.data(), input_features.size(), input_shape.data(), input_shape.size());

        const char* input_names[] = {"input_features"};
        const char* output_names[] = {"fraud_probability"};
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        
        std::cout << "Model Inference Check: " << output_tensors.front().GetTensorMutableData<float>()[0] << " [OK]" << std::endl;

        
        const int ROWS = 10000;
        const int COLS = 10000;
        std::cout << "\n--- Starting Memory Access Benchmark (" << ROWS << "x" << COLS << ") ---" << std::endl;
        
        std::vector<float> big_matrix(ROWS * COLS, 1.1f);

        double row_time = run_row_major_benchmark(big_matrix, ROWS, COLS);
        std::cout << "Row-Major Access (Cache-Friendly): " << row_time << " ms" << std::endl;

        double col_time = run_col_major_benchmark(big_matrix, ROWS, COLS);
        std::cout << "Column-Major Access (Cache-Killer):  " << col_time << " ms" << std::endl;

        std::cout << "\n[RESULT] Column-Major is " << (col_time / row_time) << "x slower than Row-Major." << std::endl;
        std::cout << "This is why memory layout matters for Deep Learning performance." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}