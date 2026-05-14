cmake_minimum_required(VERSION 3.14)
project(InferenceBenchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenMP REQUIRED)


include_directories(${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime/lib)

add_executable(benchmark_app src/main.cpp)


target_link_libraries(benchmark_app PRIVATE onnxruntime OpenMP::OpenMP_CXX)