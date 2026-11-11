cmake_minimum_required(VERSION 3.14)
project(InferenceBenchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Benchmarks are meaningless in Debug, so default to Release.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenMP REQUIRED)

# Path to an extracted ONNX Runtime release (contains include/ and lib/).
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime" CACHE PATH "ONNX Runtime install directory")

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  PATHS "${ONNXRUNTIME_ROOT}/include" "${ONNXRUNTIME_ROOT}/include/onnxruntime"
  NO_DEFAULT_PATH)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
  PATHS "${ONNXRUNTIME_ROOT}/lib"
  NO_DEFAULT_PATH)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime not found in ${ONNXRUNTIME_ROOT}. "
                      "Download a release and pass -DONNXRUNTIME_ROOT=/path/to/onnxruntime.")
endif()

add_executable(benchmark_app src/main.cpp)

target_include_directories(benchmark_app PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(benchmark_app PRIVATE "${ONNXRUNTIME_LIBRARY}" OpenMP::OpenMP_CXX)

# Absolute default model path, so the binary works from any working directory.
target_compile_definitions(benchmark_app PRIVATE
  DEFAULT_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/fraud_model.onnx")

# Find libonnxruntime.so at runtime without LD_LIBRARY_PATH (Linux/macOS).
get_filename_component(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
set_target_properties(benchmark_app PROPERTIES BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

# On Windows, copy onnxruntime.dll next to the executable.
if(WIN32 AND EXISTS "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll")
  add_custom_command(TARGET benchmark_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll" $<TARGET_FILE_DIR:benchmark_app>)
endif()
