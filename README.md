# C++ Inference Engine & Memory Locality Benchmark 🏎️

A standalone C++ inference engine for a fraud-detection model, plus a benchmark showing how CPU cache locality and multithreading affect the performance of memory-bound code.

## 🛠️ Tech Stack
* **Language:** C++17
* **Runtime:** ONNX Runtime (C++ API)
* **Build System:** CMake
* **Parallelism:** OpenMP

## 🧠 Model Integration
The project loads a pre-trained fraud-detection model (a small MLP exported from PyTorch to ONNX) and runs inference natively through ONNX Runtime, with no Python involved.

* **Input:** 10-feature vector → **Output:** fraud probability
* **Consistency check:** the C++ output is compared against the Python reference value (`0.814864`) with a tolerance of `1e-5`. The program exits with an error if they don't match.

## 📊 Cache Locality Benchmark
The benchmark sums a 400 MB matrix (`10,000 × 10,000` floats) three ways. Each variant gets one warmup run, then the median of 5 timed runs is reported.

| Access Pattern | Time | Speed vs row-major |
| :--- | ---: | ---: |
| **Row-major, 1 thread** | 143 ms | 1.0x |
| **Column-major, 1 thread** | 790 ms | **5.5x slower** |
| **Row-major, OpenMP (2 threads)** | 73 ms | **2.0x faster** |

*Measured on a 2-vCPU Linux VM, GCC, Release build. Your numbers will vary with hardware.*

### Why it matters
* **Row-major** traversal reads memory sequentially, matching how C++ stores the matrix. Each 64-byte cache line serves 16 floats, and the hardware prefetcher keeps the next lines arriving in time.
* **Column-major** traversal jumps 40 KB between consecutive reads. Almost every access touches a new cache line, so most of each line fetched from RAM is wasted.
* **OpenMP** splits the rows across cores with a `reduction`, which scales nearly linearly here.

The same principle explains why ML frameworks care about tensor layouts (NCHW vs NHWC), contiguous memory, and tiled matrix multiplication.

## 🚀 How to Build
1. Download an [ONNX Runtime release](https://github.com/microsoft/onnxruntime/releases) for your platform and extract it into `onnxruntime/` at the repo root (or pass `-DONNXRUNTIME_ROOT=/path/to/it`).
2. Build and run:

```bash
cmake -S . -B build
cmake --build build --config Release

# Linux / macOS
./build/benchmark_app

# Windows
.\build\Release\benchmark_app.exe
```

The model path is built in at compile time, so the binary runs from any directory. You can also pass a different model path as the first argument.
