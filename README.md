# C++ Inference Engine & Memory Locality Benchmark 🏎️

This project explores the intersection of Low-Level Systems Engineering and Deep Learning. It features a standalone C++ inference engine for fraud detection and a benchmark suite demonstrating the physical impact of CPU cache locality on performance.

## 🛠️ Tech Stack
* **Language:** C++17
* **Runtime:** ONNX Runtime (C++ API)
* **Build System:** CMake
* **Parallelism:** OpenMP

## 🧠 Model Integration
The project loads a pre-trained Fraud Detection model (exported from PyTorch via ONNX) and executes inference in a bare-metal environment. 

* **Input:** 10-feature vector
* **Consistency Check:** C++ output matches Python (FastAPI) output exactly to 6 decimal places ($0.814864$).

## 📊 The Physics of Performance: Cache Locality
To demonstrate the importance of hardware-aware programming, I implemented a benchmark comparing two different ways of traversing a 400MB matrix ($10,000 \times 10,000$ elements).

| Access Pattern | Execution Time | Context |
| :--- | :--- | :--- |
| **Row-Major** | 376.8 ms | **Cache-Friendly:** Follows C++ memory layout. |
| **Column-Major** | 428.3 ms | **Cache-Killer:** Forces constant RAM fetches. |

### Key Insight:
Even with modern **Hardware Prefetching**, the Column-Major access was **1.14x slower**. This proves that high-performance ML models must be designed with the CPU's L1/L2 cache hierarchy in mind to avoid "Memory Walls."

## 🚀 How to Build
```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
.\Release\benchmark_app.exe