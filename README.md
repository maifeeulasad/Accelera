# Accelera - Memory-Efficient Matrix Operations Framework

A framework for performing large matrix operations on memory-constrained GPUs through intelligent chunking and CPU-GPU memory management.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-required-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## 🚀 Problem Statement

When working with large matrices on GPUs with limited VRAM, operations like matrix multiplication can cause **Out-of-Memory (OOM) errors**. Accelera solves this by:

- **🧩 Breaking large operations into smaller chunks**
- **💾 Intelligently offloading intermediate results to CPU/RAM**  
- **🔄 Dynamically managing GPU memory**
- **🎯 Providing a seamless API for large matrix operations**

## ✨ Features

- **🤖 Automatic chunking** for matrix operations
- **🧠 Dynamic memory management** between GPU and CPU
- **⚡ CUDA-optimized** for NVIDIA GPUs
- **📊 Configurable chunk sizes** based on available VRAM
- **📈 Progress tracking** for long-running operations
- **📋 Memory usage monitoring**
- **🔌 Multiple input types** (PyTorch tensors, NumPy arrays)

## 🏃‍♂️ Quick Start

```python
import accelera as acc

# Initialize with automatic VRAM detection
engine = acc.MatrixEngine(auto_detect_memory=True)

# Perform large matrix multiplication that might cause OOM on small GPUs
A = acc.Matrix.random((10000, 8000))  # 10k x 8k matrix (~305 MB)
B = acc.Matrix.random((8000, 12000))  # 8k x 12k matrix (~366 MB)

# This will automatically chunk and manage memory
C = engine.matmul(A, B)  # Result: 10k x 12k matrix (~458 MB)

print(f"✅ Success! Result shape: {C.shape}")
```

### 🎯 Real-world Example

```python
# Scenario: Training a large neural network layer on a 4GB GPU
import accelera as acc

engine = acc.MatrixEngine()

# Large weight matrix (would normally cause OOM)
weights = acc.Matrix.randn((20000, 15000))  # ~1.1 GB
inputs = acc.Matrix.randn((15000, 8000))    # ~457 MB

# Forward pass - automatically chunked if needed
output = engine.matmul(weights, inputs)     # ~610 MB result

# Check memory usage
memory_info = engine.get_memory_info()
print(f"GPU utilization: {memory_info['gpu_utilization']:.1f}%")
```

## 📦 Installation

### Requirements

- **Python 3.8+**
- **PyTorch 2.0+** with CUDA support
- **NVIDIA GPU** with CUDA drivers
- **Sufficient CPU RAM** for temporary storage

### Install

```bash
# Clone the repository
git clone https://github.com/maifeeulasad/accelera
cd accelera

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# Verify installation
make verify
```

## 🛠️ Usage Examples

### Basic Operations

```python
import accelera as acc
import numpy as np

# Initialize engine
engine = acc.MatrixEngine(auto_detect_memory=True, enable_progress=True)

# Matrix multiplication
A = acc.Matrix.randn((5000, 4000))
B = acc.Matrix.randn((4000, 6000))
C = engine.matmul(A, B)

# Element-wise operations
X = acc.Matrix.randn((3000, 4000))
Y = acc.Matrix.randn((3000, 4000))

# Addition
Z1 = engine.add(X, Y)

# Element-wise multiplication  
Z2 = engine.multiply(X, Y)

# Works with NumPy arrays and PyTorch tensors too!
A_np = np.random.randn(1000, 800).astype(np.float32)
B_np = np.random.randn(800, 1200).astype(np.float32)
C_from_numpy = engine.matmul(A_np, B_np)
```

### Advanced Configuration

```python
# Custom chunking strategy
engine = acc.MatrixEngine(
    chunking_strategy='adaptive',  # 'row', 'tile', 'adaptive'
    chunk_size=1024,               # Manual chunk size
    enable_progress=True           # Show progress bars
)

# Manual memory management
engine.set_chunk_size(512)                    # Smaller chunks for limited memory
engine.enable_auto_memory_detection(False)    # Disable auto-detection
engine.cleanup()                              # Force GPU memory cleanup

# Memory monitoring
memory_info = engine.get_memory_info()
print(f"GPU Memory: {memory_info['gpu_available_gb']:.2f}GB available")
print(f"CPU Memory: {memory_info['cpu_available_gb']:.2f}GB available")
```

## 📊 Performance Comparison

Run the benchmark to see how Accelera performs on your system:

```bash
# Run full benchmark suite
make benchmark

# Test specific matrix size
python examples/benchmark.py --custom-size 4000 3000 5000

# Quick demo
make demo
```

## 📁 Project Structure

```
accelera/
├── accelera/                  # Core framework
│   ├── __init__.py            # Main package exports
│   ├── engine.py              # MatrixEngine - main API
│   ├── matrix.py              # Matrix wrapper class
│   ├── memory_manager.py      # GPU/CPU memory management
│   ├── chunking.py            # Chunking strategies
│   └── config.py              # Configuration and logging
├── examples/                  # Usage examples
│   ├── basic_usage.py         # Basic operations demo
│   ├── advanced_usage.py      # Advanced features demo
│   └── benchmark.py           # Performance benchmarking
├── tests/                     # Unit tests
│   └── test_accelera.py       # Comprehensive test suite
├── DOCUMENTATION.md           # Detailed documentation
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
└── Makefile                   # Development commands
```

## 🧪 Running Examples

```bash
# Basic usage example
python examples/basic_usage.py

# Advanced features demonstration
python examples/advanced_usage.py

# Performance benchmarking
python examples/benchmark.py

# Or use make commands
make examples
make benchmark
```

## 🔧 Development

```bash
# Install development dependencies
make dev-install

# Run tests
make test

# Run linting
make lint

# Format code
make format

# Clean build artifacts
make clean
```

## 📖 Documentation

- **[Complete Documentation](DOCUMENTATION.md)** - Detailed API reference and usage guide
- **[Examples](examples/)** - Practical usage examples  
- **[Tests](tests/)** - Unit tests and integration tests

## 🎯 Use Cases

- **🧠 Deep Learning**: Training large neural networks on consumer GPUs
- **🔬 Scientific Computing**: Large matrix operations in research
- **📊 Data Processing**: Batch processing of large datasets
- **🎮 Computer Graphics**: Large transformation matrices
- **📈 Financial Modeling**: Risk calculations with large covariance matrices

## ⚠️ System Requirements

- **NVIDIA GPU** (optional)
- **CUDA** (not sure about minimum version)

## 🤝 Contributing

Following the guidelines in [`claude.md`](claude.md):

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Follow the coding standards**: Small commits, clear intent, boring solutions
4. **Add tests** for new functionality
5. **Submit a pull request** with clear description

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch** team for the excellent tensor library
- **NVIDIA** for CUDA and GPU computing
- **Community** feedback and contributions

---

**💡 Pro Tip**: Start with the basic example, then explore advanced features. The framework is designed to be simple by default but powerful when needed!

```python
# Get started in 3 lines
import accelera as acc
engine = acc.MatrixEngine()
result = engine.matmul(large_matrix_A, large_matrix_B)
```