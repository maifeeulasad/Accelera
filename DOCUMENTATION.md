# Accelera Documentation

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Concepts](#core-concepts)
5. [API Reference](#api-reference)
6. [Advanced Usage](#advanced-usage)
7. [Performance Guidelines](#performance-guidelines)
8. [Troubleshooting](#troubleshooting)

## Overview

Accelera is a memory-efficient matrix operations framework designed to handle large matrix computations on memory-constrained GPUs. It automatically manages memory allocation, chunks operations when necessary, and provides a seamless API that prevents Out-of-Memory (OOM) errors.

### Key Features

- **Automatic Memory Management**: Monitors GPU memory and automatically switches between direct and chunked computation
- **Intelligent Chunking**: Multiple strategies for breaking down large operations
- **CPU-GPU Orchestration**: Seamless data movement between CPU and GPU memory
- **Progress Tracking**: Real-time progress bars for long-running operations
- **Multiple Input Types**: Works with PyTorch tensors, NumPy arrays, and custom Matrix objects

### When to Use Accelera

- Working with large matrices that don't fit in GPU memory
- Need to perform matrix operations on memory-constrained GPUs (e.g., consumer GPUs)
- Want automatic memory management without manual chunking
- Batch processing of multiple large matrix operations

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA support
- NVIDIA GPU with CUDA drivers
- Sufficient CPU RAM for temporary storage

### Install from Source

```bash
git clone https://github.com/maifeeulasad/accelera
cd accelera
python3 -m build --no-isolation
make install       # to install as library only
make install-bin   # to install the executable like system
```

### Verify Installation

```python
import accelera as acc
import torch

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")

# Initialize engine (will show GPU info)
engine = acc.MatrixEngine()
```

## Quick Start

### Basic Matrix Multiplication

```python
import accelera as acc

# Initialize engine with automatic memory detection
engine = acc.MatrixEngine(auto_detect_memory=True)

# Create large matrices
A = acc.Matrix.randn((8000, 6000))  # 8k x 6k matrix
B = acc.Matrix.randn((6000, 10000)) # 6k x 10k matrix

# Perform multiplication (automatically chunks if needed)
C = engine.matmul(A, B)

print(f"Result shape: {C.shape}")  # (8000, 10000)
```

### Element-wise Operations

```python
# Element-wise operations
X = acc.Matrix.randn((5000, 4000))
Y = acc.Matrix.randn((5000, 4000))

# Addition
Z1 = engine.add(X, Y)

# Element-wise multiplication  
Z2 = engine.multiply(X, Y)
```

### Working with Different Input Types

```python
import numpy as np
import torch

# NumPy arrays
A_np = np.random.randn(1000, 800).astype(np.float32)
B_np = np.random.randn(800, 1200).astype(np.float32)
C = engine.matmul(A_np, B_np)

# PyTorch tensors
A_torch = torch.randn(1000, 800)
B_torch = torch.randn(800, 1200)
C = engine.matmul(A_torch, B_torch)
```

## Core Concepts

### Memory Manager

The `MemoryManager` handles all GPU memory operations:

- **Memory Monitoring**: Tracks available GPU and CPU memory
- **Automatic Cleanup**: Frees unused GPU memory
- **Safe Transfers**: Validates memory before GPU transfers
- **Optimal Sizing**: Calculates optimal chunk sizes

### Chunking Strategies

Accelera supports multiple chunking strategies:

#### Row Chunking (Default)
Processes matrices in row-wise chunks. Best for most matrix operations.

```python
from accelera.chunking import RowChunking

strategy = RowChunking()
engine = acc.MatrixEngine(chunking_strategy='row')
```

#### Adaptive Chunking  
Dynamically adjusts chunk size based on memory pressure.

```python
engine = acc.MatrixEngine(chunking_strategy='adaptive')
```

#### Tile Chunking
Processes matrices in 2D tiles for better cache locality.

```python
from accelera.chunking import TileChunking

strategy = TileChunking(tile_size=(1024, 1024))
```

### Matrix Wrapper

The `Matrix` class provides a high-level interface:

```python
# Create matrices
A = acc.Matrix.zeros((1000, 500))    # Zero matrix
B = acc.Matrix.ones((500, 800))      # Ones matrix  
C = acc.Matrix.eye(1000)             # Identity matrix
D = acc.Matrix.randn((1000, 500))    # Random normal

# Matrix operations
E = A + B  # Addition
F = A * B  # Element-wise multiplication
G = A @ B  # Matrix multiplication (use engine for large matrices)

# Properties
print(f"Shape: {A.shape}")
print(f"Device: {A.device}")
print(f"Memory: {A.memory_size() / 1e6:.1f} MB")
```

## API Reference

### MatrixEngine

#### Constructor

```python
MatrixEngine(
    device: Optional[str] = None,
    auto_detect_memory: bool = True,
    chunking_strategy: str = 'adaptive',
    chunk_size: Optional[int] = None,
    enable_progress: bool = True
)
```

**Parameters:**
- `device`: CUDA device (e.g., 'cuda:0'). Auto-detects if None.
- `auto_detect_memory`: Whether to automatically detect optimal chunk sizes
- `chunking_strategy`: Strategy for chunking ('row', 'tile', 'adaptive')
- `chunk_size`: Fixed chunk size (ignored if auto_detect_memory=True)
- `enable_progress`: Whether to show progress bars

#### Methods

##### `matmul(A, B, chunk_strategy=None)`

Matrix multiplication with automatic memory management.

**Parameters:**
- `A`: Left matrix (M x K)
- `B`: Right matrix (K x N)
- `chunk_strategy`: Override default chunking strategy

**Returns:** Result matrix (M x N)

##### `add(A, B)` 

Element-wise addition.

##### `multiply(A, B)`

Element-wise multiplication.

##### `get_memory_info()`

Returns current memory usage statistics.

##### `cleanup()`

Force cleanup of GPU memory.

### Matrix

#### Constructor

```python
Matrix(
    data: Union[torch.Tensor, np.ndarray, list],
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None
)
```

#### Static Methods

```python
Matrix.zeros(shape, dtype=torch.float32, device='cpu')
Matrix.ones(shape, dtype=torch.float32, device='cpu')  
Matrix.eye(n, m=None, dtype=torch.float32, device='cpu')
Matrix.random(shape, dtype=torch.float32, device='cpu')
Matrix.randn(shape, dtype=torch.float32, device='cpu')
```

#### Properties

- `shape`: Matrix dimensions
- `dtype`: Data type
- `device`: Current device
- `storage_device`: Storage device (CPU)

#### Methods

- `to_gpu(device='cuda:0')`: Move to GPU
- `to_cpu()`: Move to CPU storage
- `tensor()`: Get underlying PyTorch tensor
- `numpy()`: Convert to NumPy array
- `clone()`: Create deep copy

## Advanced Usage

### Custom Chunking Strategy

```python
from accelera.chunking import ChunkingStrategy

class CustomChunking(ChunkingStrategy):
    def get_chunks(self, shape, chunk_size):
        # Implement custom chunking logic
        pass

# Use custom strategy
engine = acc.MatrixEngine()
engine.chunking_strategy = CustomChunking()
```

### Manual Memory Management

```python
# Disable automatic memory detection
engine = acc.MatrixEngine(auto_detect_memory=False)

# Set manual chunk size
engine.set_chunk_size(512)

# Re-enable automatic detection
engine.enable_auto_memory_detection(True)
```

### Memory Monitoring

```python
# Get detailed memory info
memory_info = engine.get_memory_info()

print(f"GPU utilization: {memory_info['gpu_utilization']:.1f}%")
print(f"GPU available: {memory_info['gpu_available_gb']:.2f} GB")
print(f"CPU available: {memory_info['cpu_available_gb']:.2f} GB")
```

### Batch Processing

```python
# Process multiple matrix pairs efficiently
matrices_A = [acc.Matrix.randn((1000, 800)) for _ in range(10)]
matrices_B = [acc.Matrix.randn((800, 1200)) for _ in range(10)]

results = []
for A, B in zip(matrices_A, matrices_B):
    C = engine.matmul(A, B)
    results.append(C)
    
    # Optional: cleanup between operations
    engine.cleanup()
```

## Performance Guidelines

### Optimal Matrix Sizes

- **Row-major operations**: Ensure the first dimension is chunkable
- **Memory efficiency**: Matrices with ~1-10GB total size work well
- **Chunk sizes**: Usually 500-2000 rows per chunk depending on GPU memory

### Memory Optimization

```python
# Monitor memory usage
def monitor_operation():
    before = engine.get_memory_info()
    result = engine.matmul(A, B)
    after = engine.get_memory_info()
    
    print(f"Memory change: {after['gpu_utilization'] - before['gpu_utilization']:.1f}%")
    return result

# Force cleanup after operations
engine.cleanup()

# Use smaller chunk sizes for memory-constrained environments
engine.set_chunk_size(256)
```

### Performance Tuning

```python
# Disable progress bars for batch processing
engine = acc.MatrixEngine(enable_progress=False)

# Use row chunking for better performance
engine = acc.MatrixEngine(chunking_strategy='row')

# Prefetch chunks for better throughput
from accelera.chunking import ChunkIterator
iterator = ChunkIterator(tensor, strategy, chunk_size, memory_manager, prefetch=True)
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory

**Problem**: Even with chunking, getting OOM errors.

**Solutions:**
1. Reduce chunk size manually:
   ```python
   engine.set_chunk_size(128)  # Smaller chunks
   ```

2. Check for memory leaks:
   ```python
   engine.cleanup()  # Force cleanup
   ```

3. Monitor memory usage:
   ```python
   print(engine.get_memory_info())
   ```

#### Slow Performance

**Problem**: Operations are slower than expected.

**Solutions:**
1. Check if chunking is necessary:
   ```python
   # Monitor if direct computation is possible
   memory_needed = A.memory_size() + B.memory_size()
   available = engine.memory_manager.get_available_memory()
   print(f"Direct computation possible: {memory_needed < available}")
   ```

2. Optimize chunking strategy:
   ```python
   # Try row chunking for better performance
   engine = acc.MatrixEngine(chunking_strategy='row')
   ```

3. Disable progress bars:
   ```python
   engine = acc.MatrixEngine(enable_progress=False)
   ```

#### Dimension Mismatch

**Problem**: Matrix dimension errors.

**Solution:**
```python
print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")

# For matmul: A.shape[-1] must equal B.shape[-2]
assert A.shape[-1] == B.shape[-2], f"Incompatible shapes: {A.shape} @ {B.shape}"
```

### Debugging

#### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run operations with debug info
engine = acc.MatrixEngine()
result = engine.matmul(A, B)
```

#### Memory Debugging

```python
def debug_memory():
    info = engine.get_memory_info()
    for key, value in info.items():
        print(f"{key}: {value}")

debug_memory()  # Before operation
result = engine.matmul(A, B)
debug_memory()  # After operation
```

### Environment Variables

Configure Accelera behavior with environment variables:

```bash
export ACCELERA_MEMORY_THRESHOLD=0.8    # Use 80% of GPU memory
export ACCELERA_MIN_CHUNK_SIZE=1        # Minimum chunk size
export ACCELERA_MAX_CHUNK_SIZE=5000     # Maximum chunk size
export ACCELERA_ENABLE_PREFETCH=true    # Enable chunk prefetching
export ACCELERA_LOG_LEVEL=DEBUG         # Set logging level
```

### Performance Benchmarking

```python
import time

def benchmark_operation(A, B, num_runs=3):
    times = []
    
    for i in range(num_runs):
        engine.cleanup()  # Clean start
        
        start = time.time()
        result = engine.matmul(A, B)
        end = time.time()
        
        times.append(end - start)
        print(f"Run {i+1}: {times[-1]:.2f}s")
    
    avg_time = sum(times) / len(times)
    print(f"Average: {avg_time:.2f}s")
    
    return result, avg_time

# Benchmark your operations
A = acc.Matrix.randn((2000, 1500))
B = acc.Matrix.randn((1500, 2500))
result, avg_time = benchmark_operation(A, B)
```