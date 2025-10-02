"""
Advanced example demonstrating different chunking strategies and memory management.
"""

import torch
import numpy as np
import time
import accelera as acc

def benchmark_strategies(A, B, engine):
    """Benchmark different chunking strategies."""
    strategies = ['adaptive', 'row']
    results = {}
    
    print("Benchmarking chunking strategies...")
    print("-" * 40)
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy...")
        
        # Clear GPU memory before each test
        engine.cleanup()
        
        start_time = time.time()
        try:
            result = engine.matmul(A, B, chunk_strategy=strategy)
            end_time = time.time()
            
            results[strategy] = {
                'time': end_time - start_time,
                'success': True,
                'shape': result.shape
            }
            
            print(f"✅ {strategy}: {end_time - start_time:.2f}s")
            
        except Exception as e:
            results[strategy] = {
                'time': float('inf'),
                'success': False,
                'error': str(e)
            }
            print(f"❌ {strategy}: {str(e)}")
    
    return results

def memory_pressure_test(engine):
    """Test behavior under memory pressure."""
    print("\n" + "=" * 50)
    print("Memory Pressure Test")
    print("=" * 50)
    
    # Start with small matrices and gradually increase size
    sizes = [(1000, 1000), (2000, 2000), (4000, 4000), (6000, 6000)]
    
    for M, K in sizes:
        N = K
        print(f"\nTesting {M}x{K} @ {K}x{N} matrices...")
        
        try:
            # Create matrices
            A = acc.Matrix.randn((M, K), dtype=torch.float32)
            B = acc.Matrix.randn((K, N), dtype=torch.float32)
            
            # Get memory info before operation
            mem_before = engine.get_memory_info()
            
            # Perform operation
            start_time = time.time()
            C = engine.matmul(A, B)
            end_time = time.time()
            
            # Get memory info after operation
            mem_after = engine.get_memory_info()
            
            print(f"✅ Completed in {end_time - start_time:.2f}s")
            print(f"   GPU usage: {mem_before['gpu_utilization']:.1f}% → {mem_after['gpu_utilization']:.1f}%")
            print(f"   Result shape: {C.shape}")
            
            # Cleanup
            del A, B, C
            engine.cleanup()
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
            break

def compare_with_pytorch(engine):
    """Compare performance with standard PyTorch operations."""
    print("\n" + "=" * 50)
    print("Comparison with Standard PyTorch")
    print("=" * 50)
    
    # Use moderate size that should fit in GPU memory
    M, K, N = 2000, 1500, 2500
    
    print(f"Comparing {M}x{K} @ {K}x{N} matrix multiplication...")
    
    # Create test matrices
    A_acc = acc.Matrix.randn((M, K), dtype=torch.float32)
    B_acc = acc.Matrix.randn((K, N), dtype=torch.float32)
    
    A_torch = A_acc.tensor().clone()
    B_torch = B_acc.tensor().clone()
    
    # Test standard PyTorch (if it fits in memory)
    try:
        print("\nTesting standard PyTorch...")
        if torch.cuda.is_available():
            A_gpu = A_torch.cuda()
            B_gpu = B_torch.cuda()
            
            torch.cuda.synchronize()
            start_time = time.time()
            
            C_torch = torch.matmul(A_gpu, B_gpu)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            pytorch_time = end_time - start_time
            print(f"✅ PyTorch: {pytorch_time:.3f}s")
            
            # Move result back to CPU for comparison
            C_torch_cpu = C_torch.cpu()
            
            # Cleanup GPU memory
            del A_gpu, B_gpu, C_torch
            torch.cuda.empty_cache()
            
        else:
            print("❌ CUDA not available for PyTorch comparison")
            return
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("❌ PyTorch: Out of memory")
            pytorch_time = float('inf')
            C_torch_cpu = None
        else:
            raise
    
    # Test Accelera
    print("Testing Accelera...")
    start_time = time.time()
    C_acc = engine.matmul(A_acc, B_acc)
    end_time = time.time()
    
    accelera_time = end_time - start_time
    print(f"✅ Accelera: {accelera_time:.3f}s")
    
    # Compare results if both succeeded
    if C_torch_cpu is not None:
        print("\nComparing results...")
        diff = torch.abs(C_torch_cpu - C_acc.tensor()).max()
        print(f"Maximum difference: {diff:.2e}")
        
        if diff < 1e-4:
            print("✅ Results match!")
        else:
            print("❌ Results don't match!")
    
    # Performance comparison
    if pytorch_time != float('inf'):
        if accelera_time < pytorch_time:
            speedup = pytorch_time / accelera_time
            print(f"🚀 Accelera is {speedup:.2f}x faster!")
        else:
            slowdown = accelera_time / pytorch_time
            print(f"⚠️ Accelera is {slowdown:.2f}x slower (but uses less memory)")

def main():
    print("🔬 Accelera Framework - Advanced Example")
    print("=" * 50)
    
    # Initialize engine
    engine = acc.MatrixEngine(auto_detect_memory=True, enable_progress=True)
    
    # Display system info
    memory_info = engine.get_memory_info()
    print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'}")
    print(f"GPU Memory: {memory_info['gpu_total_gb']:.2f} GB total")
    print(f"CPU Memory: {memory_info['cpu_total_gb']:.2f} GB total")
    
    # Create test matrices
    M, K, N = 3000, 2500, 3500
    print(f"\nCreating test matrices: {M}x{K} @ {K}x{N}")
    
    A = acc.Matrix.randn((M, K), dtype=torch.float32)
    B = acc.Matrix.randn((K, N), dtype=torch.float32)
    
    print(f"Total memory needed: {(A.memory_size() + B.memory_size() + M*N*4) / 1e9:.2f} GB")
    
    # Benchmark different strategies
    benchmark_results = benchmark_strategies(A, B, engine)
    
    # Find best strategy
    best_strategy = min(
        [k for k, v in benchmark_results.items() if v['success']], 
        key=lambda k: benchmark_results[k]['time']
    )
    print(f"\n🏆 Best strategy: {best_strategy} ({benchmark_results[best_strategy]['time']:.2f}s)")
    
    # Memory pressure test
    memory_pressure_test(engine)
    
    # Compare with PyTorch
    compare_with_pytorch(engine)
    
    # Manual chunking example
    print("\n" + "=" * 50)
    print("Manual Chunking Configuration")
    print("=" * 50)
    
    # Disable auto memory detection and set manual chunk size
    engine.enable_auto_memory_detection(False)
    engine.set_chunk_size(500)  # Small chunks
    
    print("Testing with manual chunk size of 500...")
    start_time = time.time()
    C_manual = engine.matmul(A, B)
    end_time = time.time()
    
    print(f"✅ Manual chunking: {end_time - start_time:.2f}s")
    
    # Re-enable auto detection
    engine.enable_auto_memory_detection(True)
    
    # Final cleanup
    engine.cleanup()
    print("\n🎉 Advanced example completed!")

if __name__ == "__main__":
    main()