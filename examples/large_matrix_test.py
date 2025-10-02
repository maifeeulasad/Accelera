"""
Large Matrix Test - Demonstrates chunking on memory-constrained scenarios.

This test creates very large matrices that should trigger chunking even on high-end GPUs.
"""

import torch
import accelera as acc
import time

def test_large_matrices():
    print("🔥 Large Matrix Chunking Test")
    print("=" * 50)
    
    # Initialize engine
    engine = acc.MatrixEngine(auto_detect_memory=True, enable_progress=True)
    
    # Show initial memory
    memory_info = engine.get_memory_info()
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Total GPU Memory: {memory_info['gpu_total_gb']:.2f} GB")
    print(f"Available GPU Memory: {memory_info['gpu_available_gb']:.2f} GB")
    print()
    
    # Test different matrix sizes to find chunking threshold
    test_sizes = [
        (8000, 6000, 10000),   # ~1.9 GB total
        (12000, 8000, 15000),  # ~5.7 GB total  
        (15000, 10000, 18000), # ~10.8 GB total
        (20000, 12000, 25000), # ~22.4 GB total - should definitely chunk
    ]
    
    for i, (M, K, N) in enumerate(test_sizes):
        total_memory_gb = (M * K + K * N + M * N) * 4 / 1e9
        print(f"Test {i+1}: {M}x{K} @ {K}x{N}")
        print(f"Estimated memory needed: {total_memory_gb:.2f} GB")
        
        try:
            # Create large matrices
            print("Creating matrices...")
            A = acc.Matrix.randn((M, K), dtype=torch.float32)
            B = acc.Matrix.randn((K, N), dtype=torch.float32)
            
            print(f"Matrix A: {A.memory_size() / 1e9:.2f} GB")
            print(f"Matrix B: {B.memory_size() / 1e9:.2f} GB")
            
            # Perform multiplication
            print("Computing matrix multiplication...")
            start_time = time.time()
            
            C = engine.matmul(A, B)
            
            end_time = time.time()
            
            print(f"✅ Success! Time: {end_time - start_time:.2f}s")
            print(f"Result shape: {C.shape}")
            print(f"Result memory: {C.memory_size() / 1e9:.2f} GB")
            
            # Show final memory usage
            final_memory = engine.get_memory_info()
            print(f"GPU utilization: {final_memory['gpu_utilization']:.1f}%")
            
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
        
        print("-" * 50)
        
        # Cleanup between tests
        engine.cleanup()
        torch.cuda.empty_cache()

def test_forced_chunking():
    print("\n🧪 Forced Chunking Test")
    print("=" * 50)
    
    # Force small chunk size to demonstrate chunking on moderate matrices
    engine = acc.MatrixEngine(
        auto_detect_memory=False,
        chunking_strategy='adaptive',
        enable_progress=True
    )
    
    # Set very small chunk size to force chunking
    engine.set_chunk_size(256)
    
    print("Using forced small chunk size: 256")
    print("This will demonstrate chunking even on smaller matrices.")
    print()
    
    # Create moderate-sized matrices
    M, K, N = 4000, 3000, 5000
    print(f"Creating {M}x{K} @ {K}x{N} matrices...")
    
    A = acc.Matrix.randn((M, K), dtype=torch.float32)
    B = acc.Matrix.randn((K, N), dtype=torch.float32)
    
    print(f"Matrix A memory: {A.memory_size() / 1e6:.1f} MB")
    print(f"Matrix B memory: {B.memory_size() / 1e6:.1f} MB")
    print()
    
    # This should now use chunking
    print("Computing with forced chunking...")
    start_time = time.time()
    
    C = engine.matmul(A, B)
    
    end_time = time.time()
    
    print(f"✅ Completed with chunking! Time: {end_time - start_time:.2f}s")
    print(f"Result shape: {C.shape}")

def test_memory_efficiency():
    print("\n💾 Memory Efficiency Comparison")
    print("=" * 50)
    
    M, K, N = 6000, 4000, 8000
    print(f"Testing {M}x{K} @ {K}x{N} matrices")
    
    # Test 1: Standard PyTorch (if it fits)
    try:
        print("\n1. Standard PyTorch approach:")
        A_torch = torch.randn(M, K, dtype=torch.float32).cuda()
        B_torch = torch.randn(K, N, dtype=torch.float32).cuda()
        
        start_time = time.time()
        C_torch = torch.matmul(A_torch, B_torch)
        end_time = time.time()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"✅ PyTorch: {end_time - start_time:.3f}s")
        print(f"Peak GPU memory: {peak_memory:.2f} GB")
        
        # Cleanup
        del A_torch, B_torch, C_torch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("❌ PyTorch: Out of memory!")
        else:
            raise
    
    # Test 2: Accelera approach
    print("\n2. Accelera approach:")
    torch.cuda.reset_peak_memory_stats()
    
    engine = acc.MatrixEngine(enable_progress=False)
    A_acc = acc.Matrix.randn((M, K), dtype=torch.float32)
    B_acc = acc.Matrix.randn((K, N), dtype=torch.float32)
    
    start_time = time.time()
    C_acc = engine.matmul(A_acc, B_acc)
    end_time = time.time()
    
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    print(f"✅ Accelera: {end_time - start_time:.3f}s")
    print(f"Peak GPU memory: {peak_memory:.2f} GB")
    
    engine.cleanup()

def main():
    print("🚀 Accelera Large Matrix Testing Suite")
    print("This will test the chunking capabilities of the Accelera framework.")
    print()
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires a CUDA-capable GPU.")
        return
    
    # Run tests
    test_large_matrices()
    test_forced_chunking()
    test_memory_efficiency()
    
    print("\n🎉 Large matrix testing completed!")

if __name__ == "__main__":
    main()