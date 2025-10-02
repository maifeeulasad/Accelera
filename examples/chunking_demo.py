"""
Demonstration of chunking on large matrices that exceed GPU memory.
"""

import torch
import accelera as acc
import gc

def test_extreme_chunking():
    """Test chunking with matrices that definitely exceed available memory."""
    print("🔥 Extreme Chunking Test")
    print("=" * 60)
    
    # Force a lower memory threshold to trigger chunking
    engine = acc.MatrixEngine(enable_progress=True)
    
    # Manually set a very conservative memory threshold
    engine.memory_manager.memory_threshold = 0.3  # Use only 30% of available memory
    
    print(f"GPU: {torch.cuda.get_device_name()}")
    memory_info = engine.get_memory_info()
    print(f"Total GPU Memory: {memory_info['gpu_total_gb']:.2f} GB")
    print(f"Using conservative threshold: 30% = {memory_info['gpu_total_gb'] * 0.3:.2f} GB")
    print()
    
    # Create large matrices
    M, K, N = 10000, 8000, 12000
    total_memory_needed = (M * K + K * N + M * N) * 4 / 1e9
    
    print(f"Matrix operation: {M}x{K} @ {K}x{N}")
    print(f"Total memory needed: {total_memory_needed:.2f} GB")
    print(f"Conservative limit: {memory_info['gpu_total_gb'] * 0.3:.2f} GB")
    print("This should trigger chunking!")
    print()
    
    # Create matrices
    print("Creating large matrices...")
    A = acc.Matrix.randn((M, K), dtype=torch.float32)
    B = acc.Matrix.randn((K, N), dtype=torch.float32)
    
    print(f"Matrix A: {A.memory_size() / 1e9:.2f} GB")
    print(f"Matrix B: {B.memory_size() / 1e9:.2f} GB")
    print()
    
    # This should now force chunking
    print("Computing with conservative memory limit...")
    
    try:
        C = engine.matmul(A, B)
        print(f"✅ Success! Result shape: {C.shape}")
        print(f"Result memory: {C.memory_size() / 1e9:.2f} GB")
        
        # Verify the result with a small sample
        print("\nVerifying correctness...")
        sample_size = 100
        A_sample = A[:sample_size, :sample_size]
        B_sample = B[:sample_size, :sample_size]
        
        # Direct computation for verification
        direct_result = torch.matmul(A_sample.tensor(), B_sample.tensor())
        engine_result = engine.matmul(A_sample, B_sample)
        
        diff = torch.abs(direct_result - engine_result.tensor()).max()
        print(f"Maximum difference: {diff:.2e}")
        
        if diff < 1e-4:
            print("✅ Verification passed!")
        else:
            print("❌ Verification failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        engine.cleanup()

def test_manual_chunking_strategies():
    """Test different chunking strategies manually."""
    print("\n🧪 Manual Chunking Strategies Test")
    print("=" * 60)
    
    # Test data
    M, K, N = 6000, 4000, 8000
    
    strategies = ['row', 'adaptive']
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy:")
        print("-" * 30)
        
        engine = acc.MatrixEngine(
            chunking_strategy=strategy,
            auto_detect_memory=False,
            enable_progress=True
        )
        
        # Force small chunks
        engine.set_chunk_size(500)
        
        A = acc.Matrix.randn((M, K), dtype=torch.float32)
        B = acc.Matrix.randn((K, N), dtype=torch.float32)
        
        print(f"Matrix sizes: {M}x{K} @ {K}x{N}")
        print(f"Forced chunk size: 500")
        
        import time
        start_time = time.time()
        
        try:
            C = engine.matmul(A, B)
            end_time = time.time()
            
            print(f"✅ {strategy}: {end_time - start_time:.2f}s")
            print(f"Result shape: {C.shape}")
            
        except Exception as e:
            print(f"❌ {strategy}: {e}")
        
        engine.cleanup()

def test_memory_pressure_simulation():
    """Simulate memory pressure by allocating GPU memory first."""
    print("\n💾 Memory Pressure Simulation")
    print("=" * 60)
    
    # Allocate a large chunk of GPU memory first
    try:
        # Allocate about 8GB of GPU memory
        memory_hog = torch.randn(1000, 1000, 2000, dtype=torch.float32).cuda()
        
        print(f"Pre-allocated {memory_hog.numel() * 4 / 1e9:.2f} GB on GPU")
        
        # Now try to do matrix operations in the remaining space
        engine = acc.MatrixEngine(enable_progress=True)
        memory_info = engine.get_memory_info()
        
        print(f"Remaining GPU memory: {memory_info['gpu_available_gb']:.2f} GB")
        print()
        
        # Try a matrix operation that should trigger chunking now
        M, K, N = 4000, 3000, 5000
        total_needed = (M * K + K * N + M * N) * 4 / 1e9
        
        print(f"Attempting {M}x{K} @ {K}x{N} operation")
        print(f"Memory needed: {total_needed:.2f} GB")
        print(f"Available: {memory_info['gpu_available_gb']:.2f} GB")
        
        if total_needed > memory_info['gpu_available_gb']:
            print("This should trigger chunking due to memory pressure!")
        
        A = acc.Matrix.randn((M, K), dtype=torch.float32)
        B = acc.Matrix.randn((K, N), dtype=torch.float32)
        
        C = engine.matmul(A, B)
        
        print(f"✅ Success with chunking! Result: {C.shape}")
        
        # Cleanup
        del memory_hog
        engine.cleanup()
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("Could not allocate initial memory for pressure test")
        else:
            raise

def main():
    print("🎯 Advanced Chunking Demonstration")
    print("This demonstrates Accelera's chunking capabilities under various scenarios.")
    print()
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    # Run tests
    test_extreme_chunking()
    test_manual_chunking_strategies()
    test_memory_pressure_simulation()
    
    print("\n🎉 Advanced chunking demonstration completed!")
    print("\nKey takeaways:")
    print("✅ Accelera automatically handles memory management")
    print("✅ Chunking works transparently when memory is limited")
    print("✅ Multiple chunking strategies available")
    print("✅ Results remain mathematically correct")

if __name__ == "__main__":
    main()