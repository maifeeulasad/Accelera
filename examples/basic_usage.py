"""
Basic usage example of Accelera framework.

Demonstrates how to perform large matrix operations without running into OOM errors.
"""

import torch
import numpy as np
import accelera as acc

def main():
    print("🚀 Accelera Framework - Basic Example")
    print("=" * 50)
    
    # Initialize the matrix engine with automatic memory detection
    engine = acc.MatrixEngine(auto_detect_memory=True, enable_progress=True)
    
    # Display initial memory information
    memory_info = engine.get_memory_info()
    print(f"GPU Memory Available: {memory_info['gpu_available_gb']:.2f} GB")
    print(f"CPU Memory Available: {memory_info['cpu_available_gb']:.2f} GB")
    print()
    
    # Create large matrices that might cause OOM on small GPUs
    print("Creating large matrices...")
    
    # Adjust these sizes based on your GPU memory
    # For demonstration, using moderate sizes that should work on most GPUs
    M, K, N = 5000, 4000, 6000
    
    print(f"Matrix A: {M} x {K}")
    print(f"Matrix B: {K} x {N}")
    print(f"Expected result: {M} x {N}")
    
    # Create matrices using Accelera's Matrix class
    A = acc.Matrix.randn((M, K), dtype=torch.float32)
    B = acc.Matrix.randn((K, N), dtype=torch.float32)
    
    print(f"Matrix A memory: {A.memory_size() / 1e9:.2f} GB")
    print(f"Matrix B memory: {B.memory_size() / 1e9:.2f} GB")
    print()
    
    # Perform matrix multiplication
    print("Performing matrix multiplication A @ B...")
    print("This will automatically chunk the operation if needed.")
    
    try:
        # This operation will automatically manage memory and chunk if necessary
        C = engine.matmul(A, B)
        
        print(f"✅ Success! Result shape: {C.shape}")
        print(f"Result memory: {C.memory_size() / 1e9:.2f} GB")
        
        # Verify the result with a small sample
        print("\nVerifying result with direct computation on small sample...")
        
        # Take small samples for verification
        sample_size = 100
        A_sample = A[:sample_size, :sample_size]
        B_sample = B[:sample_size, :sample_size]
        
        # Direct computation
        C_direct = A_sample @ B_sample
        C_engine = engine.matmul(A_sample, B_sample)
        
        # Check if results are close
        diff = torch.abs(C_direct.tensor() - C_engine.tensor()).max()
        print(f"Maximum difference: {diff:.2e}")
        
        if diff < 2e-5:  # Slightly higher tolerance for numerical differences
            print("✅ Verification passed!")
        else:
            print("❌ Verification failed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Demonstrate element-wise operations
    print("\n" + "=" * 50)
    print("Element-wise Operations Example")
    print("=" * 50)
    
    # Create matrices for element-wise operations
    print("Creating matrices for element-wise operations...")
    X = acc.Matrix.randn((3000, 4000), dtype=torch.float32)
    Y = acc.Matrix.randn((3000, 4000), dtype=torch.float32)
    
    print(f"Matrix X: {X.shape}")
    print(f"Matrix Y: {Y.shape}")
    
    # Addition
    print("\nPerforming element-wise addition...")
    Z_add = engine.add(X, Y)
    print(f"✅ Addition result shape: {Z_add.shape}")
    
    # Multiplication
    print("Performing element-wise multiplication...")
    Z_mul = engine.multiply(X, Y)
    print(f"✅ Multiplication result shape: {Z_mul.shape}")
    
    # Final memory cleanup
    print("\nCleaning up GPU memory...")
    engine.cleanup()
    
    final_memory = engine.get_memory_info()
    print(f"Final GPU utilization: {final_memory['gpu_utilization']:.1f}%")
    
    print("\n🎉 Example completed successfully!")

if __name__ == "__main__":
    main()