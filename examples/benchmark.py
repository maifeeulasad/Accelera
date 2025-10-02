"""
Benchmark script for Accelera framework.

Tests performance and memory efficiency compared to standard PyTorch operations.
"""

import time
import torch
import numpy as np
import argparse
import sys
import accelera as acc
from typing import List, Tuple, Dict


def format_memory_size(bytes_size: float) -> str:
    """Format memory size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def benchmark_matrix_sizes() -> List[Tuple[int, int, int]]:
    """Get list of matrix sizes to benchmark."""
    return [
        (1000, 800, 1200),     # Small - should fit in most GPUs
        (2000, 1500, 2500),    # Medium
        (4000, 3000, 5000),    # Large
        (6000, 4500, 7500),    # Very large
        (8000, 6000, 10000),   # Huge - likely to need chunking
    ]


def run_pytorch_benchmark(A: torch.Tensor, B: torch.Tensor, num_runs: int = 3) -> Dict:
    """Benchmark standard PyTorch matrix multiplication."""
    if not torch.cuda.is_available():
        return {'success': False, 'error': 'CUDA not available'}
    
    try:
        # Move to GPU
        A_gpu = A.cuda()
        B_gpu = B.cuda()
        
        # Warmup
        _ = torch.matmul(A_gpu, B_gpu)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            
            result = torch.matmul(A_gpu, B_gpu)
            
            torch.cuda.synchronize()
            end = time.time()
            
            times.append(end - start)
        
        # Memory usage
        peak_memory = torch.cuda.max_memory_allocated()
        
        # Cleanup
        del A_gpu, B_gpu, result
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        return {
            'success': True,
            'times': times,
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'peak_memory': peak_memory
        }
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            return {'success': False, 'error': 'Out of memory'}
        else:
            raise


def run_accelera_benchmark(A: torch.Tensor, B: torch.Tensor, num_runs: int = 3) -> Dict:
    """Benchmark Accelera matrix multiplication."""
    try:
        # Initialize engine
        engine = acc.MatrixEngine(enable_progress=False)
        
        # Convert to Accelera matrices
        A_acc = acc.Matrix(A)
        B_acc = acc.Matrix(B)
        
        # Warmup
        _ = engine.matmul(A_acc, B_acc)
        engine.cleanup()
        
        # Benchmark
        times = []
        peak_memory = 0
        
        for _ in range(num_runs):
            torch.cuda.reset_peak_memory_stats()
            
            start = time.time()
            result = engine.matmul(A_acc, B_acc)
            end = time.time()
            
            times.append(end - start)
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())
            
            # Cleanup between runs
            engine.cleanup()
        
        return {
            'success': True,
            'times': times,
            'avg_time': sum(times) / len(times),
            'min_time': min(times),
            'peak_memory': peak_memory
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}


def run_benchmark_suite(matrix_sizes: List[Tuple[int, int, int]], 
                       num_runs: int = 3,
                       verify_results: bool = True) -> Dict:
    """Run complete benchmark suite."""
    
    print("🚀 Accelera Framework Benchmark Suite")
    print("=" * 60)
    
    # System info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {format_memory_size(gpu_memory)}")
    else:
        print("❌ CUDA not available")
        return {}
    
    print(f"Benchmark runs per test: {num_runs}")
    print(f"Verify results: {verify_results}")
    print()
    
    results = {}
    
    for i, (M, K, N) in enumerate(matrix_sizes):
        print(f"Test {i+1}/{len(matrix_sizes)}: {M}x{K} @ {K}x{N}")
        print("-" * 40)
        
        # Calculate memory requirements
        A_size = M * K * 4  # float32
        B_size = K * N * 4
        C_size = M * N * 4
        total_size = A_size + B_size + C_size
        
        print(f"Memory requirement: {format_memory_size(total_size)}")
        
        # Create test matrices
        A = torch.randn(M, K, dtype=torch.float32)
        B = torch.randn(K, N, dtype=torch.float32)
        
        # Run PyTorch benchmark
        print("Testing PyTorch...")
        pytorch_result = run_pytorch_benchmark(A, B, num_runs)
        
        # Run Accelera benchmark
        print("Testing Accelera...")
        accelera_result = run_accelera_benchmark(A, B, num_runs)
        
        # Store results
        test_key = f"{M}x{K}x{N}"
        results[test_key] = {
            'matrix_size': (M, K, N),
            'memory_requirement': total_size,
            'pytorch': pytorch_result,
            'accelera': accelera_result
        }
        
        # Print results
        print("\nResults:")
        
        if pytorch_result['success']:
            print(f"  PyTorch:  {pytorch_result['avg_time']:.3f}s "
                  f"(peak memory: {format_memory_size(pytorch_result['peak_memory'])})")
        else:
            print(f"  PyTorch:  ❌ {pytorch_result['error']}")
        
        if accelera_result['success']:
            print(f"  Accelera: {accelera_result['avg_time']:.3f}s "
                  f"(peak memory: {format_memory_size(accelera_result['peak_memory'])})")
        else:
            print(f"  Accelera: ❌ {accelera_result['error']}")
        
        # Performance comparison
        if pytorch_result['success'] and accelera_result['success']:
            speedup = pytorch_result['avg_time'] / accelera_result['avg_time']
            memory_ratio = accelera_result['peak_memory'] / pytorch_result['peak_memory']
            
            if speedup > 1:
                print(f"  🚀 Accelera is {speedup:.2f}x faster")
            elif speedup < 0.8:
                print(f"  ⚠️ Accelera is {1/speedup:.2f}x slower")
            else:
                print(f"  ≈ Similar performance ({speedup:.2f}x)")
            
            print(f"  💾 Memory usage: {memory_ratio:.2f}x of PyTorch")
        
        # Result verification
        if verify_results and pytorch_result['success'] and accelera_result['success']:
            print("Verifying result correctness...")
            try:
                # Small sample verification
                sample_size = min(100, M, N)
                A_sample = A[:sample_size, :sample_size]
                B_sample = B[:sample_size, :sample_size]
                
                # PyTorch result
                pytorch_sample = torch.matmul(A_sample.cuda(), B_sample.cuda()).cpu()
                
                # Accelera result
                engine = acc.MatrixEngine(enable_progress=False)
                accelera_sample = engine.matmul(A_sample, B_sample).tensor()
                
                # Compare
                diff = torch.abs(pytorch_sample - accelera_sample).max()
                if diff < 1e-4:
                    print("  ✅ Results verified")
                else:
                    print(f"  ❌ Results differ (max diff: {diff:.2e})")
                
                engine.cleanup()
                
            except Exception as e:
                print(f"  ⚠️ Verification failed: {e}")
        
        print()
        
        # Cleanup
        del A, B
        torch.cuda.empty_cache()
    
    return results


def print_summary(results: Dict):
    """Print benchmark summary."""
    print("=" * 60)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 60)
    
    # Count successes
    pytorch_successes = sum(1 for r in results.values() if r['pytorch']['success'])
    accelera_successes = sum(1 for r in results.values() if r['accelera']['success'])
    total_tests = len(results)
    
    print(f"Tests completed: {total_tests}")
    print(f"PyTorch successes: {pytorch_successes}/{total_tests}")
    print(f"Accelera successes: {accelera_successes}/{total_tests}")
    
    # Performance stats
    if accelera_successes > 0:
        successful_tests = [
            r for r in results.values() 
            if r['pytorch']['success'] and r['accelera']['success']
        ]
        
        if successful_tests:
            speedups = [
                r['pytorch']['avg_time'] / r['accelera']['avg_time'] 
                for r in successful_tests
            ]
            
            avg_speedup = sum(speedups) / len(speedups)
            
            print(f"\nPerformance comparison (where both succeeded):")
            print(f"  Average speedup: {avg_speedup:.2f}x")
            print(f"  Best speedup: {max(speedups):.2f}x")
            print(f"  Worst speedup: {min(speedups):.2f}x")
    
    # Memory advantage
    memory_advantages = []
    for r in results.values():
        if not r['pytorch']['success'] and r['accelera']['success']:
            memory_advantages.append(r['memory_requirement'])
    
    if memory_advantages:
        print(f"\nMemory advantages (Accelera succeeded where PyTorch failed):")
        for size in memory_advantages:
            print(f"  - {format_memory_size(size)} operation")
    
    print()


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description='Benchmark Accelera framework')
    parser.add_argument('--runs', type=int, default=3, 
                       help='Number of runs per test (default: 3)')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip result verification')
    parser.add_argument('--custom-size', nargs=3, type=int, metavar=('M', 'K', 'N'),
                       help='Test custom matrix size MxK @ KxN')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. This benchmark requires a CUDA-capable GPU.")
        sys.exit(1)
    
    # Determine test sizes
    if args.custom_size:
        matrix_sizes = [tuple(args.custom_size)]
        print(f"Running custom benchmark: {args.custom_size[0]}x{args.custom_size[1]} @ {args.custom_size[1]}x{args.custom_size[2]}")
    else:
        matrix_sizes = benchmark_matrix_sizes()
    
    # Run benchmark suite
    results = run_benchmark_suite(
        matrix_sizes=matrix_sizes,
        num_runs=args.runs,
        verify_results=not args.no_verify
    )
    
    # Print summary
    if results:
        print_summary(results)
    
    print("🎉 Benchmark completed!")


if __name__ == "__main__":
    main()