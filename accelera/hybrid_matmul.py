"""
Hybrid GPU-CPU Matrix Multiplication Engine

PyTorch-native implementation of hybrid matmul using tiling and streaming
for extremely large tensors. No additional dependencies required.
"""

import torch
import math
import logging
from typing import Optional, Tuple, List
import threading
from queue import Queue
import time

logger = logging.getLogger(__name__)


class TileScheduler:
    """Manages tiling and scheduling of large tensor operations."""
    
    def __init__(self, A_shape: Tuple, B_shape: Tuple, gpu_memory_limit: int):
        """
        Initialize tile scheduler.
        
        Args:
            A_shape: Shape of matrix A
            B_shape: Shape of matrix B
            gpu_memory_limit: GPU memory limit in bytes
        """
        self.A_shape = A_shape
        self.B_shape = B_shape
        self.gpu_memory_limit = gpu_memory_limit
        
        # Determine output shape
        self.output_shape = self._compute_output_shape(A_shape, B_shape)
        self._compute_optimal_tiling()
    
    def _compute_output_shape(self, A_shape: Tuple, B_shape: Tuple) -> Tuple:
        """Compute output shape for matrix multiplication."""
        if len(A_shape) == 2 and len(B_shape) == 2:
            return (A_shape[0], B_shape[1])
        
        # For batched operations, handle batch dimensions
        batch_dims_A = A_shape[:-2]
        batch_dims_B = B_shape[:-2]
        
        if not batch_dims_A and not batch_dims_B:
            return (A_shape[-2], B_shape[-1])
        
        # Compute broadcasted batch dimensions
        max_ndim = max(len(batch_dims_A), len(batch_dims_B))
        padded_A = (1,) * (max_ndim - len(batch_dims_A)) + batch_dims_A
        padded_B = (1,) * (max_ndim - len(batch_dims_B)) + batch_dims_B
        
        broadcast_dims = []
        for dim_A, dim_B in zip(padded_A, padded_B):
            if dim_A == dim_B:
                broadcast_dims.append(dim_A)
            elif dim_A == 1:
                broadcast_dims.append(dim_B)
            elif dim_B == 1:
                broadcast_dims.append(dim_A)
            else:
                raise ValueError(f"Incompatible batch dimensions: {dim_A} and {dim_B}")
        
        return tuple(broadcast_dims) + (A_shape[-2], B_shape[-1])
    
    def _compute_optimal_tiling(self):
        """Compute optimal tile sizes based on available GPU memory."""
        element_size = 4  # float32
        
        # Start with reasonable tile sizes
        M = self.output_shape[-2]
        N = self.output_shape[-1]
        K = self.A_shape[-1]
        
        self.tile_M = min(512, M)
        self.tile_N = min(512, N)
        self.tile_K = min(512, K)
        
        # Adjust based on memory constraints
        while True:
            # Memory for A tile: tile_M * tile_K
            # Memory for B tile: tile_K * tile_N
            # Memory for C tile: tile_M * tile_N
            total_memory = (
                self.tile_M * self.tile_K +
                self.tile_K * self.tile_N +
                self.tile_M * self.tile_N
            ) * element_size
            
            # Use only 40% of GPU memory limit for tiles
            if total_memory <= self.gpu_memory_limit * 0.4:
                break
            
            # Reduce largest dimension
            if self.tile_M >= self.tile_N and self.tile_M >= self.tile_K:
                self.tile_M = max(64, self.tile_M // 2)
            elif self.tile_N >= self.tile_M and self.tile_N >= self.tile_K:
                self.tile_N = max(64, self.tile_N // 2)
            else:
                self.tile_K = max(64, self.tile_K // 2)
            
            # Minimum tile size
            if self.tile_M == 64 and self.tile_N == 64 and self.tile_K == 64:
                break
        
        logger.info(f"Optimal tile sizes: M={self.tile_M}, N={self.tile_N}, K={self.tile_K}")
    
    def generate_tile_operations(self) -> List[Tuple]:
        """Generate list of tile operations to perform."""
        operations = []
        M = self.output_shape[-2]
        N = self.output_shape[-1]
        K = self.A_shape[-1]
        
        # For batched operations
        batch_dims = self.output_shape[:-2]
        if batch_dims:
            import itertools
            batch_indices = list(itertools.product(*[range(d) for d in batch_dims]))
        else:
            batch_indices = [()]
        
        for batch_idx in batch_indices:
            for i in range(0, M, self.tile_M):
                for j in range(0, N, self.tile_N):
                    # For each output tile, we need to accumulate across K dimension
                    operations.append((
                        batch_idx,
                        (i, min(i + self.tile_M, M)),
                        (j, min(j + self.tile_N, N))
                    ))
        
        logger.info(f"Generated {len(operations)} tile operations")
        return operations


class HybridMatMul:
    """
    Hybrid GPU-CPU matrix multiplication for extremely large tensors.
    Uses PyTorch native operations with tiling and streaming.
    """
    
    def __init__(self, 
                 gpu_memory_limit_gb: float = 2.0,
                 num_threads: int = 4,
                 enable_progress: bool = False):
        """
        Initialize hybrid matmul engine.
        
        Args:
            gpu_memory_limit_gb: Maximum GPU memory to use in GB
            num_threads: Number of threads for parallel tile processing
            enable_progress: Show progress during computation
        """
        self.gpu_memory_limit = int(gpu_memory_limit_gb * 1024**3)
        self.num_threads = num_threads
        self.enable_progress = enable_progress
        
        # Memory tracking
        self.gpu_memory_used = 0
        self.lock = threading.Lock()
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initialized HybridMatMul with {gpu_memory_limit_gb}GB GPU memory limit")
    
    def _estimate_tensor_memory(self, shape: Tuple) -> int:
        """Estimate memory required for a tensor in bytes."""
        element_size = 4  # float32
        return math.prod(shape) * element_size
    
    def _allocate_gpu_memory(self, size: int) -> bool:
        """Check if we can allocate GPU memory (thread-safe)."""
        with self.lock:
            if self.gpu_memory_used + size <= self.gpu_memory_limit:
                self.gpu_memory_used += size
                return True
            return False
    
    def _release_gpu_memory(self, size: int):
        """Release allocated GPU memory (thread-safe)."""
        with self.lock:
            self.gpu_memory_used = max(0, self.gpu_memory_used - size)
    
    def matmul_large(self,
                     A: torch.Tensor,
                     B: torch.Tensor,
                     output_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform matrix multiplication on large tensors using tiling.
        
        Args:
            A: Input tensor A
            B: Input tensor B
            output_tensor: Pre-allocated output tensor (optional)
            
        Returns:
            Result of matrix multiplication A @ B
        """
        # Ensure float32
        A = A.float() if A.dtype != torch.float32 else A
        B = B.float() if B.dtype != torch.float32 else B
        
        # Initialize tile scheduler
        scheduler = TileScheduler(A.shape, B.shape, self.gpu_memory_limit)
        
        # Allocate output tensor if not provided
        if output_tensor is None:
            # Try to allocate on GPU first, fall back to CPU if needed
            try:
                output_tensor = torch.zeros(scheduler.output_shape, dtype=torch.float32, device=self.device)
            except RuntimeError:
                logger.warning("Cannot allocate output on GPU, using CPU")
                output_tensor = torch.zeros(scheduler.output_shape, dtype=torch.float32, device='cpu')
        
        # Generate tile operations
        operations = scheduler.generate_tile_operations()
        
        # Process tiles
        self._process_tiles(A, B, output_tensor, operations, scheduler)
        
        return output_tensor
    
    def _process_tiles(self,
                      A: torch.Tensor,
                      B: torch.Tensor,
                      C: torch.Tensor,
                      operations: List[Tuple],
                      scheduler: TileScheduler):
        """Process tiles with memory management."""
        total_operations = len(operations)
        completed = 0
        
        for batch_idx, (m_start, m_end), (n_start, n_end) in operations:
            self._process_single_tile(A, B, C, batch_idx, m_start, m_end, n_start, n_end, scheduler)
            
            completed += 1
            if self.enable_progress and completed % 10 == 0:
                progress = (completed / total_operations) * 100
                logger.info(f"Progress: {progress:.1f}% ({completed}/{total_operations})")
        
        logger.info("All tile operations completed")
    
    def _process_single_tile(self,
                            A: torch.Tensor,
                            B: torch.Tensor,
                            C: torch.Tensor,
                            batch_idx: Tuple,
                            m_start: int,
                            m_end: int,
                            n_start: int,
                            n_end: int,
                            scheduler: TileScheduler):
        """Process a single tile operation."""
        K = A.shape[-1]
        tile_M = m_end - m_start
        tile_N = n_end - n_start
        
        # Accumulate result across K dimension in chunks
        for k_start in range(0, K, scheduler.tile_K):
            k_end = min(k_start + scheduler.tile_K, K)
            tile_K = k_end - k_start
            
            # Calculate memory for this tile
            tile_memory = (tile_M * tile_K + tile_K * tile_N + tile_M * tile_N) * 4
            
            # Extract tiles
            if batch_idx:
                A_tile = A[batch_idx][m_start:m_end, k_start:k_end]
                B_tile = B[batch_idx][k_start:k_end, n_start:n_end]
            else:
                A_tile = A[m_start:m_end, k_start:k_end]
                B_tile = B[k_start:k_end, n_start:n_end]
            
            # Compute on GPU if possible, otherwise CPU
            try:
                if self.device.type == 'cuda' and self._allocate_gpu_memory(tile_memory):
                    try:
                        # Move to GPU and compute
                        A_gpu = A_tile.to(self.device)
                        B_gpu = B_tile.to(self.device)
                        
                        # Use original matmul to avoid recursion
                        if hasattr(torch, '_original_matmul'):
                            C_tile = torch._original_matmul(A_gpu, B_gpu)
                        else:
                            C_tile = torch.matmul(A_gpu, B_gpu)
                        
                        # Move result back
                        C_tile = C_tile.cpu() if C.device.type == 'cpu' else C_tile
                        
                        # Cleanup GPU memory
                        del A_gpu, B_gpu
                        
                    finally:
                        self._release_gpu_memory(tile_memory)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    # CPU computation
                    if hasattr(torch, '_original_matmul'):
                        C_tile = torch._original_matmul(A_tile.cpu(), B_tile.cpu())
                    else:
                        C_tile = torch.matmul(A_tile.cpu(), B_tile.cpu())
                
                # Accumulate to output
                if batch_idx:
                    C[batch_idx][m_start:m_end, n_start:n_end] += C_tile.to(C.device)
                else:
                    C[m_start:m_end, n_start:n_end] += C_tile.to(C.device)
                
            except RuntimeError as e:
                logger.error(f"Error processing tile: {e}")
                raise
    
    def benchmark(self, A: torch.Tensor, B: torch.Tensor) -> dict:
        """
        Benchmark memory usage and performance.
        
        Returns:
            Dictionary with benchmark results
        """
        logger.info("="*60)
        logger.info("HYBRID MATMUL BENCHMARK")
        logger.info("="*60)
        
        # Calculate memory requirements
        input_memory_gb = (A.numel() + B.numel()) * 4 / (1024**3)
        output_shape = (A.shape[0], B.shape[1])
        output_memory_gb = math.prod(output_shape) * 4 / (1024**3)
        total_memory_gb = input_memory_gb + output_memory_gb
        
        logger.info(f"Input tensor A: {A.shape}, Memory: {A.numel() * 4 / (1024**3):.2f} GB")
        logger.info(f"Input tensor B: {B.shape}, Memory: {B.numel() * 4 / (1024**3):.2f} GB")
        logger.info(f"Output tensor: {output_shape}, Memory: {output_memory_gb:.2f} GB")
        logger.info(f"Total memory: {total_memory_gb:.2f} GB")
        logger.info(f"GPU limit: {self.gpu_memory_limit / (1024**3):.2f} GB")
        
        # Perform multiplication with timing
        start_time = time.time()
        result = self.matmul_large(A, B)
        hybrid_time = time.time() - start_time
        
        logger.info(f"Hybrid matmul time: {hybrid_time:.2f}s")
        
        return {
            'time': hybrid_time,
            'input_memory_gb': input_memory_gb,
            'output_memory_gb': output_memory_gb,
            'total_memory_gb': total_memory_gb,
            'result': result
        }
