"""
Memory-Efficient Matrix Engine with Hybrid CPU-GPU Strategy

This engine provides hybrid CPU-GPU computation and CPU fallback for very low 
memory thresholds to prevent excessive chunking that causes memory leaks.
"""

import torch
import threading
import time
import logging
from typing import Optional, Union

from .engine import MatrixEngine, Matrix
from .hybrid_matmul import HybridMatMul

logger = logging.getLogger(__name__)


class MemoryEfficientEngine(MatrixEngine):
    """
    Matrix engine with hybrid CPU-GPU strategy and intelligent fallback.
    
    This engine prevents memory leaks by using:
    1. Hybrid tiled matmul for extremely large operations
    2. CPU fallback for very low memory thresholds
    3. GPU-first storage to avoid RAM exhaustion
    """
    
    def __init__(self, 
                 memory_threshold_gb: float = 1.0,
                 enable_progress: bool = False,
                 chunk_size: Optional[int] = None,
                 fallback_strategy: str = "cpu",
                 prefer_gpu_storage: bool = False,
                 use_hybrid_matmul: bool = True):
        """
        Initialize memory-efficient engine.
        
        Args:
            memory_threshold_gb: Memory threshold for operations  
            enable_progress: Whether to show progress bars
            chunk_size: Optional chunk size override
            fallback_strategy: Strategy for low memory ('cpu', 'default_chunk', or 'hybrid')
            prefer_gpu_storage: Keep results on GPU when possible
            use_hybrid_matmul: Use hybrid tiled matmul for very large operations
        """
        super().__init__(
            memory_threshold_gb=memory_threshold_gb,
            enable_progress=enable_progress,
            chunk_size=chunk_size
        )
        self.fallback_strategy = fallback_strategy
        self.prefer_gpu_storage = prefer_gpu_storage
        self.use_hybrid_matmul = use_hybrid_matmul
        
        # Initialize hybrid matmul engine if enabled
        if self.use_hybrid_matmul:
            self.hybrid_engine = HybridMatMul(
                gpu_memory_limit_gb=memory_threshold_gb * 2,  # Give it more headroom
                enable_progress=enable_progress
            )
            logger.info("Hybrid matmul engine initialized")
        else:
            self.hybrid_engine = None
        
    def _should_use_fallback(self, total_memory_gb: float) -> bool:
        """
        Determine if we should use fallback strategy for very low memory thresholds.
        
        Args:
            total_memory_gb: Total memory required for operation
            
        Returns:
            True if should use fallback to avoid excessive chunking
        """
        # Use fallback for very low thresholds to avoid chunking overhead
        return self.memory_threshold_gb < 0.05  # Less than 50MB threshold
        
    def _use_default_chunking(self, a: torch.Tensor, b: torch.Tensor) -> Matrix:
        """
        Use default chunking strategy instead of memory-based chunking.
        Keeps matrices intact and uses simple row-based chunking.
        """
        logger.info(f"[MEMORY_EFFICIENT_ENGINE] Using default chunking fallback (threshold: {self.memory_threshold_gb:.3f}GB)")
        
        # Simple default chunking - process in reasonable chunks regardless of memory
        default_chunk_size = 1000  # Process 1000 rows at a time
        
        a_rows = a.shape[0]
        result_chunks = []
        
        logger.info(f"[MEMORY_EFFICIENT_ENGINE] Processing {a_rows} rows in chunks of {default_chunk_size}")
        
        for start_row in range(0, a_rows, default_chunk_size):
            end_row = min(start_row + default_chunk_size, a_rows)
            
            # Extract chunk from matrix A
            a_chunk = a[start_row:end_row]
            
            # Compute chunk result using ORIGINAL PyTorch function to avoid recursion
            if hasattr(torch, '_original_matmul'):
                chunk_result = torch._original_matmul(a_chunk, b)
            else:
                # Fallback if original function not stored yet (shouldn't happen in normal usage)
                import torch.nn.functional as F
                chunk_result = F.linear(a_chunk, b.T).T if b.dim() == 2 else torch.matmul(a_chunk, b)
            result_chunks.append(chunk_result)
            
            # Clean up intermediate tensors
            del a_chunk, chunk_result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Concatenate all chunks
        result = torch.cat(result_chunks, dim=0)
        
        # Clean up chunk list
        del result_chunks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"[MEMORY_EFFICIENT_ENGINE] Default chunking completed, result shape: {result.shape}")
        return Matrix(result)
        
    def matmul(self, a_matrix: Matrix, b_matrix: Matrix) -> Matrix:
        """
        Override parent matmul with hybrid strategy and intelligent fallback.
        Chooses best approach based on operation size and memory constraints.
        """
        import torch
        
        # Get tensor data from Matrix objects
        a = a_matrix._tensor
        b = b_matrix._tensor
        
        logger.info(f"[MEMORY_EFFICIENT_ENGINE] Computing matrix multiplication: {a.shape} @ {b.shape}")
        
        # Calculate memory requirements
        bytes_per_element = 4  # float32
        a_memory_gb = (a.numel() * bytes_per_element) / (1024**3)
        b_memory_gb = (b.numel() * bytes_per_element) / (1024**3)
        result_memory_gb = (a.shape[0] * b.shape[1] * bytes_per_element) / (1024**3)
        total_memory_gb = a_memory_gb + b_memory_gb + result_memory_gb
        
        logger.info(f"[MEMORY_EFFICIENT_ENGINE] Memory required: A={a_memory_gb:.3f}GB, B={b_memory_gb:.3f}GB, "
                   f"Result={result_memory_gb:.3f}GB, Total={total_memory_gb:.3f}GB")
        
        # Decision tree for choosing computation strategy
        
        # 1. For extremely large operations, use hybrid tiled matmul
        if self.use_hybrid_matmul and total_memory_gb > self.memory_threshold_gb * 3:
            logger.info(f"[MEMORY_EFFICIENT_ENGINE] Using hybrid tiled matmul for very large operation")
            result_tensor = self.hybrid_engine.matmul_large(a, b)
            return Matrix(result_tensor)
        
        # 2. For very low thresholds, use fallback strategy
        if self._should_use_fallback(total_memory_gb):
            
            if self.fallback_strategy == "hybrid" and self.hybrid_engine:
                logger.info(f"[MEMORY_EFFICIENT_ENGINE] Using hybrid matmul fallback")
                result_tensor = self.hybrid_engine.matmul_large(a, b)
                return Matrix(result_tensor)
                
            elif self.fallback_strategy == "cpu":
                logger.info(f"[MEMORY_EFFICIENT_ENGINE] Using CPU fallback for very low threshold ({self.memory_threshold_gb:.3f}GB)")
                
                # The Matrix class seems to move tensors to CPU by default
                # Let's assume we want to return results on GPU (cuda:0) since this is for GPU acceleration
                target_device = torch.device('cuda:0')
                
                # Move tensors to CPU for computation
                a_cpu = a.cpu()
                b_cpu = b.cpu()
                
                # Perform computation on CPU using ORIGINAL PyTorch function to avoid recursion
                if hasattr(torch, '_original_matmul'):
                    result_cpu = torch._original_matmul(a_cpu, b_cpu)
                else:
                    # Fallback if original function not stored yet (shouldn't happen in normal usage)
                    import torch.nn.functional as F
                    result_cpu = F.linear(a_cpu, b_cpu.T).T if b_cpu.dim() == 2 else torch.matmul(a_cpu, b_cpu)
                
                # Move result back to target device (GPU)
                result = result_cpu.to(target_device) 
                logger.info(f"[MEMORY_EFFICIENT_ENGINE] CPU computation completed, result moved to {result.device}")
                
                # Clean up CPU tensors
                del a_cpu, b_cpu, result_cpu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Prepare result matrix using Matrix class
                result_matrix = Matrix(result)
                
                return result_matrix
                
            elif self.fallback_strategy == "default_chunk":
                # Use default chunking strategy
                return self._use_default_chunking(a, b)
        
        # For higher thresholds, use normal parent logic
        logger.info(f"[MEMORY_EFFICIENT_ENGINE] Using normal processing (threshold: {self.memory_threshold_gb:.3f}GB)")
        return super().matmul(a_matrix, b_matrix)
        
    def cleanup(self):
        """Enhanced cleanup with aggressive memory management."""
        try:
            # Aggressive GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Force garbage collection
                import gc
                gc.collect()
                
        except Exception as e:
            logger.warning(f"Enhanced cleanup failed: {e}")
            
        # Call parent cleanup
        try:
            super().cleanup()
        except Exception as e:
            logger.warning(f"Parent cleanup failed: {e}")