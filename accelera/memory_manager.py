"""
Memory Management Module

Handles GPU memory monitoring, allocation, and CPU-GPU transfers.
"""

import torch
import psutil
import gc
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages memory allocation and transfers between CPU and GPU.
    
    Provides utilities for:
    - GPU memory monitoring
    - Automatic memory cleanup
    - CPU-GPU tensor transfers
    - Memory usage optimization
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize memory manager.
        
        Args:
            device: CUDA device to use (e.g., 'cuda:0'). Auto-detects if None.
        """
        self.device = self._setup_device(device)
        self.total_gpu_memory = torch.cuda.get_device_properties(self.device).total_memory
        self.memory_threshold = 0.9  # Use 90% of available memory
        
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """Setup and validate CUDA device."""
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
                raise RuntimeError("CUDA is not available. This framework requires NVIDIA GPU.")
        
        device = torch.device(device)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        # Get total memory for this device
        total_memory = torch.cuda.get_device_properties(device).total_memory
        
        logger.info(f"Using device: {device}")
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(f"Total GPU memory: {total_memory / 1e9:.2f} GB")
        
        return device
    
    def get_available_memory(self) -> int:
        """Get available GPU memory in bytes."""
        torch.cuda.empty_cache()
        return self.total_gpu_memory - torch.cuda.memory_allocated(self.device)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get comprehensive memory usage statistics."""
        gpu_allocated = torch.cuda.memory_allocated(self.device)
        gpu_cached = torch.cuda.memory_reserved(self.device)
        gpu_available = self.get_available_memory()
        
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        
        return {
            'gpu_allocated_gb': gpu_allocated / 1e9,
            'gpu_cached_gb': gpu_cached / 1e9,
            'gpu_available_gb': gpu_available / 1e9,
            'gpu_total_gb': self.total_gpu_memory / 1e9,
            'gpu_utilization': (gpu_allocated / self.total_gpu_memory) * 100,
            'cpu_available_gb': cpu_memory.available / 1e9,
            'cpu_total_gb': cpu_memory.total / 1e9,
            'cpu_utilization': cpu_memory.percent
        }
    
    def can_allocate(self, size_bytes: int) -> bool:
        """Check if we can allocate the given amount of GPU memory."""
        available = self.get_available_memory()
        return available >= size_bytes * 1.1  # 10% safety margin
    
    def estimate_tensor_size(self, shape: tuple, dtype: torch.dtype = torch.float32) -> int:
        """Estimate memory size of a tensor in bytes."""
        element_size = torch.tensor(0, dtype=dtype).element_size()
        total_elements = 1
        for dim in shape:
            total_elements *= dim
        return total_elements * element_size
    
    def move_to_gpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to GPU with memory validation."""
        if tensor.device == self.device:
            return tensor
            
        size_bytes = tensor.numel() * tensor.element_size()
        
        if not self.can_allocate(size_bytes):
            raise RuntimeError(f"Cannot allocate {size_bytes / 1e9:.2f} GB on GPU. "
                             f"Available: {self.get_available_memory() / 1e9:.2f} GB")
        
        try:
            # Create GPU tensor and immediately clean up CPU reference if possible
            gpu_tensor = tensor.to(self.device)
            return gpu_tensor
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Force cleanup and retry once
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                try:
                    gpu_tensor = tensor.to(self.device)
                    return gpu_tensor
                except RuntimeError:
                    # Still failed - propagate the OOM error
                    raise
            else:
                raise
    
    def move_to_cpu(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to CPU and free GPU memory."""
        cpu_tensor = tensor.cpu()
        if tensor.device != torch.device('cpu'):
            del tensor
            torch.cuda.empty_cache()
        return cpu_tensor
    
    def cleanup(self):
        """Force aggressive cleanup of GPU memory."""
        # Force Python garbage collection
        gc.collect()
        
        # Clear PyTorch GPU cache
        torch.cuda.empty_cache()
        
        # Force GPU synchronization to ensure operations complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Check for severe memory pressure (but avoid recursion)
        try:
            available_memory = torch.cuda.get_device_properties(self.device).total_memory - torch.cuda.memory_allocated(self.device)
            if available_memory < self.total_gpu_memory * 0.1:  # Less than 10% available
                logger.warning("Severe memory pressure detected, performing aggressive cleanup")
                # Multiple rounds of cache clearing (avoid calling get_available_memory again)
                for _ in range(3):
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        except RuntimeError:
            # If we can't check memory, just do basic cleanup
            pass
        
    def get_optimal_chunk_size(self, matrix_shape: tuple, operation: str = 'matmul') -> int:
        """
        Calculate optimal chunk size based on available memory.
        
        Args:
            matrix_shape: Shape of the matrix to be processed
            operation: Type of operation ('matmul', 'add', etc.)
            
        Returns:
            Optimal chunk size for the first dimension
        """
        available_memory = self.get_available_memory() * self.memory_threshold
        
        if operation == 'matmul':
            # For matrix multiplication A @ B, we need memory for:
            # - Input chunk of A
            # - Full matrix B (kept on GPU)
            # - Output chunk
            
            rows, cols = matrix_shape[0], matrix_shape[1]
            element_size = 4  # float32
            
            # Memory per row of operations
            memory_per_row = cols * element_size  # Input row
            memory_per_row += cols * matrix_shape[1] * element_size  # Output row (worst case)
            
            max_rows = int(available_memory / memory_per_row)
            
            # Ensure we have at least 1 row and don't exceed matrix size
            chunk_size = max(1, min(max_rows, rows))
            
        else:
            # For element-wise operations
            element_size = 4  # float32
            total_elements_per_row = 1
            for dim in matrix_shape[1:]:
                total_elements_per_row *= dim
                
            memory_per_row = total_elements_per_row * element_size * 3  # Input + output + temp
            max_rows = int(available_memory / memory_per_row)
            chunk_size = max(1, min(max_rows, matrix_shape[0]))
        
        logger.debug(f"Calculated optimal chunk size: {chunk_size} for operation: {operation}")
        return chunk_size