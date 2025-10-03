"""
Matrix Engine Module

Core engine for memory-efficient matrix operations with automatic chunking.
"""

import torch
import numpy as np
from typing import Optional, Union, Tuple, List
from tqdm import tqdm
import logging

from .memory_manager import MemoryManager
from .chunking import ChunkingStrategy, RowChunking, AdaptiveChunking, ChunkIterator, create_chunking_strategy
from .matrix import Matrix

logger = logging.getLogger(__name__)


class MatrixEngine:
    """
    Core engine for memory-efficient matrix operations.
    
    Automatically manages GPU memory and chunks large operations to prevent OOM errors.
    Provides seamless API for matrix operations that would otherwise fail on memory-constrained GPUs.
    """
    
    def __init__(self, 
                 device: Optional[str] = None,
                 auto_detect_memory: bool = True,
                 chunking_strategy: str = 'adaptive',
                 chunk_size: Optional[int] = None,
                 enable_progress: bool = True,
                 memory_threshold_gb: Optional[float] = None):
        """
        Initialize Matrix Engine.
        
        Args:
            device: CUDA device to use (e.g., 'cuda:0'). Auto-detects if None.
            auto_detect_memory: Whether to automatically detect optimal chunk sizes
            chunking_strategy: Strategy for chunking ('row', 'tile', 'adaptive')
            chunk_size: Fixed chunk size (ignored if auto_detect_memory=True)
            enable_progress: Whether to show progress bars for long operations
            memory_threshold_gb: Force chunking if operation exceeds this threshold (GB)
        """
        # Initialize memory manager
        self.memory_manager = MemoryManager(device)
        self.device = self.memory_manager.device
        
        # Configuration
        self.auto_detect_memory = auto_detect_memory
        self.enable_progress = enable_progress
        self.memory_threshold_gb = memory_threshold_gb
        self.default_chunk_size = chunk_size or 1024
        
        # Initialize chunking strategy
        self.chunking_strategy = self._create_chunking_strategy(chunking_strategy)
        
        logger.info(f"MatrixEngine initialized on {self.device}")
        self._log_memory_info()
    
    def _create_chunking_strategy(self, strategy_type: str) -> ChunkingStrategy:
        """Create chunking strategy based on type."""
        if strategy_type == 'adaptive':
            return AdaptiveChunking(self.memory_manager)
        elif strategy_type == 'row':
            return RowChunking()
        else:
            return create_chunking_strategy(strategy_type, memory_manager=self.memory_manager)
    
    def _log_memory_info(self):
        """Log current memory information."""
        memory_info = self.memory_manager.get_memory_usage()
        logger.info(f"GPU Memory: {memory_info['gpu_available_gb']:.2f}GB available / "
                   f"{memory_info['gpu_total_gb']:.2f}GB total")
        logger.info(f"CPU Memory: {memory_info['cpu_available_gb']:.2f}GB available / "
                   f"{memory_info['cpu_total_gb']:.2f}GB total")
    
    def _get_optimal_chunk_size(self, matrix_shape: Tuple[int, ...], operation: str) -> int:
        """Get optimal chunk size for the given operation."""
        if self.auto_detect_memory:
            return self.memory_manager.get_optimal_chunk_size(matrix_shape, operation)
        else:
            return self.default_chunk_size
    
    def _prepare_matrix(self, matrix: Union[Matrix, torch.Tensor, np.ndarray]) -> Matrix:
        """Convert input to Matrix object if needed."""
        if isinstance(matrix, Matrix):
            return matrix
        elif isinstance(matrix, (torch.Tensor, np.ndarray)):
            return Matrix(matrix)
        else:
            raise TypeError(f"Unsupported matrix type: {type(matrix)}")
    
    def matmul(self, 
               A: Union[Matrix, torch.Tensor, np.ndarray], 
               B: Union[Matrix, torch.Tensor, np.ndarray],
               chunk_strategy: Optional[str] = None) -> Matrix:
        """
        Memory-efficient matrix multiplication A @ B.
        
        Args:
            A: Left matrix (M x K)
            B: Right matrix (K x N)  
            chunk_strategy: Override default chunking strategy
            
        Returns:
            Result matrix (M x N)
        """
        # Convert to Matrix objects
        A = self._prepare_matrix(A)
        B = self._prepare_matrix(B)
        
        # Validate dimensions
        if A.shape[-1] != B.shape[-2]:
            raise ValueError(f"Matrix dimensions incompatible: {A.shape} @ {B.shape}")
        
        logger.info(f"Computing matrix multiplication: {A.shape} @ {B.shape}")
        
        # Calculate memory requirements in bytes
        A_memory = A.memory_size()
        B_memory = B.memory_size()
        result_memory = A.shape[0] * B.shape[1] * 4  # float32 = 4 bytes
        total_memory_needed = A_memory + B_memory + result_memory
        
        logger.info(f"Memory required: A={A_memory/1e9:.3f}GB, B={B_memory/1e9:.3f}GB, "
                   f"Result={result_memory/1e9:.3f}GB, Total={total_memory_needed/1e9:.3f}GB")
        
        # Get current available memory
        available_memory = self.memory_manager.get_available_memory()
        logger.info(f"Available GPU memory: {available_memory/1e9:.3f}GB")
        
        # Force chunking if memory threshold is set and exceeded
        if self.memory_threshold_gb is not None:
            total_memory_gb = total_memory_needed / 1e9
            if total_memory_gb >= self.memory_threshold_gb:
                logger.info(f"Memory threshold ({self.memory_threshold_gb:.3f}GB) exceeded, using chunked computation")
                return self._chunked_matmul(A, B, chunk_strategy)
        
        # Use chunking if we don't have enough memory
        if not self.memory_manager.can_allocate(total_memory_needed):
            logger.info("Insufficient memory, using chunked computation")
            return self._chunked_matmul(A, B, chunk_strategy)
        
        logger.info("Sufficient memory available, computing directly on GPU")
        return self._direct_matmul(A, B)
    
    def _direct_matmul(self, A: Matrix, B: Matrix) -> Matrix:
        """Direct matrix multiplication on GPU."""
        # Move both matrices to GPU
        A_gpu = self.memory_manager.move_to_gpu(A.tensor())
        B_gpu = self.memory_manager.move_to_gpu(B.tensor())
        
        # Compute result using ORIGINAL PyTorch function to avoid recursion
        if hasattr(torch, '_original_matmul'):
            result = torch._original_matmul(A_gpu, B_gpu)
        else:
            # Fallback if original function not stored yet
            result = torch.matmul(A_gpu, B_gpu)
        
        # Move result back to CPU for storage
        result_cpu = self.memory_manager.move_to_cpu(result)
        
        # Cleanup GPU memory
        del A_gpu, B_gpu
        self.memory_manager.cleanup()
        
        return Matrix(result_cpu)
    
    def _chunked_matmul(self, A: Matrix, B: Matrix, chunk_strategy: Optional[str] = None) -> Matrix:
        """Chunked matrix multiplication for large matrices."""
        # Use specified strategy or default
        strategy = self.chunking_strategy
        if chunk_strategy:
            strategy = self._create_chunking_strategy(chunk_strategy)
        
        # Get optimal chunk size
        chunk_size = self._get_optimal_chunk_size(A.shape, 'matmul')
        
        # Keep B on GPU throughout computation (if it fits)
        B_gpu = None
        try:
            if self.memory_manager.can_allocate(B.memory_size()):
                B_gpu = self.memory_manager.move_to_gpu(B.tensor())
                logger.info("Loaded B matrix to GPU for reuse")
        except RuntimeError:
            logger.warning("Could not load B matrix to GPU, will transfer chunks as needed")
        
        # Initialize result matrix (force CPU storage to prevent memory leaks)
        result_shape = (A.shape[0], B.shape[1])
        result = Matrix.zeros(result_shape, dtype=A.dtype, device='cpu')
        
        # Get chunks directly without iterator to prevent memory accumulation
        A_tensor = A.tensor()
        chunks = strategy.get_chunks(A_tensor.shape, chunk_size)
        
        # Progress bar
        pbar = None
        if self.enable_progress:
            pbar = tqdm(total=len(chunks), desc="Matrix multiplication", unit="chunk")
        
        try:
            current_row = 0
            
            for chunk_idx, chunk_slice in enumerate(chunks):
                # Extract chunk directly without iterator
                A_chunk_cpu = A_tensor[chunk_slice]
                
                # Only move to GPU when needed and clean up immediately
                chunk_rows = A_chunk_cpu.shape[0]
                
                # Check available memory before attempting GPU computation
                available_memory_gb = self.memory_manager.get_available_memory() / (1024**3)
                
                if available_memory_gb < 0.3:  # Less than 300MB available - use CPU
                    logger.info(f"Low GPU memory ({available_memory_gb:.2f}GB), computing on CPU")
                    # Use CPU for computation - no GPU transfer needed
                    B_cpu = B.tensor().cpu()
                    if hasattr(torch, '_original_matmul'):
                        chunk_result = torch._original_matmul(A_chunk_cpu, B_cpu)
                    else:
                        chunk_result = torch.matmul(A_chunk_cpu, B_cpu)
                    del B_cpu  # Clean up immediately
                else:
                    # Try GPU computation
                    try:
                        # Move A chunk to GPU just for computation
                        A_chunk = self.memory_manager.move_to_gpu(A_chunk_cpu)
                        
                        # Get corresponding B matrix (either from GPU or move chunk)
                        if B_gpu is not None:
                            B_compute = B_gpu
                        else:
                            # Try to create temporary B_compute, fallback to CPU if OOM
                            try:
                                B_compute = self.memory_manager.move_to_gpu(B.tensor())
                            except RuntimeError as oom_error:
                                if "out of memory" in str(oom_error).lower():
                                    # Can't fit B on GPU, use CPU computation
                                    logger.warning("Cannot fit B matrix on GPU, using CPU computation")
                                    B_cpu = B.tensor().cpu()
                                    if hasattr(torch, '_original_matmul'):
                                        chunk_result = torch._original_matmul(A_chunk_cpu, B_cpu)
                                    else:
                                        chunk_result = torch.matmul(A_chunk_cpu, B_cpu)
                                    del B_cpu
                                    # Set flag to skip GPU computation
                                    B_compute = None
                                else:
                                    raise
                        
                        # Only proceed with GPU computation if B_compute is available
                        if B_compute is not None:
                            # Compute chunk result on GPU using ORIGINAL PyTorch function to avoid recursion
                            if hasattr(torch, '_original_matmul'):
                                chunk_result_gpu = torch._original_matmul(A_chunk, B_compute)
                            else:
                                chunk_result_gpu = torch.matmul(A_chunk, B_compute)
                            
                            # Move result to CPU and clean up GPU tensors immediately
                            chunk_result = self.memory_manager.move_to_cpu(chunk_result_gpu)
                            
                            # Immediate cleanup of all GPU tensors used in this iteration
                            del chunk_result_gpu, A_chunk
                            if B_gpu is None:  # B_compute was created temporarily
                                del B_compute
                            torch.cuda.empty_cache()
                        # If B_compute is None, chunk_result was already computed on CPU above
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            logger.warning("GPU OOM during computation, falling back to CPU")
                            # Clean up any partial GPU allocations
                            torch.cuda.empty_cache()
                            # Fallback to CPU computation using ORIGINAL PyTorch function to avoid recursion
                            B_cpu = B.tensor().cpu()
                            if hasattr(torch, '_original_matmul'):
                                chunk_result = torch._original_matmul(A_chunk_cpu, B_cpu)
                            else:
                                chunk_result = torch.matmul(A_chunk_cpu, B_cpu)
                            del B_cpu
                        else:
                            raise
                
                # Store result in CPU matrix (ensure CPU-only storage)
                result_slice = slice(current_row, current_row + chunk_rows)
                if isinstance(chunk_result, torch.Tensor):
                    # Ensure chunk_result is on CPU
                    chunk_result = chunk_result.cpu()
                    result[result_slice, :] = Matrix(chunk_result)
                else:
                    result[result_slice, :] = Matrix(chunk_result.tensor().cpu())
                
                current_row += chunk_rows
                
                # Aggressive cleanup of all chunk memory
                del A_chunk_cpu, chunk_result
                
                # Force immediate GPU memory cleanup after every chunk
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # Additional aggressive memory cleanup every 5 chunks
                if chunk_idx % 5 == 0:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Additional cleanup if memory is getting low
                current_available_gb = self.memory_manager.get_available_memory() / (1024**3)
                if current_available_gb < 1.0:  # Less than 1GB available
                    # Force Python garbage collection
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                
                # Check if we're running out of memory - switch to CPU fallback for future operations
                current_available_gb = self.memory_manager.get_available_memory() / (1024**3)
                if current_available_gb < 0.4:  # Less than 400MB available
                    logger.warning(f"Critical GPU memory low ({current_available_gb:.2f}GB), will prioritize CPU fallback")
                    # Force cleanup and prepare for CPU processing
                    if B_gpu is not None:
                        del B_gpu
                        B_gpu = None
                    # Force aggressive cleanup
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                self.memory_manager.cleanup()
                
                if pbar:
                    pbar.update(1)
                    # Update progress bar with memory info
                    memory_info = self.memory_manager.get_memory_usage()
                    pbar.set_postfix({
                        'GPU_util': f"{memory_info['gpu_utilization']:.1f}%",
                        'GPU_avail': f"{memory_info['gpu_available_gb']:.1f}GB"
                    })
        
        finally:
            if pbar:
                pbar.close()
            if B_gpu is not None:
                del B_gpu
                B_gpu = None
            
            # Force final cleanup
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.memory_manager.cleanup()
        
        logger.info("Matrix multiplication completed successfully")
        
        # Ensure result is on CPU and clean up any remaining GPU memory
        if hasattr(result, 'cleanup_gpu'):
            result.cleanup_gpu()
        
        return result
    
    def add(self, 
            A: Union[Matrix, torch.Tensor, np.ndarray], 
            B: Union[Matrix, torch.Tensor, np.ndarray]) -> Matrix:
        """
        Memory-efficient matrix addition A + B.
        
        Args:
            A: First matrix
            B: Second matrix
            
        Returns:
            Result matrix A + B
        """
        A = self._prepare_matrix(A)
        B = self._prepare_matrix(B)
        
        if A.shape != B.shape:
            raise ValueError(f"Matrix shapes must match: {A.shape} vs {B.shape}")
        
        logger.info(f"Computing matrix addition: {A.shape} + {B.shape}")
        
        # Check if we can do the operation directly
        total_memory_needed = A.memory_size() + B.memory_size() + A.memory_size()  # A + B + result
        
        if self.memory_manager.can_allocate(total_memory_needed):
            return self._direct_add(A, B)
        
        return self._chunked_add(A, B)
    
    def _direct_add(self, A: Matrix, B: Matrix) -> Matrix:
        """Direct addition on GPU."""
        A_gpu = self.memory_manager.move_to_gpu(A.tensor())
        B_gpu = self.memory_manager.move_to_gpu(B.tensor())
        
        result = A_gpu + B_gpu
        result_cpu = self.memory_manager.move_to_cpu(result)
        
        del A_gpu, B_gpu
        self.memory_manager.cleanup()
        
        return Matrix(result_cpu)
    
    def _chunked_add(self, A: Matrix, B: Matrix) -> Matrix:
        """Chunked addition for large matrices."""
        chunk_size = self._get_optimal_chunk_size(A.shape, 'add')
        
        # Initialize result
        result = Matrix.zeros(A.shape, dtype=A.dtype)
        
        # Create chunk iterators
        A_iter = ChunkIterator(A.tensor(), self.chunking_strategy, chunk_size, self.memory_manager)
        B_iter = ChunkIterator(B.tensor(), self.chunking_strategy, chunk_size, self.memory_manager)
        
        pbar = None
        if self.enable_progress:
            pbar = tqdm(total=len(A_iter), desc="Matrix addition", unit="chunk")
        
        try:
            current_row = 0
            
            for A_chunk, B_chunk in zip(A_iter, B_iter):
                chunk_rows = A_chunk.shape[0]
                
                # Compute chunk result
                chunk_result = A_chunk + B_chunk
                
                # Store result
                result_slice = slice(current_row, current_row + chunk_rows)
                result[result_slice] = Matrix(self.memory_manager.move_to_cpu(chunk_result))
                
                current_row += chunk_rows
                
                # Cleanup
                del A_chunk, B_chunk, chunk_result
                self.memory_manager.cleanup()
                
                if pbar:
                    pbar.update(1)
        
        finally:
            if pbar:
                pbar.close()
            self.memory_manager.cleanup()
        
        return result
    
    def multiply(self, 
                 A: Union[Matrix, torch.Tensor, np.ndarray], 
                 B: Union[Matrix, torch.Tensor, np.ndarray]) -> Matrix:
        """Element-wise multiplication (same as add but with multiplication)."""
        A = self._prepare_matrix(A)
        B = self._prepare_matrix(B)
        
        if A.shape != B.shape:
            raise ValueError(f"Matrix shapes must match: {A.shape} vs {B.shape}")
        
        logger.info(f"Computing element-wise multiplication: {A.shape} * {B.shape}")
        
        total_memory_needed = A.memory_size() + B.memory_size() + A.memory_size()
        
        if self.memory_manager.can_allocate(total_memory_needed):
            return self._direct_multiply(A, B)
        
        return self._chunked_multiply(A, B)
    
    def _direct_multiply(self, A: Matrix, B: Matrix) -> Matrix:
        """Direct element-wise multiplication on GPU."""
        A_gpu = self.memory_manager.move_to_gpu(A.tensor())
        B_gpu = self.memory_manager.move_to_gpu(B.tensor())
        
        result = A_gpu * B_gpu
        result_cpu = self.memory_manager.move_to_cpu(result)
        
        del A_gpu, B_gpu
        self.memory_manager.cleanup()
        
        return Matrix(result_cpu)
    
    def _chunked_multiply(self, A: Matrix, B: Matrix) -> Matrix:
        """Chunked element-wise multiplication."""
        chunk_size = self._get_optimal_chunk_size(A.shape, 'multiply')
        
        result = Matrix.zeros(A.shape, dtype=A.dtype)
        
        A_iter = ChunkIterator(A.tensor(), self.chunking_strategy, chunk_size, self.memory_manager)
        B_iter = ChunkIterator(B.tensor(), self.chunking_strategy, chunk_size, self.memory_manager)
        
        pbar = None
        if self.enable_progress:
            pbar = tqdm(total=len(A_iter), desc="Element-wise multiplication", unit="chunk")
        
        try:
            current_row = 0
            
            for A_chunk, B_chunk in zip(A_iter, B_iter):
                chunk_rows = A_chunk.shape[0]
                
                chunk_result = A_chunk * B_chunk
                
                result_slice = slice(current_row, current_row + chunk_rows)
                result[result_slice] = Matrix(self.memory_manager.move_to_cpu(chunk_result))
                
                current_row += chunk_rows
                
                del A_chunk, B_chunk, chunk_result
                self.memory_manager.cleanup()
                
                if pbar:
                    pbar.update(1)
        
        finally:
            if pbar:
                pbar.close()
            self.memory_manager.cleanup()
        
        return result
    
    def get_memory_info(self) -> dict:
        """Get current memory usage information."""
        return self.memory_manager.get_memory_usage()
    
    def cleanup(self):
        """Force cleanup of GPU memory."""
        self.memory_manager.cleanup()
    
    def set_chunk_size(self, chunk_size: int):
        """Set default chunk size."""
        self.default_chunk_size = chunk_size
        self.auto_detect_memory = False
    
    def enable_auto_memory_detection(self, enable: bool = True):
        """Enable or disable automatic memory detection."""
        self.auto_detect_memory = enable