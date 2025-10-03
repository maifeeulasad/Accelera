"""
Subprocess-based Matrix Engine Module

Runs matrix operations in separate subprocesses to ensure complete GPU memory cleanup.
"""

import torch
import numpy as np
import multiprocessing as mp
import pickle
import tempfile
import os
import logging
from typing import Optional, Union, Tuple, Any

from .engine import MatrixEngine
from .matrix import Matrix

logger = logging.getLogger(__name__)


def _subprocess_matmul(args):
    """
    Worker function to perform matrix multiplication in subprocess.
    
    Args:
        args: Tuple containing (A_path, B_path, result_path, engine_config)
    """
    try:
        A_path, B_path, result_path, engine_config = args
        
        # Load matrices from temporary files
        A_data = torch.load(A_path, map_location='cpu')
        B_data = torch.load(B_path, map_location='cpu')
        
        # Create engine with config
        engine = MatrixEngine(**engine_config)
        
        # Perform multiplication
        result = engine.matmul(A_data, B_data)
        
        # Convert result to CPU tensor if it's a Matrix
        if isinstance(result, Matrix):
            result_tensor = result.to_cpu()._tensor
        else:
            result_tensor = result.cpu() if result.is_cuda else result
        
        # Save result
        torch.save(result_tensor, result_path)
        
        # Explicit cleanup
        del A_data, B_data, result, result_tensor
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        logger.error(f"Subprocess matrix multiplication failed: {e}")
        return False


class SubprocessMatrixEngine:
    """
    Matrix engine that runs operations in subprocesses for automatic memory cleanup.
    
    This ensures that GPU memory is completely freed after each operation since
    the subprocess terminates and the OS cleans up all allocated resources.
    """
    
    def __init__(self, 
                 device: Optional[str] = None,
                 auto_detect_memory: bool = True,
                 chunking_strategy: str = 'adaptive',
                 chunk_size: Optional[int] = None,
                 enable_progress: bool = True,
                 memory_threshold_gb: Optional[float] = None,
                 subprocess_threshold_gb: float = 0.1):
        """
        Initialize Subprocess Matrix Engine.
        
        Args:
            device: CUDA device to use (e.g., 'cuda:0'). Auto-detects if None.
            auto_detect_memory: Whether to automatically detect optimal chunk sizes
            chunking_strategy: Strategy for chunking ('row', 'tile', 'adaptive')
            chunk_size: Fixed chunk size (ignored if auto_detect_memory=True)
            enable_progress: Whether to show progress bars for long operations
            memory_threshold_gb: Force chunking if operation exceeds this threshold (GB)
            subprocess_threshold_gb: Use subprocess if operation exceeds this threshold (GB)
        """
        self.subprocess_threshold_gb = subprocess_threshold_gb
        
        # Store engine configuration for subprocess creation
        self.engine_config = {
            'device': device,
            'auto_detect_memory': auto_detect_memory,
            'chunking_strategy': chunking_strategy,
            'chunk_size': chunk_size,
            'enable_progress': enable_progress,
            'memory_threshold_gb': memory_threshold_gb
        }
        
        # Create main engine for small operations
        self.main_engine = MatrixEngine(**self.engine_config)
        
        logger.info(f"SubprocessMatrixEngine initialized (subprocess threshold: {subprocess_threshold_gb:.3f}GB)")
    
    def matmul(self, A: Union[torch.Tensor, Matrix, np.ndarray], 
               B: Union[torch.Tensor, Matrix, np.ndarray],
               chunk_strategy: Optional[str] = None) -> Matrix:
        """
        Memory-efficient matrix multiplication A @ B with subprocess isolation.
        
        Args:
            A: Left matrix (M x K)
            B: Right matrix (K x N)  
            chunk_strategy: Override default chunking strategy
            
        Returns:
            Result matrix (M x N)
        """
        # Convert inputs to tensors for size calculation
        if isinstance(A, Matrix):
            A_tensor = A._tensor
        elif isinstance(A, np.ndarray):
            A_tensor = torch.from_numpy(A).float()
        else:
            A_tensor = A
            
        if isinstance(B, Matrix):
            B_tensor = B._tensor
        elif isinstance(B, np.ndarray):
            B_tensor = torch.from_numpy(B).float()
        else:
            B_tensor = B
        
        # Calculate memory requirements
        A_memory = A_tensor.numel() * 4  # float32 = 4 bytes
        B_memory = B_tensor.numel() * 4
        result_memory = A_tensor.shape[0] * B_tensor.shape[1] * 4
        total_memory_needed = A_memory + B_memory + result_memory
        total_memory_gb = total_memory_needed / 1e9
        
        logger.info(f"Matrix multiplication: {A_tensor.shape} @ {B_tensor.shape}")
        logger.info(f"Total memory needed: {total_memory_gb:.3f}GB")
        
        # Use subprocess if memory exceeds threshold
        if total_memory_gb >= self.subprocess_threshold_gb:
            logger.info(f"Using subprocess execution (threshold: {self.subprocess_threshold_gb:.3f}GB)")
            return self._subprocess_matmul(A, B, chunk_strategy)
        else:
            logger.info("Using main process execution")
            result = self.main_engine.matmul(A, B, chunk_strategy)
            # Ensure cleanup after main process operation
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return result
    
    def _subprocess_matmul(self, A, B, chunk_strategy):
        """Perform matrix multiplication in subprocess."""
        # Create temporary files for data transfer
        with tempfile.TemporaryDirectory() as temp_dir:
            A_path = os.path.join(temp_dir, 'A.pt')
            B_path = os.path.join(temp_dir, 'B.pt')
            result_path = os.path.join(temp_dir, 'result.pt')
            
            # Convert to tensors and save to files
            if isinstance(A, Matrix):
                A_tensor = A.to_cpu()._tensor
            elif isinstance(A, np.ndarray):
                A_tensor = torch.from_numpy(A).float()
            else:
                A_tensor = A.cpu() if A.is_cuda else A
                
            if isinstance(B, Matrix):
                B_tensor = B.to_cpu()._tensor
            elif isinstance(B, np.ndarray):
                B_tensor = torch.from_numpy(B).float()
            else:
                B_tensor = B.cpu() if B.is_cuda else B
            
            torch.save(A_tensor, A_path)
            torch.save(B_tensor, B_path)
            
            # Prepare arguments for subprocess
            args = (A_path, B_path, result_path, self.engine_config)
            
            # Run in subprocess
            try:
                # Use spawn method to ensure complete process isolation
                ctx = mp.get_context('spawn')
                with ctx.Pool(1) as pool:
                    success = pool.apply(_subprocess_matmul, (args,))
                    pool.close()
                    pool.join()
                
                if not success:
                    raise RuntimeError("Subprocess matrix multiplication failed")
                
                # Load result
                result_tensor = torch.load(result_path, map_location='cpu')
                result_matrix = Matrix(result_tensor)
                
                logger.info("✅ Subprocess matrix multiplication completed successfully")
                return result_matrix
                
            except Exception as e:
                logger.error(f"Subprocess execution failed: {e}")
                logger.info("Falling back to main process execution")
                return self.main_engine.matmul(A, B, chunk_strategy)
            finally:
                # Cleanup main process memory
                del A_tensor, B_tensor
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def add(self, A: Union[torch.Tensor, Matrix, np.ndarray], 
            B: Union[torch.Tensor, Matrix, np.ndarray]) -> Matrix:
        """Element-wise addition."""
        return self.main_engine.add(A, B)
    
    def multiply(self, A: Union[torch.Tensor, Matrix, np.ndarray], 
                 B: Union[torch.Tensor, Matrix, np.ndarray]) -> Matrix:
        """Element-wise multiplication."""
        return self.main_engine.multiply(A, B)
    
    def transpose(self, A: Union[torch.Tensor, Matrix, np.ndarray]) -> Matrix:
        """Matrix transpose."""
        return self.main_engine.transpose(A)
    
    def norm(self, A: Union[torch.Tensor, Matrix, np.ndarray], p: Union[int, float, str] = 'fro') -> float:
        """Matrix norm."""
        return self.main_engine.norm(A, p)
    
    def svd(self, A: Union[torch.Tensor, Matrix, np.ndarray]) -> Tuple[Matrix, Matrix, Matrix]:
        """Singular Value Decomposition."""
        return self.main_engine.svd(A)
    
    def get_memory_info(self) -> dict:
        """Get GPU memory information."""
        return self.main_engine.get_memory_info()