"""
Matrix Wrapper Module

Provides a high-level Matrix class with automatic memory management.
"""

import torch
import numpy as np
from typing import Union, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class Matrix:
    """
    High-level matrix wrapper with automatic memory management.
    
    Handles CPU-GPU transfers transparently and provides a numpy-like interface
    for matrix operations with automatic chunking for large operations.
    """
    
    def __init__(self, 
                 data: Union[torch.Tensor, np.ndarray, list], 
                 device: Optional[str] = None,
                 dtype: Optional[torch.dtype] = None):
        """
        Initialize Matrix wrapper.
        
        Args:
            data: Matrix data (tensor, numpy array, or list)
            device: Device to store the matrix ('cpu', 'cuda', etc.)
            dtype: Data type for the matrix
        """
        # Convert to tensor if needed
        if isinstance(data, np.ndarray):
            self._tensor = torch.from_numpy(data)
        elif isinstance(data, list):
            self._tensor = torch.tensor(data)
        elif isinstance(data, torch.Tensor):
            self._tensor = data.clone()
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        # Set dtype if specified
        if dtype is not None:
            self._tensor = self._tensor.to(dtype)
        
        # Set device - default to CPU for storage
        self._storage_device = torch.device(device if device is not None else 'cpu')
        self._tensor = self._tensor.to(self._storage_device)
        
        # Track if currently on GPU for operations
        self._gpu_device = None
        
    @property
    def shape(self) -> torch.Size:
        """Get matrix shape."""
        return self._tensor.shape
    
    @property
    def dtype(self) -> torch.dtype:
        """Get matrix data type."""
        return self._tensor.dtype
    
    @property
    def device(self) -> torch.device:
        """Get current device of the matrix."""
        return self._tensor.device
    
    @property
    def storage_device(self) -> torch.device:
        """Get storage device (where matrix is kept when not in use)."""
        return self._storage_device
    
    def size(self) -> torch.Size:
        """Get matrix size (alias for shape)."""
        return self.shape
    
    def numel(self) -> int:
        """Get total number of elements."""
        return self._tensor.numel()
    
    def memory_size(self) -> int:
        """Get memory size in bytes."""
        return self._tensor.numel() * self._tensor.element_size()
    
    def to_gpu(self, device: str = 'cuda:0') -> 'Matrix':
        """
        Move matrix to GPU.
        
        Args:
            device: GPU device to move to
            
        Returns:
            Self for method chaining
        """
        gpu_device = torch.device(device)
        self._tensor = self._tensor.to(gpu_device)
        self._gpu_device = gpu_device
        return self
    
    def to_cpu(self) -> 'Matrix':
        """
        Move matrix to CPU storage and clean up GPU memory.
        
        Returns:
            Self for method chaining
        """
        if self._tensor.device != self._storage_device:
            # Move to CPU
            cpu_tensor = self._tensor.cpu()
            # Delete GPU reference
            del self._tensor
            # Force GPU cleanup
            torch.cuda.empty_cache()
            self._tensor = cpu_tensor
        self._gpu_device = None
        return self
    
    def tensor(self) -> torch.Tensor:
        """Get underlying tensor."""
        return self._tensor
    
    def numpy(self) -> np.ndarray:
        """Convert to numpy array (moves to CPU if needed)."""
        if self._tensor.device != torch.device('cpu'):
            return self._tensor.cpu().numpy()
        return self._tensor.numpy()
    
    def cleanup_gpu(self):
        """Explicitly clean up any GPU memory held by this matrix."""
        if self._tensor.device.type == 'cuda':
            # Move to CPU and clean up GPU memory
            cpu_tensor = self._tensor.cpu()
            del self._tensor
            torch.cuda.empty_cache()
            self._tensor = cpu_tensor
            self._gpu_device = None
    
    def __del__(self):
        """Destructor to ensure GPU memory is cleaned up."""
        try:
            if hasattr(self, '_tensor') and self._tensor.device.type == 'cuda':
                self.cleanup_gpu()
        except:
            # Ignore errors during cleanup
            pass
    
    def clone(self) -> 'Matrix':
        """Create a deep copy of the matrix."""
        return Matrix(self._tensor.clone(), 
                     device=str(self._storage_device), 
                     dtype=self.dtype)
    
    def __getitem__(self, key) -> 'Matrix':
        """Get slice of matrix."""
        return Matrix(self._tensor[key], 
                     device=str(self._storage_device), 
                     dtype=self.dtype)
    
    def __setitem__(self, key, value):
        """Set slice of matrix."""
        if isinstance(value, Matrix):
            self._tensor[key] = value._tensor
        elif isinstance(value, torch.Tensor):
            self._tensor[key] = value
        else:
            self._tensor[key] = torch.tensor(value, dtype=self.dtype)
    
    def __repr__(self) -> str:
        """String representation of matrix."""
        return f"Matrix(shape={self.shape}, dtype={self.dtype}, device={self.device})"
    
    def __str__(self) -> str:
        """String representation of matrix."""
        return f"Matrix({self._tensor})"
    
    # Mathematical operations
    def __add__(self, other) -> 'Matrix':
        """Addition."""
        if isinstance(other, Matrix):
            result = self._tensor + other._tensor
        else:
            result = self._tensor + other
        return Matrix(result, device=str(self._storage_device), dtype=self.dtype)
    
    def __sub__(self, other) -> 'Matrix':
        """Subtraction."""
        if isinstance(other, Matrix):
            result = self._tensor - other._tensor
        else:
            result = self._tensor - other
        return Matrix(result, device=str(self._storage_device), dtype=self.dtype)
    
    def __mul__(self, other) -> 'Matrix':
        """Element-wise multiplication."""
        if isinstance(other, Matrix):
            result = self._tensor * other._tensor
        else:
            result = self._tensor * other
        return Matrix(result, device=str(self._storage_device), dtype=self.dtype)
    
    def __truediv__(self, other) -> 'Matrix':
        """Element-wise division."""
        if isinstance(other, Matrix):
            result = self._tensor / other._tensor
        else:
            result = self._tensor / other
        return Matrix(result, device=str(self._storage_device), dtype=self.dtype)
    
    def __matmul__(self, other) -> 'Matrix':
        """Matrix multiplication (use MatrixEngine for large matrices)."""
        if not isinstance(other, Matrix):
            raise TypeError("Matrix multiplication requires another Matrix")
        
        # Use ORIGINAL PyTorch function to avoid recursion if interceptors are active
        if hasattr(torch, '_original_matmul'):
            result = torch._original_matmul(self._tensor, other._tensor)
        else:
            result = torch.matmul(self._tensor, other._tensor)
        return Matrix(result, device=str(self._storage_device), dtype=self.dtype)
    
    def transpose(self, dim0: int = -2, dim1: int = -1) -> 'Matrix':
        """Transpose matrix."""
        result = self._tensor.transpose(dim0, dim1)
        return Matrix(result, device=str(self._storage_device), dtype=self.dtype)
    
    def T(self) -> 'Matrix':
        """Transpose matrix (property-like access)."""
        return self.transpose()
    
    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> Union['Matrix', torch.Tensor]:
        """Sum along dimension."""
        result = self._tensor.sum(dim=dim, keepdim=keepdim)
        if dim is None and not keepdim:
            return result  # Return scalar
        return Matrix(result, device=str(self._storage_device), dtype=self.dtype)
    
    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> Union['Matrix', torch.Tensor]:
        """Mean along dimension."""
        result = self._tensor.mean(dim=dim, keepdim=keepdim)
        if dim is None and not keepdim:
            return result  # Return scalar
        return Matrix(result, device=str(self._storage_device), dtype=self.dtype)
    
    def max(self, dim: Optional[int] = None, keepdim: bool = False):
        """Maximum along dimension."""
        if dim is None:
            return self._tensor.max()
        result = self._tensor.max(dim=dim, keepdim=keepdim)
        if hasattr(result, 'values'):  # torch.max returns named tuple when dim is specified
            values = Matrix(result.values, device=str(self._storage_device), dtype=self.dtype)
            return values, result.indices
        return Matrix(result, device=str(self._storage_device), dtype=self.dtype)
    
    def min(self, dim: Optional[int] = None, keepdim: bool = False):
        """Minimum along dimension."""
        if dim is None:
            return self._tensor.min()
        result = self._tensor.min(dim=dim, keepdim=keepdim)
        if hasattr(result, 'values'):  # torch.min returns named tuple when dim is specified
            values = Matrix(result.values, device=str(self._storage_device), dtype=self.dtype)
            return values, result.indices
        return Matrix(result, device=str(self._storage_device), dtype=self.dtype)
    
    @staticmethod
    def zeros(shape: Tuple[int, ...], 
              dtype: torch.dtype = torch.float32, 
              device: str = 'cpu') -> 'Matrix':
        """Create matrix filled with zeros."""
        tensor = torch.zeros(shape, dtype=dtype)
        return Matrix(tensor, device=device, dtype=dtype)
    
    @staticmethod
    def ones(shape: Tuple[int, ...], 
             dtype: torch.dtype = torch.float32, 
             device: str = 'cpu') -> 'Matrix':
        """Create matrix filled with ones."""
        tensor = torch.ones(shape, dtype=dtype)
        return Matrix(tensor, device=device, dtype=dtype)
    
    @staticmethod
    def eye(n: int, 
            m: Optional[int] = None,
            dtype: torch.dtype = torch.float32, 
            device: str = 'cpu') -> 'Matrix':
        """Create identity matrix."""
        tensor = torch.eye(n, m, dtype=dtype)
        return Matrix(tensor, device=device, dtype=dtype)
    
    @staticmethod
    def random(shape: Tuple[int, ...], 
               dtype: torch.dtype = torch.float32, 
               device: str = 'cpu') -> 'Matrix':
        """Create matrix with random values."""
        tensor = torch.rand(shape, dtype=dtype)
        return Matrix(tensor, device=device, dtype=dtype)
    
    @staticmethod
    def randn(shape: Tuple[int, ...], 
              dtype: torch.dtype = torch.float32, 
              device: str = 'cpu') -> 'Matrix':
        """Create matrix with random normal values."""
        tensor = torch.randn(shape, dtype=dtype)
        return Matrix(tensor, device=device, dtype=dtype)
    
    @staticmethod
    def from_numpy(array: np.ndarray, device: str = 'cpu') -> 'Matrix':
        """Create matrix from numpy array."""
        return Matrix(array, device=device)
    
    @staticmethod
    def from_tensor(tensor: torch.Tensor, device: str = 'cpu') -> 'Matrix':
        """Create matrix from torch tensor."""
        return Matrix(tensor, device=device)