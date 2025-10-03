"""
PyTorch Interceptor Module

Transparently intercepts PyTorch tensor operations and routes large operations
through Accelera for automatic memory management.
"""

import torch
import torch.nn.functional as F
import functools
import logging
from typing import Optional, Any, Callable
import os

from .memory_efficient_engine import MemoryEfficientEngine
from .matrix import Matrix

logger = logging.getLogger(__name__)


class AcceleraConfig:
    """Configuration for PyTorch interception."""
    
    def __init__(self):
        self.enabled = True
        self.min_size_threshold = int(os.getenv('ACCELERA_MIN_SIZE', '2048'))  # Higher threshold for safety
        self.memory_threshold_gb = float(os.getenv('ACCELERA_MEMORY_THRESHOLD_GB', '1.0'))  # Higher memory threshold
        self.engine = None
        self.verbose = os.getenv('ACCELERA_VERBOSE', 'false').lower() == 'true'
    
    def get_engine(self) -> MemoryEfficientEngine:
        """Get or create the global Accelera memory-efficient engine."""
        if self.engine is None:
            # Get fallback strategy from environment (default to CPU)
            fallback_strategy = os.getenv('ACCELERA_FALLBACK_STRATEGY', 'cpu')
            
            # Create memory-efficient engine with configuration from environment/CLI
            self.engine = MemoryEfficientEngine(
                enable_progress=False,  # Disable progress for transparent operation
                memory_threshold_gb=self.memory_threshold_gb,  # Pass the threshold
                fallback_strategy=fallback_strategy  # Pass fallback strategy
            )
        return self.engine


# Global configuration
_config = AcceleraConfig()


def should_intercept(tensor_a: torch.Tensor, tensor_b: torch.Tensor = None) -> bool:
    """
    Determine if operation should be intercepted by Accelera.
    
    Args:
        tensor_a: First tensor
        tensor_b: Second tensor (optional)
        
    Returns:
        True if operation should use Accelera
    """
    if not _config.enabled:
        return False
    
    # Check if tensors are large enough
    def is_large_enough(tensor):
        if tensor is None:
            return False
            
        # Only intercept actual tensors, not scalars
        if not isinstance(tensor, torch.Tensor):
            return False
        
        # Don't intercept if tensor has unusual dimensions (could break shape expectations)
        if len(tensor.shape) < 2:  # Skip 1D tensors
            return False
            
        if len(tensor.shape) > 4:  # Skip tensors with more than 4 dimensions for safety
            return False
        
        # Check dimension threshold - be more conservative
        max_dim = max(tensor.shape) if tensor.shape else 0
        if max_dim < _config.min_size_threshold:
            return False
        
        # Check memory threshold
        memory_gb = tensor.numel() * tensor.element_size() / 1e9
        return memory_gb >= _config.memory_threshold_gb
    
    # Only intercept if at least one tensor is large enough and compatible
    tensor_a_large = is_large_enough(tensor_a)
    tensor_b_large = is_large_enough(tensor_b)
    
    # Additional safety check: if we have a 4D tensor (likely from FLUX model), be extra careful
    if isinstance(tensor_a, torch.Tensor) and len(tensor_a.shape) == 4:
        # For 4D tensors, only intercept if they're really large to avoid breaking model expectations
        min_4d_size = max(_config.min_size_threshold * 4, 2048)  # Higher threshold for 4D tensors
        if max(tensor_a.shape) < min_4d_size:
            return False
    
    if isinstance(tensor_b, torch.Tensor) and len(tensor_b.shape) == 4:
        min_4d_size = max(_config.min_size_threshold * 4, 2048)
        if max(tensor_b.shape) < min_4d_size:
            return False
    
    return tensor_a_large or tensor_b_large


def log_interception(operation: str, shapes: tuple, use_accelera: bool):
    """Log operation interception for debugging."""
    if _config.verbose:
        status = "ACCELERA" if use_accelera else "PYTORCH"
        # Handle cases where arguments might not be tensors
        formatted_shapes = []
        for shape in shapes:
            if hasattr(shape, 'shape'):
                formatted_shapes.append(shape.shape)
            else:
                formatted_shapes.append(f"scalar({type(shape).__name__})")
        logger.info(f"[{status}] {operation} with shapes {tuple(formatted_shapes)}")


def accelera_matmul(input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Accelera-powered matrix multiplication."""
    # Only intercept if both arguments are tensors
    if isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor) and should_intercept(input, other):
        log_interception("matmul", (input, other), True)
        
        # Store original shape and device info for debugging
        original_input_shape = input.shape
        original_other_shape = other.shape
        original_device = input.device
        original_dtype = input.dtype
        
        try:
            engine = _config.get_engine()
            
            # Convert to Accelera matrices
            A = Matrix(input)
            B = Matrix(other)
            
            # Perform operation
            result = engine.matmul(A, B)
            
            # Convert back to tensor on original device and dtype
            result_tensor = result.tensor().to(device=original_device, dtype=original_dtype)
            
            # Verify the result shape is what we expect for matrix multiplication
            expected_shape = original_input_shape[:-1] + original_other_shape[-1:]
            if result_tensor.shape != expected_shape:
                logger.warning(f"[ACCELERA] Shape mismatch detected! Expected {expected_shape}, got {result_tensor.shape}")
                logger.warning(f"[ACCELERA] Falling back to original PyTorch for safety")
                return torch._original_matmul(input, other, out=out)
            
            # Debug: Check if shape is preserved correctly
            if _config.verbose:
                logger.info(f"[ACCELERA] matmul shapes: {original_input_shape} @ {original_other_shape} -> {result_tensor.shape}")
            
            if out is not None:
                out.copy_(result_tensor)
                return out
            
            return result_tensor
            
        except Exception as e:
            # If anything goes wrong with Accelera, fall back to original PyTorch
            logger.warning(f"[ACCELERA] Error in matmul, falling back to PyTorch: {e}")
            return torch._original_matmul(input, other, out=out)
    else:
        log_interception("matmul", (input, other), False)
        return torch._original_matmul(input, other, out=out)


def accelera_mm(input: torch.Tensor, mat2: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Accelera-powered mm (matrix multiplication)."""
    if isinstance(input, torch.Tensor) and isinstance(mat2, torch.Tensor) and should_intercept(input, mat2):
        log_interception("mm", (input, mat2), True)
        return accelera_matmul(input, mat2, out=out)
    else:
        log_interception("mm", (input, mat2), False)
        return torch._original_mm(input, mat2, out=out)


def accelera_bmm(input: torch.Tensor, mat2: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Accelera-powered batch matrix multiplication."""
    if isinstance(input, torch.Tensor) and isinstance(mat2, torch.Tensor) and should_intercept(input, mat2):
        log_interception("bmm", (input, mat2), True)
        
        # Handle batch dimension
        batch_size = input.shape[0]
        results = []
        
        engine = _config.get_engine()
        
        for i in range(batch_size):
            A = Matrix(input[i])
            B = Matrix(mat2[i])
            result = engine.matmul(A, B)
            results.append(result.tensor())
        
        result_tensor = torch.stack(results).to(input.device)
        
        if out is not None:
            out.copy_(result_tensor)
            return out
        
        return result_tensor
    else:
        log_interception("bmm", (input, mat2), False)
        return torch._original_bmm(input, mat2, out=out)
        return torch._original_bmm(input, mat2, out=out)


def accelera_add(input: torch.Tensor, other: torch.Tensor, *, alpha: float = 1, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Accelera-powered tensor addition."""
    if alpha != 1:
        # Handle alpha scaling with original PyTorch
        return torch._original_add(input, other, alpha=alpha, out=out)
    
    if isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor) and should_intercept(input, other):
        log_interception("add", (input, other), True)
        
        try:
            # Store original properties
            original_device = input.device
            original_dtype = input.dtype
            original_shape = input.shape
            
            engine = _config.get_engine()
            A = Matrix(input)
            B = Matrix(other)
            
            result = engine.add(A, B)
            result_tensor = result.tensor().to(device=original_device, dtype=original_dtype)
            
            # Verify shape preservation
            if result_tensor.shape != original_shape:
                logger.warning(f"[ACCELERA] Add shape mismatch! Expected {original_shape}, got {result_tensor.shape}")
                return torch._original_add(input, other, alpha=alpha, out=out)
            
            if out is not None:
                out.copy_(result_tensor)
                return out
            
            return result_tensor
            
        except Exception as e:
            logger.warning(f"[ACCELERA] Error in add, falling back to PyTorch: {e}")
            return torch._original_add(input, other, alpha=alpha, out=out)
    else:
        log_interception("add", (input, other), False)
        return torch._original_add(input, other, alpha=alpha, out=out)


def accelera_mul(input: torch.Tensor, other: torch.Tensor, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Accelera-powered element-wise multiplication."""
    if isinstance(input, torch.Tensor) and isinstance(other, torch.Tensor) and should_intercept(input, other):
        log_interception("mul", (input, other), True)
        
        try:
            # Store original properties
            original_device = input.device
            original_dtype = input.dtype
            original_shape = input.shape
            
            engine = _config.get_engine()
            A = Matrix(input)
            B = Matrix(other)
            
            result = engine.multiply(A, B)
            result_tensor = result.tensor().to(device=original_device, dtype=original_dtype)
            
            # Verify shape preservation
            if result_tensor.shape != original_shape:
                logger.warning(f"[ACCELERA] Mul shape mismatch! Expected {original_shape}, got {result_tensor.shape}")
                return torch._original_mul(input, other, out=out)
            
            if out is not None:
                out.copy_(result_tensor)
                return out
            
            return result_tensor
            
        except Exception as e:
            logger.warning(f"[ACCELERA] Error in mul, falling back to PyTorch: {e}")
            return torch._original_mul(input, other, out=out)
    else:
        log_interception("mul", (input, other), False)
        return torch._original_mul(input, other, out=out)


def patch_torch():
    """
    Patch PyTorch functions to use Accelera for large operations.
    
    This is the main function that intercepts PyTorch operations.
    """
    if hasattr(torch, '_accelera_patched'):
        logger.warning("PyTorch already patched with Accelera")
        return
    
    logger.info("Patching PyTorch with Accelera interceptors...")
    
    # Store original functions
    torch._original_matmul = torch.matmul
    torch._original_mm = torch.mm
    torch._original_bmm = torch.bmm
    torch._original_add = torch.add
    torch._original_mul = torch.mul
    
    # Patch with Accelera versions
    torch.matmul = accelera_matmul
    torch.mm = accelera_mm
    torch.bmm = accelera_bmm
    torch.add = accelera_add
    torch.mul = accelera_mul
    
    # Also patch tensor methods
    original_tensor_matmul = torch.Tensor.matmul
    original_tensor_mm = torch.Tensor.mm
    original_tensor_add = torch.Tensor.add
    original_tensor_mul = torch.Tensor.mul
    
    def tensor_matmul(self, other):
        return accelera_matmul(self, other)
    
    def tensor_mm(self, mat2):
        return accelera_mm(self, mat2)
    
    def tensor_add(self, other, *, alpha=1):
        return accelera_add(self, other, alpha=alpha)
    
    def tensor_mul(self, other):
        return accelera_mul(self, other)
    
    torch.Tensor.matmul = tensor_matmul
    torch.Tensor.mm = tensor_mm
    torch.Tensor.add = tensor_add
    torch.Tensor.mul = tensor_mul
    
    # Store original tensor methods for restoration
    torch.Tensor._original_matmul = original_tensor_matmul
    torch.Tensor._original_mm = original_tensor_mm
    torch.Tensor._original_add = original_tensor_add
    torch.Tensor._original_mul = original_tensor_mul
    
    # Mark as patched
    torch._accelera_patched = True
    
    logger.info("✅ PyTorch successfully patched with Accelera!")


def unpatch_torch():
    """
    Restore original PyTorch functions.
    """
    if not hasattr(torch, '_accelera_patched'):
        logger.warning("PyTorch not patched with Accelera")
        return
    
    logger.info("Restoring original PyTorch functions...")
    
    # Restore module functions
    torch.matmul = torch._original_matmul
    torch.mm = torch._original_mm
    torch.bmm = torch._original_bmm
    torch.add = torch._original_add
    torch.mul = torch._original_mul
    
    # Restore tensor methods
    torch.Tensor.matmul = torch.Tensor._original_matmul
    torch.Tensor.mm = torch.Tensor._original_mm
    torch.Tensor.add = torch.Tensor._original_add
    torch.Tensor.mul = torch.Tensor._original_mul
    
    # Clean up
    delattr(torch, '_accelera_patched')
    delattr(torch, '_original_matmul')
    delattr(torch, '_original_mm')
    delattr(torch, '_original_bmm')
    delattr(torch, '_original_add')
    delattr(torch, '_original_mul')
    
    delattr(torch.Tensor, '_original_matmul')
    delattr(torch.Tensor, '_original_mm')
    delattr(torch.Tensor, '_original_add')
    delattr(torch.Tensor, '_original_mul')
    
    logger.info("✅ Original PyTorch functions restored!")


def configure(enabled: bool = True,
              min_size_threshold: int = 1000,
              memory_threshold_gb: float = 1.0,
              verbose: bool = False):
    """
    Configure Accelera interception behavior.
    
    Args:
        enabled: Whether to enable Accelera interception
        min_size_threshold: Minimum tensor dimension to intercept
        memory_threshold_gb: Minimum memory usage (GB) to intercept
        verbose: Whether to log all intercepted operations
    """
    global _config
    _config.enabled = enabled
    _config.min_size_threshold = min_size_threshold
    _config.memory_threshold_gb = memory_threshold_gb
    _config.verbose = verbose
    
    if verbose:
        logger.setLevel(logging.INFO)
    
    logger.info(f"Accelera configuration updated: enabled={enabled}, "
                f"min_size={min_size_threshold}, memory_threshold={memory_threshold_gb}GB")


def status():
    """Get current Accelera interception status."""
    patched = hasattr(torch, '_accelera_patched')
    
    print("🚀 Accelera PyTorch Interception Status")
    print("=" * 40)
    print(f"Patched: {'✅ Yes' if patched else '❌ No'}")
    print(f"Enabled: {'✅ Yes' if _config.enabled else '❌ No'}")
    print(f"Min size threshold: {_config.min_size_threshold}")
    print(f"Memory threshold: {_config.memory_threshold_gb} GB")
    print(f"Verbose logging: {'✅ Yes' if _config.verbose else '❌ No'}")
    
    if _config.engine:
        memory_info = _config.engine.get_memory_info()
        print(f"GPU utilization: {memory_info['gpu_utilization']:.1f}%")


# Auto-patch if requested via environment variable
if os.getenv('ACCELERA_AUTO_PATCH', 'false').lower() == 'true':
    patch_torch()