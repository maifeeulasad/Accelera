"""
Chunking Strategy Module

Defines different strategies for breaking down large operations into manageable chunks.
"""

import torch
import numpy as np
from typing import List, Tuple, Iterator, Optional
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def get_chunks(self, shape: Tuple[int, ...], chunk_size: int) -> List[Tuple[slice, ...]]:
        """Get list of slices for chunking a tensor."""
        pass


class RowChunking(ChunkingStrategy):
    """
    Chunk along the first dimension (rows).
    
    Most common strategy for matrix operations where we process
    a fixed number of rows at a time.
    """
    
    def get_chunks(self, shape: Tuple[int, ...], chunk_size: int) -> List[Tuple[slice, ...]]:
        """
        Generate row-wise chunks.
        
        Args:
            shape: Shape of the tensor to chunk
            chunk_size: Number of rows per chunk
            
        Returns:
            List of slice tuples for each chunk
        """
        rows = shape[0]
        chunks = []
        
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            # Create slice tuple: (row_slice, all_other_dims...)
            chunk_slice = (slice(start, end),) + tuple(slice(None) for _ in shape[1:])
            chunks.append(chunk_slice)
            
        logger.debug(f"Generated {len(chunks)} row chunks for shape {shape}")
        return chunks


class TileChunking(ChunkingStrategy):
    """
    Chunk in 2D tiles for better cache locality.
    
    Useful for very large matrices where both dimensions
    need to be chunked.
    """
    
    def __init__(self, tile_size: Tuple[int, int]):
        """
        Initialize tile chunking.
        
        Args:
            tile_size: (rows, cols) size of each tile
        """
        self.tile_size = tile_size
    
    def get_chunks(self, shape: Tuple[int, ...], chunk_size: int = None) -> List[Tuple[slice, ...]]:
        """
        Generate 2D tile chunks.
        
        Args:
            shape: Shape of the tensor to chunk
            chunk_size: Ignored for tile chunking (uses tile_size instead)
            
        Returns:
            List of slice tuples for each tile
        """
        if len(shape) < 2:
            raise ValueError("Tile chunking requires at least 2D tensors")
            
        rows, cols = shape[0], shape[1]
        tile_rows, tile_cols = self.tile_size
        
        chunks = []
        
        for row_start in range(0, rows, tile_rows):
            row_end = min(row_start + tile_rows, rows)
            
            for col_start in range(0, cols, tile_cols):
                col_end = min(col_start + tile_cols, cols)
                
                # Create slice tuple
                chunk_slice = (
                    slice(row_start, row_end),
                    slice(col_start, col_end)
                ) + tuple(slice(None) for _ in shape[2:])
                
                chunks.append(chunk_slice)
        
        logger.debug(f"Generated {len(chunks)} tile chunks for shape {shape}")
        return chunks


class AdaptiveChunking(ChunkingStrategy):
    """
    Adaptive chunking that adjusts chunk size based on available memory.
    
    Starts with larger chunks and reduces size if memory pressure is detected.
    """
    
    def __init__(self, memory_manager, min_chunk_size: int = 1):
        """
        Initialize adaptive chunking.
        
        Args:
            memory_manager: MemoryManager instance for memory monitoring
            min_chunk_size: Minimum allowed chunk size
        """
        self.memory_manager = memory_manager
        self.min_chunk_size = min_chunk_size
        self.last_successful_chunk_size = None
    
    def get_chunks(self, shape: Tuple[int, ...], chunk_size: int) -> List[Tuple[slice, ...]]:
        """
        Generate adaptive chunks that adjust to memory pressure.
        
        Args:
            shape: Shape of the tensor to chunk
            chunk_size: Initial chunk size suggestion
            
        Returns:
            List of slice tuples for each chunk
        """
        # Use last successful chunk size if available
        if self.last_successful_chunk_size is not None:
            chunk_size = min(chunk_size, self.last_successful_chunk_size)
        
        rows = shape[0]
        chunks = []
        current_pos = 0
        current_chunk_size = chunk_size
        
        while current_pos < rows:
            # Check available memory and adjust chunk size if needed
            available_memory = self.memory_manager.get_available_memory()
            
            if available_memory < self.memory_manager.total_gpu_memory * 0.2:  # Less than 20% available
                current_chunk_size = max(self.min_chunk_size, current_chunk_size // 2)
                logger.warning(f"Low memory detected, reducing chunk size to {current_chunk_size}")
            
            end = min(current_pos + current_chunk_size, rows)
            chunk_slice = (slice(current_pos, end),) + tuple(slice(None) for _ in shape[1:])
            chunks.append(chunk_slice)
            
            current_pos = end
            
            # Track successful chunk size
            self.last_successful_chunk_size = current_chunk_size
        
        logger.debug(f"Generated {len(chunks)} adaptive chunks for shape {shape}")
        return chunks


class ChunkIterator:
    """
    Iterator for processing tensor chunks with automatic memory management.
    """
    
    def __init__(self, 
                 tensor: torch.Tensor, 
                 chunking_strategy: ChunkingStrategy,
                 chunk_size: int,
                 memory_manager,
                 prefetch: bool = True):
        """
        Initialize chunk iterator.
        
        Args:
            tensor: Tensor to iterate over
            chunking_strategy: Strategy for chunking
            chunk_size: Size of each chunk
            memory_manager: Memory manager for GPU transfers
            prefetch: Whether to prefetch next chunk to GPU
        """
        self.tensor = tensor
        self.chunking_strategy = chunking_strategy
        self.memory_manager = memory_manager
        self.prefetch = prefetch
        
        # Generate all chunks
        self.chunks = chunking_strategy.get_chunks(tensor.shape, chunk_size)
        self.current_idx = 0
        self.prefetched_chunk = None
        
    def __iter__(self):
        """Return iterator."""
        self.current_idx = 0
        self.prefetched_chunk = None
        return self
    
    def __next__(self) -> torch.Tensor:
        """Get next chunk."""
        if self.current_idx >= len(self.chunks):
            # Clean up any remaining prefetched chunk
            if self.prefetched_chunk is not None:
                if hasattr(self.prefetched_chunk, 'device') and self.prefetched_chunk.device.type == 'cuda':
                    del self.prefetched_chunk
                    torch.cuda.empty_cache()
                self.prefetched_chunk = None
            raise StopIteration
        
        # Get current chunk
        if self.prefetched_chunk is not None:
            current_chunk = self.prefetched_chunk
            self.prefetched_chunk = None
        else:
            chunk_slice = self.chunks[self.current_idx]
            current_chunk = self.tensor[chunk_slice]
            # Move to GPU but track for cleanup
            try:
                current_chunk = self.memory_manager.move_to_gpu(current_chunk)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("Could not move chunk to GPU, using CPU")
                    torch.cuda.empty_cache()
                    # Keep on CPU
                else:
                    raise
        
        # Disable prefetching to reduce memory accumulation 
        # Only prefetch if we have sufficient memory
        available_memory = self.memory_manager.get_available_memory()
        memory_threshold = self.memory_manager.total_gpu_memory * 0.3  # Only prefetch if >30% memory available
        
        if self.prefetch and self.current_idx + 1 < len(self.chunks) and available_memory > memory_threshold:
            try:
                next_slice = self.chunks[self.current_idx + 1]
                next_chunk = self.tensor[next_slice]
                self.prefetched_chunk = self.memory_manager.move_to_gpu(next_chunk)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning("Could not prefetch next chunk due to memory constraints")
                    self.prefetched_chunk = None
                    torch.cuda.empty_cache()
                else:
                    raise
        
        self.current_idx += 1
        return current_chunk
    
    def __del__(self):
        """Clean up any remaining GPU memory."""
        try:
            if hasattr(self, 'prefetched_chunk') and self.prefetched_chunk is not None:
                if hasattr(self.prefetched_chunk, 'device') and self.prefetched_chunk.device.type == 'cuda':
                    del self.prefetched_chunk
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                self.prefetched_chunk = None
        except:
            # Ignore cleanup errors
            pass
    
    def __len__(self):
        """Return number of chunks."""
        return len(self.chunks)


def create_chunking_strategy(strategy_type: str, **kwargs) -> ChunkingStrategy:
    """
    Factory function to create chunking strategies.
    
    Args:
        strategy_type: Type of strategy ('row', 'tile', 'adaptive')
        **kwargs: Additional arguments for strategy initialization
        
    Returns:
        ChunkingStrategy instance
    """
    if strategy_type == 'row':
        return RowChunking()
    elif strategy_type == 'tile':
        tile_size = kwargs.get('tile_size', (1024, 1024))
        return TileChunking(tile_size)
    elif strategy_type == 'adaptive':
        memory_manager = kwargs.get('memory_manager')
        if memory_manager is None:
            raise ValueError("AdaptiveChunking requires memory_manager parameter")
        min_chunk_size = kwargs.get('min_chunk_size', 1)
        return AdaptiveChunking(memory_manager, min_chunk_size)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy_type}")