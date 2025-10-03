"""
Accelera - Memory-Efficient Matrix Operations Framework

A framework for performing large matrix operations on memory-constrained GPUs
through intelligent chunking and CPU-GPU memory management.
"""

from .memory_manager import MemoryManager
from .chunking import ChunkingStrategy, RowChunking, AdaptiveChunking, TileChunking
from .matrix import Matrix
from .engine import MatrixEngine
from .subprocess_engine import SubprocessMatrixEngine
from .config import setup_logging, config
from . import interceptor

__version__ = "0.1.0"
__all__ = [
    "MatrixEngine", 
    "SubprocessMatrixEngine",
    "Matrix", 
    "MemoryManager", 
    "ChunkingStrategy", 
    "RowChunking", 
    "AdaptiveChunking", 
    "TileChunking",
    "setup_logging",
    "config",
    "interceptor"
]

# Setup default logging
setup_logging(level=config.log_level, log_file=config.log_file)