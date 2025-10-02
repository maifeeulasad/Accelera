"""
Configuration and logging setup for Accelera framework.
"""

import logging
import os
from typing import Optional


def setup_logging(level: str = "INFO", 
                 log_file: Optional[str] = None,
                 enable_cuda_logging: bool = False):
    """
    Setup logging configuration for Accelera.
    
    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: Optional file to write logs to
        enable_cuda_logging: Whether to enable CUDA memory logging
    """
    # Configure logging format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Setup basic configuration
    logging_config = {
        'level': getattr(logging, level.upper()),
        'format': log_format,
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    
    if log_file:
        logging_config['filename'] = log_file
        logging_config['filemode'] = 'a'
    
    logging.basicConfig(**logging_config)
    
    # Configure CUDA logging if requested
    if not enable_cuda_logging:
        # Suppress CUDA warnings for cleaner output
        logging.getLogger('torch.cuda').setLevel(logging.ERROR)
    
    # Set Accelera logger level
    accelera_logger = logging.getLogger('accelera')
    accelera_logger.setLevel(getattr(logging, level.upper()))


class Config:
    """Configuration class for Accelera framework."""
    
    def __init__(self):
        # Memory management settings
        self.memory_threshold = float(os.getenv('ACCELERA_MEMORY_THRESHOLD', '0.9'))
        self.min_chunk_size = int(os.getenv('ACCELERA_MIN_CHUNK_SIZE', '1'))
        self.max_chunk_size = int(os.getenv('ACCELERA_MAX_CHUNK_SIZE', '10000'))
        
        # Performance settings
        self.enable_prefetch = os.getenv('ACCELERA_ENABLE_PREFETCH', 'true').lower() == 'true'
        self.enable_progress = os.getenv('ACCELERA_ENABLE_PROGRESS', 'true').lower() == 'true'
        
        # Device settings
        self.default_device = os.getenv('ACCELERA_DEFAULT_DEVICE', 'cuda:0')
        self.fallback_to_cpu = os.getenv('ACCELERA_FALLBACK_TO_CPU', 'false').lower() == 'true'
        
        # Logging settings
        self.log_level = os.getenv('ACCELERA_LOG_LEVEL', 'INFO')
        self.log_file = os.getenv('ACCELERA_LOG_FILE', None)
    
    def validate(self):
        """Validate configuration values."""
        if not 0.1 <= self.memory_threshold <= 1.0:
            raise ValueError("memory_threshold must be between 0.1 and 1.0")
        
        if self.min_chunk_size < 1:
            raise ValueError("min_chunk_size must be at least 1")
        
        if self.max_chunk_size < self.min_chunk_size:
            raise ValueError("max_chunk_size must be >= min_chunk_size")


# Global configuration instance
config = Config()