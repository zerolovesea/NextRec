"""
Utilities package for NextRec

This package provides various utility functions organized by category:
- optimizer: Optimizer and scheduler utilities
- initializer: Weight initialization utilities
- embedding: Embedding dimension calculation
- device_utils: Device management and selection
- tensor_utils: Tensor operations and conversions
- file_utils: File I/O operations
- model_utils: Model-related utilities
- feature_utils: Feature processing utilities

Date: create on 13/11/2025
Last update: 03/12/2025 (refactored)
Author: Yang Zhou, zyaztec@gmail.com
"""

from .optimizer import get_optimizer, get_scheduler
from .initializer import get_initializer
from .embedding import get_auto_embedding_dim
from .device import resolve_device, get_device_info
from .tensor import to_tensor, stack_tensors, concat_tensors, pad_sequence_tensors
from .file import resolve_file_paths, read_table, load_dataframes, iter_file_chunks, default_output_dir
from .model import merge_features, get_mlp_output_dim
from .feature import normalize_to_list
from . import optimizer, initializer, embedding

__all__ = [
    # Optimizer & Scheduler
    'get_optimizer',
    'get_scheduler',
    
    # Initializer
    'get_initializer',
    
    # Embedding
    'get_auto_embedding_dim',
    
    # Device utilities
    'resolve_device',
    'get_device_info',
    
    # Tensor utilities
    'to_tensor',
    'stack_tensors',
    'concat_tensors',
    'pad_sequence_tensors',
    
    # File utilities
    'resolve_file_paths',
    'read_table',
    'load_dataframes',
    'iter_file_chunks',
    'default_output_dir',
    
    # Model utilities
    'merge_features',
    'get_mlp_output_dim',
    
    # Feature utilities
    'normalize_to_list',
    
    # Module exports
    'optimizer',
    'initializer',
    'embedding',
]
