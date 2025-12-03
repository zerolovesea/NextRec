"""
Data utilities package for NextRec

This package provides data processing and manipulation utilities organized by category:
- batch_utils: Batch collation and processing
- data_processing: Data manipulation and user ID extraction
- data_utils: Legacy module (re-exports from specialized modules)
- dataloader: Dataset and DataLoader implementations
- preprocessor: Data preprocessing pipeline

Date: create on 13/11/2025
Last update: 03/12/2025 (refactored)
Author: Yang Zhou, zyaztec@gmail.com
"""

# Batch utilities
from nextrec.data.batch_utils import collate_fn, batch_to_dict, stack_section

# Data processing utilities
from nextrec.data.data_processing import (
    get_column_data,
    split_dict_random,
    build_eval_candidates,
    get_user_ids,
)

# File utilities (from utils package)
from nextrec.utils.file import (
    resolve_file_paths,
    iter_file_chunks,
    read_table,
    load_dataframes,
    default_output_dir,
)

# DataLoader components
from nextrec.data.dataloader import (
    TensorDictDataset,
    FileDataset,
    RecDataLoader,
    build_tensors_from_data,
)

# Preprocessor
from nextrec.data.preprocessor import DataProcessor

# Feature definitions
from nextrec.basic.features import FeatureSet

# Legacy module (for backward compatibility)
from nextrec.data import data_utils

__all__ = [
    # Batch utilities
    'collate_fn',
    'batch_to_dict',
    'stack_section',
    
    # Data processing
    'get_column_data',
    'split_dict_random',
    'build_eval_candidates',
    'get_user_ids',
    
    # File utilities
    'resolve_file_paths',
    'iter_file_chunks',
    'read_table',
    'load_dataframes',
    'default_output_dir',
    
    # DataLoader
    'TensorDictDataset',
    'FileDataset',
    'RecDataLoader',
    'build_tensors_from_data',
    
    # Preprocessor
    'DataProcessor',
    
    # Features
    'FeatureSet',
    
    # Legacy module
    'data_utils',
]
