"""
Data processing utilities for NextRec (Refactored)

This module now re-exports functions from specialized submodules:
- batch_utils: collate_fn, batch_to_dict
- data_processing: get_column_data, split_dict_random, build_eval_candidates, get_user_ids
- nextrec.utils.file_utils: resolve_file_paths, iter_file_chunks, read_table, load_dataframes, default_output_dir

Date: create on 27/10/2025
Last update: 03/12/2025 (refactored)
Author: Yang Zhou, zyaztec@gmail.com
"""

# Import from new organized modules
from nextrec.data.batch_utils import collate_fn, batch_to_dict, stack_section
from nextrec.data.data_processing import (
    get_column_data,
    split_dict_random,
    build_eval_candidates,
    get_user_ids,
)
from nextrec.utils.file import (
    resolve_file_paths,
    iter_file_chunks,
    read_table,
    load_dataframes,
    default_output_dir,
)

__all__ = [
    # Batch utilities
    "collate_fn",
    "batch_to_dict",
    "stack_section",
    # Data processing
    "get_column_data",
    "split_dict_random",
    "build_eval_candidates",
    "get_user_ids",
    # File utilities
    "resolve_file_paths",
    "iter_file_chunks",
    "read_table",
    "load_dataframes",
    "default_output_dir",
]
