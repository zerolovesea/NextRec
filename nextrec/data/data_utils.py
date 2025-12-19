"""
Data processing utilities for NextRec

Date: create on 27/10/2025
Last update: 19/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

# Import from new organized modules
from nextrec.data.batch_utils import batch_to_dict, collate_fn, stack_section
from nextrec.data.data_processing import (
    build_eval_candidates,
    get_column_data,
    get_user_ids,
    split_dict_random,
)
from nextrec.utils.data import (
    default_output_dir,
    iter_file_chunks,
    load_dataframes,
    read_table,
    resolve_file_paths,
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
