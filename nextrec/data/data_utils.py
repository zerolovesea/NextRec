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
    get_group_ids,
    has_column,
    split_dict_random,
    to_column_names,
)
from nextrec.utils.data import (
    get_file_paths,
    iter_file_chunks,
    read_table,
)

__all__ = [
    # Batch utilities
    "collate_fn",
    "batch_to_dict",
    "stack_section",
    # Data processing
    "get_column_data",
    "to_column_names",
    "has_column",
    "split_dict_random",
    "build_eval_candidates",
    "get_group_ids",
    # File utilities
    "get_file_paths",
    "iter_file_chunks",
    "read_table",
]
