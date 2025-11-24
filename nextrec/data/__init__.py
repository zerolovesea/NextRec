"""
Data utilities package for NextRec

This package provides data processing and manipulation utilities.

Date: create on 13/11/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from nextrec.data.data_utils import (
    collate_fn,
    get_column_data,
    default_output_dir,
    split_dict_random,
    build_eval_candidates,
    resolve_file_paths,
    iter_file_chunks,
    read_table,
    load_dataframes,
)
from nextrec.basic.features import FeatureSpecMixin
from nextrec.data import data_utils

__all__ = [
    'collate_fn',
    'get_column_data',
    'default_output_dir',
    'split_dict_random',
    'build_eval_candidates',
    'resolve_file_paths',
    'iter_file_chunks',
    'read_table',
    'load_dataframes',
    'FeatureSpecMixin',
    'data_utils',
]
