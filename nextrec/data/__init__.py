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

from nextrec.data.dataloader import (
    TensorDictDataset,
    FileDataset,
    RecDataLoader,
    build_tensors_from_data,
)

from nextrec.data.preprocessor import DataProcessor
from nextrec.basic.features import FeatureSet
from nextrec.data import data_utils

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
    # DataLoader
    "TensorDictDataset",
    "FileDataset",
    "RecDataLoader",
    "build_tensors_from_data",
    # Preprocessor
    "DataProcessor",
    # Features
    "FeatureSet",
    # Legacy module
    "data_utils",
]
