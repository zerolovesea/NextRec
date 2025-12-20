"""
Utilities package for NextRec

Date: create on 13/11/2025
Last update: 19/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

from . import console, data, embedding, torch_utils
from .config import (
    build_feature_objects,
    build_model_instance,
    extract_feature_groups,
    load_model_class,
    register_processor_features,
    resolve_path,
    select_features,
)
from .console import (
    display_metrics_table,
    get_nextrec_version,
    log_startup_info,
    progress,
)
from .data import (
    default_output_dir,
    generate_distributed_ranking_data,
    generate_match_data,
    generate_multitask_data,
    generate_ranking_data,
    iter_file_chunks,
    load_dataframes,
    read_table,
    read_yaml,
    resolve_file_paths,
)
from .embedding import get_auto_embedding_dim
from .feature import normalize_to_list
from .model import compute_pair_scores, get_mlp_output_dim, merge_features
from .torch_utils import (
    add_distributed_sampler,
    concat_tensors,
    configure_device,
    gather_numpy,
    get_device_info,
    get_initializer,
    get_optimizer,
    get_scheduler,
    init_process_group,
    pad_sequence_tensors,
    resolve_device,
    stack_tensors,
    to_tensor,
)

__all__ = [
    # Console utilities
    "get_nextrec_version",
    "log_startup_info",
    "progress",
    "display_metrics_table",
    # Optimizer & Scheduler (torch utils)
    "get_optimizer",
    "get_scheduler",
    # Initializer (torch utils)
    "get_initializer",
    # Embedding utilities
    "get_auto_embedding_dim",
    # Device utilities (torch utils)
    "resolve_device",
    "get_device_info",
    "configure_device",
    "init_process_group",
    "gather_numpy",
    "add_distributed_sampler",
    # Tensor utilities
    "to_tensor",
    "stack_tensors",
    "concat_tensors",
    "pad_sequence_tensors",
    # Data utilities
    "resolve_file_paths",
    "read_table",
    "read_yaml",
    "load_dataframes",
    "iter_file_chunks",
    "default_output_dir",
    # Model utilities
    "merge_features",
    "get_mlp_output_dim",
    "compute_pair_scores",
    # Feature utilities
    "normalize_to_list",
    # Config utilities
    "resolve_path",
    "register_processor_features",
    "build_feature_objects",
    "extract_feature_groups",
    "select_features",
    "load_model_class",
    "build_model_instance",
    # Synthetic data utilities
    "generate_ranking_data",
    "generate_match_data",
    "generate_multitask_data",
    "generate_distributed_ranking_data",
    # Module exports
    "console",
    "data",
    "embedding",
    "torch_utils",
]
