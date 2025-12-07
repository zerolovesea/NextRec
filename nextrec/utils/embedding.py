"""
Embedding utilities for NextRec

Date: create on 13/11/2025
Checkpoint: edit on 06/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

import numpy as np


def get_auto_embedding_dim(num_classes: int) -> int:
    """
    Calculate the dim of embedding vector according to number of classes in the category.
    Formula: emb_dim = [6 * (num_classes)^(1/4)]
    Reference: Deep & Cross Network for Ad Click Predictions.(ADKDD'17)
    """
    return int(np.floor(6 * np.power(num_classes, 0.25)))
