"""
Embedding utilities for NextRec

Date: create on 13/11/2025
Checkpoint: edit on 06/12/2025
Author: Yang Zhou, zyaztec@gmail.com
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def get_auto_embedding_dim(num_classes: int) -> int:
    """
    Calculate the dim of embedding vector according to number of classes in the category.
    Formula: emb_dim = [6 * (num_classes)^(1/4)]
    Reference: Deep & Cross Network for Ad Click Predictions.(ADKDD'17)
    """
    return int(np.floor(6 * np.power(num_classes, 0.25)))


# encode multi-modal item contents into dense embeddings
def encode_multimodel_content(
    texts: list[str], model_name: str, device: str, batch_size: int = 32
) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device)
    encoder.eval()

    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = encoder(**encoded)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu()
        embeddings.append(cls_emb)
    return torch.cat(embeddings, dim=0)
