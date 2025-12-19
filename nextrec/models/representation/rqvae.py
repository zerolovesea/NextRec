"""
Residual Quantized Variational AutoEncoder (RQ-VAE) for Generative Recommendation.

Date: created on 11/12/2025
Checkpoint: edit on 13/12/2025
Author: Yang Zhou, zyaztec@gmail.com
Source code reference:
[1] Tencent-Advertising-Algorithm-Competition-2025-Baseline
Reference:
[1] Lee et al. Autoregressive Image Generation using Residual Quantization. CVPR 2022.
[2] Zeghidour et al. SoundStream: An End-to-End Neural Audio Codec. IEEE/ACM TASLP 2021.

RQ-VAE learns hierarchical discrete representations via residual quantization.
It encodes continuous embeddings (e.g., item/user embeddings) into multi-level
semantic IDs, enabling downstream tasks like retrieval, classification, or generation.

Architecture:
  (1) Encoder: Projects input embeddings to latent space
  (2) Residual Quantizer (RQ): Multi-level vector quantization on residuals
  (3) Decoder: Reconstructs original embeddings from quantized latents
  (4) Training: Reconstruction loss + codebook/commitment loss

Key Features:
- Hierarchical semantic ID extraction for multi-level representations
- Flexible codebook initialization (K-Means, Balanced K-Means, Random)
- Balanced K-Means ensures uniform cluster distribution
- Supports shared or independent codebooks across quantization levels
- Cosine or L2 distance metrics for vector quantization

RQ-VAE 通过残差量化学习分层离散表示，将连续嵌入（如物品/用户嵌入，或者多模态嵌入）编码为
多层次语义 ID，可用于检索、分类或生成等下游任务。

架构：
  (1) 编码器：将输入嵌入映射到潜在空间
  (2) 残差量化器（RQ）：对残差进行多级向量量化
  (3) 解码器：从量化后的潜在向量重构原始嵌入
  (4) 训练：重构损失 + 码本/承诺损失

核心特性：
- 分层语义 ID 提取，实现多级别表示
- 灵活的码本初始化（K-Means、均衡 K-Means、随机）
- 均衡 K-Means 确保聚类分布均匀
- 支持跨量化层级的共享或独立码本
- 支持余弦或 L2 距离度量的向量量化
"""

from __future__ import annotations

import logging
import math
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from nextrec.basic.features import DenseFeature
from nextrec.basic.loggers import colorize, setup_logger
from nextrec.basic.model import BaseModel
from nextrec.data.batch_utils import batch_to_dict
from nextrec.utils.console import progress


def kmeans(
    data: torch.Tensor, n_clusters: int, kmeans_iters: int
) -> tuple[torch.Tensor, torch.Tensor]:

    km = KMeans(n_clusters=n_clusters, max_iter=kmeans_iters, n_init="auto")

    data_cpu = data.detach().cpu()
    np_data = data_cpu.numpy()
    km.fit(np_data)
    return torch.tensor(km.cluster_centers_), torch.tensor(km.labels_)


class BalancedKmeans(nn.Module):
    """Balanced K-Means clustering implementation.
    Ensures clusters have approximately equal number of samples.
    """

    def __init__(
        self, num_clusters: int, kmeans_iters: int, tolerance: float, device: str
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.kmeans_iters = kmeans_iters
        self.tolerance = tolerance
        self.device = device
        self.codebook: torch.Tensor | None = None  # type: ignore

    def compute_distances(self, data: torch.Tensor) -> torch.Tensor:
        if self.codebook is None:
            raise RuntimeError(
                "Codebook is not initialized before computing distances."
            )
        return torch.cdist(data, self.codebook)

    def assign_clusters(self, dist: torch.Tensor) -> torch.Tensor:
        samples_cnt = dist.shape[0]
        samples_labels = torch.empty(samples_cnt, dtype=torch.long, device=self.device)
        clusters_cnt = torch.zeros(
            self.num_clusters, dtype=torch.long, device=self.device
        )

        max_per_cluster = math.ceil(samples_cnt / self.num_clusters)

        sorted_indices = torch.argsort(dist, dim=-1)

        for i in range(samples_cnt):
            assigned = False
            for j in range(self.num_clusters):
                cluster_idx = sorted_indices[i, j]
                if clusters_cnt[cluster_idx] < max_per_cluster:
                    samples_labels[i] = cluster_idx
                    clusters_cnt[cluster_idx] += 1
                    assigned = True
                    break

            if not assigned:
                cluster_idx = torch.argmin(clusters_cnt)
                samples_labels[i] = cluster_idx
                clusters_cnt[cluster_idx] += 1

        return samples_labels

    def update_codebook(
        self, data: torch.Tensor, samples_labels: torch.Tensor
    ) -> torch.Tensor:
        new_codebook = []
        for i in range(self.num_clusters):
            cluster_data = data[samples_labels == i]
            if len(cluster_data) > 0:
                new_codebook.append(cluster_data.mean(dim=0))
            else:
                assert self.codebook is not None
                new_codebook.append(self.codebook[i])
        return torch.stack(new_codebook)

    def fit(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_emb, codebook_emb_dim = data.shape
        data = data.to(self.device)

        # initialize codebook with random samples
        # If num_emb < num_clusters, sample with replacement
        if num_emb >= self.num_clusters:
            indices = torch.randperm(num_emb)[: self.num_clusters]
            self.codebook = data[indices].clone()
        else:
            # Sample with replacement and add random noise
            indices = torch.randint(0, num_emb, (self.num_clusters,))
            self.codebook = data[indices].clone()
            self.codebook += torch.randn_like(self.codebook) * 0.01

        for _ in range(self.kmeans_iters):
            dist = self.compute_distances(data)
            samples_labels = self.assign_clusters(dist)
            _new_codebook = self.update_codebook(data, samples_labels)
            assert self.codebook is not None
            if torch.norm(_new_codebook - self.codebook) < self.tolerance:
                self.codebook = _new_codebook
                break

            self.codebook = _new_codebook

        assert self.codebook is not None
        return self.codebook, samples_labels

    def predict(self, data: torch.Tensor) -> torch.Tensor:
        data = data.to(self.device)
        dist = self.compute_distances(data)
        samples_labels = self.assign_clusters(dist)
        return samples_labels


class RQEncoder(nn.Module):
    """Encoder network for RQ-VAE."""

    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()

        self.stages = nn.ModuleList()
        in_dim = input_dim

        for out_dim in hidden_dims:
            stage = nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()
            )
            self.stages.append(stage)
            in_dim = out_dim

        self.stages.append(nn.Linear(in_dim, latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


class RQDecoder(nn.Module):
    """Decoder network for RQ-VAE."""

    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()

        self.stages = nn.ModuleList()
        in_dim = latent_dim

        for out_dim in hidden_dims:
            stage = nn.Sequential(
                nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()
            )
            self.stages.append(stage)
            in_dim = out_dim

        self.stages.append(nn.Linear(in_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        return x


# Vector Quantization
class VQEmbedding(nn.Embedding):
    """
    Vector Quantization (VQ) embedding module used in VQ-VAE / RQ-VAE.

    This module maintains a learnable codebook and maps continuous input
    vectors to their nearest codebook entries.
    It supports optional one-time codebook initialization using K-Means
    to improve training stability.

    - Codebook is initialized lazily on the first forward pass.
    - Nearest-neighbor search is performed using either L2 or cosine distance.
    - The module outputs both the quantized embeddings and their discrete
      semantic IDs (codebook indices).

    Typical input shape:
        data: Tensor of shape [N, D], where D == codebook_emb_dim

    Output:
        q: Quantized embeddings of shape [N, D]
        semantic_id: Discrete codebook indices of shape [N]
    """

    def __init__(
        self,
        num_clusters,
        codebook_emb_dim: int,
        kmeans_method: str,
        kmeans_iters: int,
        distances_method: str,
        device: str,
    ):
        super(VQEmbedding, self).__init__(num_clusters, codebook_emb_dim)

        self.num_clusters = num_clusters
        self.codebook_emb_dim = codebook_emb_dim
        self.kmeans_method = kmeans_method
        self.kmeans_iters = kmeans_iters
        self.distances_method = distances_method
        self.device = device
        self.codebook_initialized = False

    def create_codebook(self, data: torch.Tensor) -> None:

        if self.codebook_initialized:
            return

        if self.kmeans_method == "kmeans":
            codebook, _ = kmeans(data, self.num_clusters, self.kmeans_iters)
        elif self.kmeans_method == "bkmeans":
            BKmeans = BalancedKmeans(
                num_clusters=self.num_clusters,
                kmeans_iters=self.kmeans_iters,
                tolerance=1e-4,
                device=self.device,
            )
            codebook, _ = BKmeans.fit(data)
        else:
            codebook = torch.randn(self.num_clusters, self.codebook_emb_dim)
        codebook = codebook.to(self.device)
        assert codebook.shape == (self.num_clusters, self.codebook_emb_dim)
        self.weight.data.copy_(codebook)
        self.codebook_initialized = True

    @torch.no_grad()
    def compute_distances(self, data: torch.Tensor) -> torch.Tensor:

        codebook_t = self.weight.t()
        assert codebook_t.shape == (self.codebook_emb_dim, self.num_clusters)
        assert data.shape[-1] == self.codebook_emb_dim

        if self.distances_method == "cosine":
            data_norm = F.normalize(data, p=2, dim=-1)
            _codebook_t_norm = F.normalize(codebook_t, p=2, dim=0)
            distances = 1 - torch.mm(data_norm, _codebook_t_norm)
        # l2
        else:
            data_norm_sq = data.pow(2).sum(dim=-1, keepdim=True)
            codebook_t_norm_sq = codebook_t.pow(2).sum(dim=0, keepdim=True)
            distances = torch.addmm(
                data_norm_sq + codebook_t_norm_sq,
                data,
                codebook_t,
                beta=1.0,
                alpha=-2.0,
            )
        return distances

    @torch.no_grad()
    def create_semantic_id(self, data: torch.Tensor) -> torch.Tensor:

        distances = self.compute_distances(data)
        semantic_id = torch.argmin(distances, dim=-1)
        return semantic_id

    def update_emb(self, semantic_id: torch.Tensor) -> torch.Tensor:

        update_emb = super().forward(semantic_id)
        return update_emb

    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        self.create_codebook(data)
        semantic_id = self.create_semantic_id(data)
        q = super().forward(semantic_id)  # codebook lookup
        return q, semantic_id


# Residual Quantizer
class RQ(nn.Module):
    """
    Residual Quantization (RQ) module for RQ-VAE.

    This module performs multi-stage vector quantization on continuous latent
    representations using a stack of VQEmbedding codebooks. Each codebook
    quantizes the residual left by the previous quantization stage, enabling
    fine-grained approximation with multiple small codebooks.

    z_e (continuous latent)
        ↓
    q_1 = VQ_1(z_e)
    r_2 = z_e - q_1
        ↓
    q_2 = VQ_2(r_2)
    ...
        ↓
    z_q = q_1 + q_2 + ... + q_L

    Input shape:
        data: Tensor of shape [N, codebook_emb_dim]

    Output:
        zq_list: List of accumulated quantized tensors at each level
        semantic_ids: Tensor of discrete codebook indices
        rq_loss: Total RQ-VAE quantization loss
    """

    def __init__(
        self,
        num_codebooks: int,
        codebook_size: list,
        codebook_emb_dim: int,
        shared_codebook: bool,
        kmeans_method: str,
        kmeans_iters: int,
        distances_method: str,
        loss_beta: float,
        device: str,
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        assert len(self.codebook_size) == self.num_codebooks
        self.codebook_emb_dim = codebook_emb_dim
        self.shared_codebook = shared_codebook

        self.kmeans_method = kmeans_method
        self.kmeans_iters = kmeans_iters
        self.distances_method = distances_method
        self.loss_beta = loss_beta
        self.device = device

        if self.shared_codebook:
            self.shared_vq = VQEmbedding(
                self.codebook_size[0],
                self.codebook_emb_dim,
                self.kmeans_method,
                self.kmeans_iters,
                self.distances_method,
                self.device,
            )
            self.vqmodules = None
        else:
            self.shared_vq = None
            self.vqmodules = nn.ModuleList(
                [
                    VQEmbedding(
                        self.codebook_size[idx],
                        self.codebook_emb_dim,
                        self.kmeans_method,
                        self.kmeans_iters,
                        self.distances_method,
                        self.device,
                    )
                    for idx in range(self.num_codebooks)
                ]
            )

    # get vq module for specific level
    def vq(self, level: int) -> VQEmbedding:
        if self.shared_codebook:
            assert self.shared_vq is not None
            return self.shared_vq
        assert self.vqmodules is not None
        return self.vqmodules[level]

    # residual quantization process, transforms continuous data to discrete codes
    def quantize(self, data: torch.Tensor):
        r = data
        z_q = torch.zeros_like(data)

        r_in_list, q_list, zq_list, semantic_id_list = [], [], [], []

        for i in range(self.num_codebooks):
            r_in = r  # current residual
            vq = self.vq(i)  # get VQ module for current level
            q, ids = vq(r_in)  # q: quantized embedding, ids: semantic IDs

            q_st = (
                r_in + (q - r_in).detach()
            )  # **IMPORTANT** straight-through estimator, stop grad on r_in side
            z_q = z_q + q_st  # accumulate quantized embeddings
            r = r - q_st  # update residual

            r_in_list.append(r_in)
            q_list.append(q)
            zq_list.append(z_q)
            semantic_id_list.append(ids.unsqueeze(-1))

        semantic_ids = torch.cat(semantic_id_list, dim=-1)
        # zq_list: list of accumulated quantized embeddings at each level
        # r_in_list: list of residuals before quantization at each level
        # q_list: list of quantized embeddings at each level
        # semantic_ids: [N, num_codebooks] discrete codebook indices
        return zq_list, r_in_list, q_list, semantic_ids

    def rqvae_loss(
        self, r_in_list: list[torch.Tensor], q_list: list[torch.Tensor]
    ) -> torch.Tensor:
        losses = []
        for r_in, q in zip(r_in_list, q_list):
            # codebook loss: move codebook towards encoder output (stop grad on encoder side)
            codebook_loss = (q - r_in.detach()).pow(2.0).mean()

            # commitment loss: encourage encoder outputs to commit to codebook (stop grad on codebook side)
            commit_loss = (r_in - q.detach()).pow(2.0).mean()

            losses.append(codebook_loss + self.loss_beta * commit_loss)

        return torch.stack(losses).sum()

    def forward(
        self, data: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:

        zq_list, r_in_list, q_list, semantic_ids = self.quantize(data)
        rq_loss = self.rqvae_loss(r_in_list, q_list)
        return zq_list, semantic_ids, rq_loss


# RQ-VAE Model
class RQVAE(BaseModel):

    @property
    def model_name(self) -> str:
        return "RQVAE"

    @property
    def default_task(self) -> str:
        # task is unused for unsupervised training, keep a valid default for BaseModel
        return "regression"

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        latent_dim: int,
        num_codebooks: int,
        codebook_size: list,
        shared_codebook: bool,
        kmeans_method,
        kmeans_iters,
        distances_method,
        loss_beta: float,
        device: str,
        dense_features: list[DenseFeature] | None = None,
        target: str | list[str] | None = None,
        **kwargs,
    ):

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.loss_beta = loss_beta

        super().__init__(
            dense_features=dense_features,
            sparse_features=None,
            sequence_features=None,
            target=target,
            task=self.default_task,
            device=device,
            **kwargs,
        )

        self.encoder = RQEncoder(input_dim, hidden_dims, latent_dim).to(self.device)
        self.decoder = RQDecoder(latent_dim, hidden_dims[::-1], input_dim).to(
            self.device
        )
        self.rq = RQ(
            num_codebooks,
            codebook_size,
            latent_dim,
            shared_codebook,
            kmeans_method,
            kmeans_iters,
            distances_method,
            loss_beta,
            self.device,
        ).to(self.device)

    def encode(self, x: torch.Tensor) -> torch.Tensor:

        return self.encoder(x)

    def decode(self, z_vq: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:

        if isinstance(z_vq, list):
            z_vq = z_vq[-1]
        return self.decoder(z_vq)

    def compute_loss(
        self, x_hat: torch.Tensor, x_gt: torch.Tensor, rqvae_loss: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        recon_loss = F.mse_loss(x_hat, x_gt, reduction="mean")
        total_loss = recon_loss + rqvae_loss
        return recon_loss, rqvae_loss, total_loss

    def prepare_loader(
        self,
        data: DataLoader | dict | list | tuple,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
    ) -> DataLoader:
        if isinstance(data, DataLoader):
            return data
        dataloader = self.prepare_data_loader(
            data=data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        if isinstance(dataloader, tuple):
            loader = dataloader[0]
        else:
            loader = dataloader
        return cast(DataLoader, loader)

    # extract input embeddings from batch data
    def extract_embeddings(self, batch_data) -> torch.Tensor:

        batch_dict = batch_to_dict(batch_data)
        X_input, _ = self.get_input(batch_dict, require_labels=False)

        if not self.all_features:
            raise ValueError(
                "[RQVAE] dense_features are required to use fit/predict helpers."
            )
        tensors: list[torch.Tensor] = []
        for name in self.feature_names:
            if name not in X_input:
                raise KeyError(
                    f"[RQVAE] Feature '{name}' not found in input batch. Available keys: {list(X_input.keys())}"
                )
            tensors.append(X_input[name].to(self.device).float())
        if not tensors:
            raise ValueError("[RQVAE] No feature tensors found in batch.")
        init_embedding = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=-1)
        if init_embedding.shape[-1] != self.input_dim:
            raise ValueError(
                f"[RQVAE] Input dim mismatch: expected {self.input_dim}, got {init_embedding.shape[-1]}."
            )

        return init_embedding

    def init_codebook(self, train_loader: DataLoader, init_batches: int) -> None:
        cached: list[torch.Tensor] = []
        for batch_idx, batch in enumerate(train_loader):
            cached.append(self.extract_embeddings(batch))
            if batch_idx >= init_batches - 1:
                break
        if not cached:
            raise ValueError("[RQVAE] No data available for codebook initialization.")

        init_data = torch.cat(cached, dim=0)

        with torch.no_grad():
            # Encode to latent space, [num_samples, latent_dim]
            z_e = self.encode(init_data)

            r = z_e  # current residual

            for level in range(self.num_codebooks):
                vq = self.rq.vq(level)
                if not vq.codebook_initialized:
                    vq.create_codebook(r)
                q, _ = vq(r)  # quantize current residual
                r = r - q  # update residual

    def get_semantic_ids(self, x_gt: torch.Tensor) -> torch.Tensor:
        z_e = self.encode(x_gt)  # encode source input to latent space
        vq_emb_list, semantic_id_list, rqvae_loss = self.rq(z_e)
        return semantic_id_list

    def forward(
        self, x_gt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encode(x_gt)
        vq_emb_list, semantic_id_list, rqvae_loss = self.rq(z_e)
        x_hat = self.decode(vq_emb_list)
        recon_loss, rqvae_loss, total_loss = self.compute_loss(x_hat, x_gt, rqvae_loss)
        return x_hat, semantic_id_list, recon_loss, rqvae_loss, total_loss

    def fit(
        self,
        train_data: DataLoader | dict | list | tuple,
        valid_data: DataLoader | dict | list | tuple | None = None,
        epochs: int = 1,
        batch_size: int = 256,
        shuffle: bool = True,
        num_workers: int = 0,
        lr: float = 1e-3,
        init_batches: int = 3,
    ):
        """
        Train RQ-VAE.

        Args:
            train_data: Training data (DataLoader, dict, or array-like) that matches dense_features.
            valid_data: Optional validation data for monitoring loss.
            epochs: Training epochs.
            batch_size: Batch size for building DataLoader when raw data is provided.
            shuffle: Shuffle training data when constructing a DataLoader.
            num_workers: Number of DataLoader workers.
            lr: Learning rate for Adam optimizer.
            init_batches: Number of batches used to initialize the codebook.
        """
        train_loader = self.prepare_loader(
            data=train_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        valid_loader = (
            self.prepare_loader(
                data=valid_data,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
            if valid_data is not None
            else None
        )

        if not self.logger_initialized and self.is_main_process:
            setup_logger(session_id=self.session_id)
            self.logger_initialized = True

        # Minimal placeholders to satisfy BaseModel.summary when running unsupervised
        if not hasattr(self, "metrics"):
            self.metrics = ["loss"]
        if not hasattr(self, "task_specific_metrics"):
            self.task_specific_metrics = {}
        if not hasattr(self, "best_metrics_mode"):
            self.best_metrics_mode = "min"

        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()

        try:
            steps_per_epoch = len(train_loader)
            is_streaming = False
        except TypeError:
            steps_per_epoch = None
            is_streaming = True

        self.init_codebook(train_loader, init_batches=init_batches)

        if self.is_main_process:
            self.summary()
            logging.info("")
            logging.info(colorize("=" * 80, bold=True))
            logging.info(
                colorize(
                    "Start streaming training" if is_streaming else "Start training",
                    bold=True,
                )
            )
            logging.info(colorize("=" * 80, bold=True))
            logging.info("")
            logging.info(colorize(f"Model device: {self.device}", bold=True))

        for epoch in range(epochs):
            total_loss = 0.0
            step_count = 0
            if is_streaming and self.is_main_process:
                logging.info("")
                logging.info(colorize(f"Epoch {epoch + 1}/{epochs}", bold=True))
            if is_streaming:
                batch_iter = enumerate(train_loader)
            else:
                tqdm_disable = not self.is_main_process
                batch_iter = enumerate(
                    progress(
                        train_loader,
                        description=f"Epoch {epoch + 1}/{epochs}",
                        total=steps_per_epoch,
                        disable=tqdm_disable,
                    )
                )
            for _, batch in batch_iter:
                embeddings = self.extract_embeddings(batch)
                _, _, recon_loss, rqvae_loss, total_batch_loss = self(embeddings)

                optimizer.zero_grad()
                total_batch_loss.backward()
                optimizer.step()

                total_loss += total_batch_loss.item()
                step_count += 1

            denom = steps_per_epoch if steps_per_epoch is not None else step_count
            avg_loss = total_loss / max(1, denom)
            train_log = f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_loss:.4f}"

            if valid_loader is not None:
                val_total = 0.0
                val_steps = 0
                with torch.no_grad():
                    for batch in valid_loader:
                        embeddings = self.extract_embeddings(batch)
                        _, _, _, _, val_loss = self(embeddings)
                        val_total += val_loss.item()
                        val_steps += 1
                try:
                    val_denom = len(valid_loader)
                except TypeError:
                    val_denom = val_steps
                val_avg = val_total / max(1, val_denom)
                if self.is_main_process:
                    logging.info(colorize(train_log))
                    logging.info(
                        colorize(
                            f"  Epoch {epoch + 1}/{epochs} - Valid Loss: {val_avg:.4f}",
                            color="cyan",
                        )
                    )
            elif self.is_main_process:
                logging.info(colorize(train_log))

        if self.is_main_process:
            logging.info("")
            logging.info(colorize("Training finished.", bold=True))
            logging.info("")
        return self

    def predict(
        self,
        data: DataLoader | dict | list | tuple,
        batch_size: int = 256,
        num_workers: int = 0,
        return_reconstruction: bool = False,
        as_numpy: bool = True,
    ) -> torch.Tensor:
        """
        Generate semantic IDs or reconstructed embeddings.

        Args:
            data: Input data aligned with dense_features.
            batch_size: Batch size for building DataLoader when raw data is provided.
            num_workers: Number of DataLoader workers.
            return_reconstruction: If True, return reconstructed embeddings; otherwise, return semantic IDs.
            as_numpy: Whether to return a NumPy array; if False, returns a torch.Tensor on CPU.
        """
        data_loader = self.prepare_loader(
            data=data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        outputs: list[torch.Tensor] = []
        self.eval()
        with torch.no_grad():
            for batch in data_loader:
                embeddings = self.extract_embeddings(batch)
                if return_reconstruction:
                    x_hat, _, _, _, _ = self(embeddings)
                    outputs.append(x_hat.detach().cpu())
                else:
                    semantic_ids = self.get_semantic_ids(embeddings)
                    outputs.append(semantic_ids.detach().cpu())

        if outputs:
            result = torch.cat(outputs, dim=0)
        else:
            out_dim = self.input_dim if return_reconstruction else self.num_codebooks
            result = torch.empty((0, out_dim))
        return result.numpy() if as_numpy else result
