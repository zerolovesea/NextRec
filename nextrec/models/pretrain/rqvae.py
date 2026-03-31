"""
Residual Quantized Variational AutoEncoder (RQ-VAE) for Generative Recommendation.

Date: created on 11/12/2025
Checkpoint: edit on 21/03/2026
Author: Yang Zhou, zyaztec@gmail.com
Source Code Reference:
- [1] Tencent-Advertising-Algorithm-Competition-2025-Baseline
Reference:
- [1] Lee et al. Autoregressive Image Generation using Residual Quantization. CVPR 2022.
- [2] Zeghidour et al. SoundStream: An End-to-End Neural Audio Codec. IEEE/ACM TASLP 2021.

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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from nextrec.basic.features import DenseFeature
from nextrec.models.pretrain.base import BasePretrainModel


def kmeans(data: torch.Tensor, n_clusters: int, kmeans_iters: int) -> tuple[torch.Tensor, torch.Tensor]:

    km = KMeans(n_clusters=n_clusters, max_iter=kmeans_iters, n_init="auto")

    data_cpu = data.detach().cpu()
    np_data = data_cpu.numpy()
    km.fit(np_data)
    return torch.tensor(km.cluster_centers_), torch.tensor(km.labels_)


class BalancedKmeans(nn.Module):
    """Balanced K-Means clustering implementation.
    Ensures clusters have approximately equal number of samples.
    """

    def __init__(self, num_clusters: int, kmeans_iters: int, tolerance: float, device: str):
        super().__init__()
        self.num_clusters = num_clusters
        self.kmeans_iters = kmeans_iters
        self.tolerance = tolerance
        self.device = device
        self.codebook: torch.Tensor | None = None  # type: ignore

    def compute_distances(self, data: torch.Tensor) -> torch.Tensor:
        if self.codebook is None:
            raise RuntimeError("Codebook is not initialized before computing distances.")
        return torch.cdist(data, self.codebook)

    def assign_clusters(self, dist: torch.Tensor) -> torch.Tensor:
        samples_cnt = dist.shape[0]
        samples_labels = torch.empty(samples_cnt, dtype=torch.long, device=self.device)
        clusters_cnt = torch.zeros(self.num_clusters, dtype=torch.long, device=self.device)

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

    def update_codebook(self, data: torch.Tensor, samples_labels: torch.Tensor) -> torch.Tensor:
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
            stage = nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())
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
            stage = nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU())
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

            q_st = r_in + (q - r_in).detach()  # **IMPORTANT** straight-through estimator, stop grad on r_in side
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

    def rqvae_loss(self, r_in_list: list[torch.Tensor], q_list: list[torch.Tensor]) -> torch.Tensor:
        losses = []
        for r_in, q in zip(r_in_list, q_list):
            # codebook loss: move codebook towards encoder output (stop grad on encoder side)
            codebook_loss = (q - r_in.detach()).pow(2.0).mean()

            # commitment loss: encourage encoder outputs to commit to codebook (stop grad on codebook side)
            commit_loss = (r_in - q.detach()).pow(2.0).mean()

            losses.append(codebook_loss + self.loss_beta * commit_loss)

        return torch.stack(losses).sum()

    def forward(self, data: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:

        zq_list, r_in_list, q_list, semantic_ids = self.quantize(data)
        rq_loss = self.rqvae_loss(r_in_list, q_list)
        return zq_list, semantic_ids, rq_loss


# RQ-VAE Model
class RQVAE(BasePretrainModel):

    @property
    def model_name(self) -> str:
        return "RQVAE"

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
        self.decoder = RQDecoder(latent_dim, hidden_dims[::-1], input_dim).to(self.device)
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

    def forward(
        self, x_gt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.encode(x_gt)
        vq_emb_list, semantic_id_list, rqvae_loss = self.rq(z_e)
        x_hat = self.decode(vq_emb_list)
        recon_loss, rqvae_loss, total_loss = self.compute_loss(x_hat, x_gt, rqvae_loss)
        return x_hat, semantic_id_list, recon_loss, rqvae_loss, total_loss
