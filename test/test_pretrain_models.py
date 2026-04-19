import torch

from nextrec.basic.features import DenseFeature
from nextrec.models.pretrain import BasePretrainModel, RQVAE


def test_pretrain_import_and_forward(device, monkeypatch):
    assert BasePretrainModel is not None
    monkeypatch.setattr(RQVAE, "default_task", "regression", raising=False)

    model = RQVAE(
        input_dim=4,
        hidden_dims=[8],
        latent_dim=4,
        num_codebooks=2,
        codebook_size=[4, 4],
        shared_codebook=False,
        kmeans_method="random",
        kmeans_iters=1,
        distances_method="l2",
        loss_beta=0.25,
        device=device,
        dense_features=[DenseFeature(name="dense_0", input_dim=4)],
        target="label",
    )

    batch = torch.randn(3, 4, device=model.device)
    x_hat, semantic_ids, recon_loss, rqvae_loss, total_loss = model(batch)

    assert x_hat.shape == batch.shape
    assert semantic_ids.shape == (3, 2)
    assert recon_loss.ndim == 0
    assert rqvae_loss.ndim == 0
    assert total_loss.ndim == 0
