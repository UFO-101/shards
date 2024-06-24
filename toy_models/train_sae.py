from curses import noraw
from re import A
from statistics import covariance
from typing import Callable, Optional
from toy_models.gated_sae import GatedSparseAutoencoder
from toy_models.lr_scheduler import get_scheduler
from toy_models.sae import SparseAutoencoder
import torch as t
from toy_models.toy_model import ToyModel
from toy_models.utils.custom_tqdm import tqdm
import plotly.graph_objects as go
from einops import einsum
from torch.utils.data import DataLoader


def train(
    dataloader: DataLoader,
    N_FEATURES: int,
    n_inputs: int,
    LR: float,
    N_EPOCHS: int,
    L1_LAMBDA: float,
    DEVICE: str,
    gated: bool,
    new_l1: bool = True,
    use_correlation_loss: bool = False,
    CORRELATION_LAMBDA: float = 0,
    lr_schedule: bool = False,
) -> SparseAutoencoder | GatedSparseAutoencoder:
    if gated:
        sae = GatedSparseAutoencoder(N_FEATURES, n_inputs).to(DEVICE)
    else:
        sae = SparseAutoencoder(N_FEATURES, n_inputs).to(DEVICE)
    optim = t.optim.Adam(sae.parameters(), lr=LR)
    scheduler = get_scheduler(
        scheduler_name="CosineAnnealingWarmRestarts",
        optimizer=optim,
        # training_steps=N_EPOCHS * len(embeds_dataloader),
        training_steps=N_EPOCHS,
        lr = 4e-4,
        warm_up_steps=0,
        decay_steps=0,
        lr_end=0.0,
        num_cycles=5,
    )

    step_history, mse_history, l1_history, correlation_history, loss_history, lr_history = [], [], [], [], [], []
    n_dead_feature_history = []

    step=0
    correlation_loss = t.zeros(1).to(DEVICE)
    for epoch in (pbar:=tqdm(range(N_EPOCHS))):
        for data in dataloader:
            data = data.to(DEVICE)
            # For SAEs `intermediate` is the feature act, for Gated SAEs it's `pi_gate`
            reconstruction, intermediate = sae(data)
            mses = (reconstruction - data).pow(2).sum(dim=-1)
            mse_term = mses.mean()
            if new_l1:
                l1_term = L1_LAMBDA * (intermediate * sae.decode_weight.norm(dim=0)).mean()
            else:
                l1_term = L1_LAMBDA * intermediate.mean()
            loss = mse_term + l1_term
            if use_correlation_loss:
                normalized_features = (intermediate - intermediate.mean(dim=0)) / (intermediate.std(dim=0) + 1e-6)
                normalized_features = normalized_features.unsqueeze(1)
                normalized_features_T = normalized_features.transpose(-1, -2)
                correlation_matrix = (normalized_features_T @ normalized_features) / normalized_features.shape[-1]
                self_mask = ~t.eye(correlation_matrix.shape[-1]).bool().to(DEVICE)
                correlation_matrix = correlation_matrix * self_mask
                correlation_loss = correlation_matrix.abs().mean()
                loss += CORRELATION_LAMBDA * correlation_loss

            if gated:
                ein_str = "... l, d l -> ... d"
                frozen_decode = einsum(intermediate, sae.decode_weight, ein_str)
                frozen_recons = frozen_decode + sae.dec_bias.detach()
                l_aux = (frozen_recons - data).pow(2).sum(dim=-1).mean()
                loss += l_aux
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_description(f"mse {mse_term.item():.2f} l1 {l1_term.item():.2f} loss {loss.item():.2f}, correlation {correlation_loss.item():.5f}")

            # Make the decoder weights have column-wise unit norm
            if not new_l1:
                sae.decode_weight.data /= sae.decode_weight.data.norm(dim=0)
            step += 1
            if lr_schedule:
                scheduler.step()

            dead_features = intermediate.sum(dim=0) == 0
            n_dead_features = dead_features.sum().item()
            if step % 1 == 0:
                step_history.append(step)
                mse_history.append(mse_term.item())
                l1_history.append(l1_term.mean().item())
                correlation_history.append(correlation_loss.item())
                loss_history.append(loss.item())
                lr_history.append(optim.param_groups[0]["lr"])
                n_dead_feature_history.append(n_dead_features)

        # top_k_mses, top_k_indices = mses.topk(n_dead_features)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=step_history, y=loss_history, name="Loss"))
    fig.add_trace(go.Scatter(x=step_history, y=mse_history, name="MSE"))
    fig.add_trace(go.Scatter(x=step_history, y=l1_history, name="L1"))
    fig.add_trace(go.Scatter(x=step_history, y=correlation_history, name="Correlation"))
    fig.add_trace(go.Scatter(x=step_history, y=lr_history, name="LR"))
    fig.add_trace(go.Scatter(x=step_history, y=n_dead_feature_history, name="Dead Features"))
    fig.show()
    return sae