# %%

import os
import sys
import torch as t
from torch import nn, Tensor
from torch.nn import functional as F
import time
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import einops
import plotly.express as px
import scipy
from pathlib import Path
from jaxtyping import Float
from typing import Optional, Tuple, Union, Callable
from toy_models.utils.custom_tqdm import tqdm
from dataclasses import dataclass
from torch.distributions.categorical import Categorical

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent

device = "cuda" if t.cuda.is_available() else "cpu"

# ======================================================
# ! 1 - TMS: Superposition in a nonprivileged basis
# ======================================================


def linear_lr(step, steps):
    return (1 - (step / steps))

def constant_lr(*_):
    return 1.0

def cosine_decay_lr(step, steps):
    return np.cos(0.5 * np.pi * step / (steps - 1))


@dataclass
class Config:
    # We optimize n_instances models in a single training loop to let us sweep over
    # sparsity or importance curves  efficiently. You should treat `n_instances` as
    # kinda like a batch dimension, but one which is built into our training setup.
    n_instances: int
    n_features: int = 5
    n_hidden: int = 2
    binary: bool = False
    n_correlated_pairs: int = 0
    n_anticorrelated_pairs: int = 0
    single_always_on_feature: bool = False


class ToyModel(nn.Module):
    W: Float[Tensor, "n_instances n_hidden n_features"]
    b_final: Float[Tensor, "n_instances n_features"]
    # Our linear map is x -> ReLU(W.T @ W @ x + b_final)

    def __init__(
        self,
        cfg: Config,
        feature_probability: Optional[Union[float, Tensor]] = None,
        importance: Optional[Union[float, Tensor]] = None,
        device = device,
    ):
        super().__init__()
        self.cfg = cfg

        if feature_probability is None: feature_probability = t.ones(())
        if isinstance(feature_probability, float): feature_probability = t.tensor(feature_probability)
        self.feature_probability = feature_probability.to(device).broadcast_to((cfg.n_instances, cfg.n_features))
        if importance is None: importance = t.ones(())
        if isinstance(importance, float): importance = t.tensor(importance)
        self.importance = importance.to(device).broadcast_to((cfg.n_instances, cfg.n_features))

        self.W = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features))))
        self.b_final = nn.Parameter(t.zeros((cfg.n_instances, cfg.n_features)))
        self.to(device)


    def forward(
        self,
        features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        hidden = self.generate_hidden(features)
        out = einops.einsum(
            hidden, self.W,
            "... instances hidden, instances hidden features -> ... instances features"
        )
        return F.relu(out + self.b_final)
    
    def generate_hidden(
        self,
        features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances hidden"]:
        hidden = einops.einsum(
           features, self.W,
           "... instances features, instances hidden features -> ... instances hidden"
        )
        return hidden

    def generate_batch(self,  batch_size: int, binary_override: Optional[bool] = None) -> Float[Tensor, "batch_size instances features"]:
        '''
        Generates a batch of data. We'll return to this function later when we apply correlations.
        '''
        # Generate the features, before randomly setting some to zero
        if self.cfg.binary or binary_override:
            feat = t.ones((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W.device)
        else:
            feat = t.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W.device)

        # Generate a random boolean array, which is 1 wherever we'll keep a feature, and zero where we'll set it to zero
        feat_seeds = t.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W.device)
        feat_is_present = feat_seeds <= self.feature_probability
        if self.cfg.single_always_on_feature:
            feat_is_present[:, :, 0] = 1

        # Create our batch from the features, where we set some to zero
        batch = t.where(feat_is_present, feat, 0.0)
        
        return batch


    def calculate_loss(
        self: "ToyModel",
        out: Float[Tensor, "batch instances features"],
        batch: Float[Tensor, "batch instances features"],
    ) -> Float[Tensor, ""]:
        '''
        Calculates the loss for a given batch, using this loss described in the Toy Models paper:

            https://transformer-circuits.pub/2022/toy_model/index.html#demonstrating-setup-loss

        Note, `self.importance` is guaranteed to broadcast with the shape of `out` and `batch`.
        '''
        error = self.importance * ((batch - out) ** 2)
        loss = einops.reduce(error, 'batch instances features -> instances', 'mean').sum()
        return loss


    def optimize(
        self,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
    ):
        '''
        Optimizes the model using the given hyperparameters.
        '''
        optimizer = t.optim.Adam(list(self.parameters()), lr=lr)

        progress_bar = tqdm(range(steps))

        for step in progress_bar:

            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group['lr'] = step_lr

            # Optimize
            optimizer.zero_grad()
            batch = self.generate_batch(batch_size)
            out = self(batch)
            loss = self.calculate_loss(out, batch)
            loss.backward()
            optimizer.step()

            # Display progress bar
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(loss=loss.item()/self.cfg.n_instances, lr=step_lr)


# ======================================================
# ! 2 - TMS: Correlated / Anticorrelated features
# ======================================================


def generate_correlated_features(self: ToyModel, batch_size, n_correlated_pairs) -> Float[Tensor, "batch_size instances features"]:
    '''
    Generates a batch of correlated features.
    Each output[i, j, 2k] and output[i, j, 2k + 1] are correlated, i.e. one is present iff the other is present.
    '''
    feat = t.rand((batch_size, self.cfg.n_instances, 2 * n_correlated_pairs), device=self.W.device)
    feat_set_seeds = t.rand((batch_size, self.cfg.n_instances, n_correlated_pairs), device=self.W.device)
    feat_set_is_present = feat_set_seeds <= self.feature_probability[:, [0]]
    feat_is_present = einops.repeat(feat_set_is_present, "batch instances features -> batch instances (features pair)", pair=2)
    return t.where(feat_is_present, feat, 0.0)


def generate_anticorrelated_features(self: ToyModel, batch_size, n_anticorrelated_pairs) -> Float[Tensor, "batch_size instances features"]:
    '''
    Generates a batch of anti-correlated features.
    Each output[i, j, 2k] and output[i, j, 2k + 1] are anti-correlated, i.e. one is present iff the other is absent.
    '''
    feat = t.rand((batch_size, self.cfg.n_instances, 2 * n_anticorrelated_pairs), device=self.W.device)
    feat_set_seeds = t.rand((batch_size, self.cfg.n_instances, n_anticorrelated_pairs), device=self.W.device)
    first_feat_seeds = t.rand((batch_size, self.cfg.n_instances, n_anticorrelated_pairs), device=self.W.device)
    feat_set_is_present = feat_set_seeds <= 2 * self.feature_probability[:, [0]]
    first_feat_is_present = first_feat_seeds <= 0.5
    first_feats = t.where(feat_set_is_present & first_feat_is_present, feat[:, :, :n_anticorrelated_pairs], 0.0)
    second_feats = t.where(feat_set_is_present & (~first_feat_is_present), feat[:, :, n_anticorrelated_pairs:], 0.0)
    return einops.rearrange(t.concat([first_feats, second_feats], dim=-1), "batch instances (pair features) -> batch instances (features pair)", pair=2)


def generate_uncorrelated_features(self: ToyModel, batch_size, n_uncorrelated) -> Float[Tensor, "batch_size instances features"]:
    '''
    Generates a batch of uncorrelated features.
    '''
    feat = t.rand((batch_size, self.cfg.n_instances, n_uncorrelated), device=self.W.device)
    feat_seeds = t.rand((batch_size, self.cfg.n_instances, n_uncorrelated), device=self.W.device)
    feat_is_present = feat_seeds <= self.feature_probability[:, [0]]
    return t.where(feat_is_present, feat, 0.0)


def generate_batch(self: ToyModel, batch_size):
    '''
    Generates a batch of data, with optional correslated & anticorrelated features.
    '''
    n_uncorrelated = self.cfg.n_features - 2 * self.cfg.n_correlated_pairs - 2 * self.cfg.n_anticorrelated_pairs
    data = []
    if self.cfg.n_correlated_pairs > 0:
        data.append(self.generate_correlated_features(batch_size, self.cfg.n_correlated_pairs))
    if self.cfg.n_anticorrelated_pairs > 0:
        data.append(self.generate_anticorrelated_features(batch_size, self.cfg.n_anticorrelated_pairs))
    if n_uncorrelated > 0:
        data.append(self.generate_uncorrelated_features(batch_size, n_uncorrelated))
    batch = t.cat(data, dim=-1)
    return batch


ToyModel.generate_correlated_features = generate_correlated_features
ToyModel.generate_anticorrelated_features = generate_anticorrelated_features
ToyModel.generate_uncorrelated_features = generate_uncorrelated_features
