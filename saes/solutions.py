# %%

import os; os.environ["ACCELERATE_DISABLE_RICH"] = "1"
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
import matplotlib.pyplot as plt
import scipy
from pathlib import Path
from jaxtyping import Float
from typing import Optional, Union, Callable
from tqdm.auto import tqdm
from dataclasses import dataclass
from torch.distributions.categorical import Categorical

# Make sure exercises are in the path
section_dir = Path(__file__).parent
exercises_dir = section_dir.parent

from plotly_utils import imshow, line, hist
from plotly_utils_toy_models import (
    plot_features_in_2d,
    plot_features_in_Nd,
    plot_correlated_features,
    plot_feature_geometry,
    frac_active_line_plot,
)


device = t.device("cuda" if t.cuda.is_available() else "cpu")
# device = t.device("mps")

MAIN = __name__ == "__main__"





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


class Model(nn.Module):
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
        hidden = einops.einsum(
           features, self.W,
           "... instances features, instances hidden features -> ... instances hidden"
        )
        out = einops.einsum(
            hidden, self.W,
            "... instances hidden, instances hidden features -> ... instances features"
        )
        return F.relu(out + self.b_final)


    # def generate_batch(self: "Model", batch_size) -> Float[Tensor, "batch_size instances features"]:
    #     '''
    #     Generates a batch of data. We'll return to this function later when we apply correlations.
    #     '''
    #     # Generate the features, before randomly setting some to zero
    #     if self.cfg.binary:
    #         feat = t.ones((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W.device)
    #     else:
    #         feat = t.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W.device)

    #     # Generate a random boolean array, which is 1 wherever we'll keep a feature, and zero where we'll set it to zero
    #     feat_seeds = t.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W.device)
    #     feat_is_present = feat_seeds <= self.feature_probability

    #     # Create our batch from the features, where we set some to zero
    #     batch = t.where(feat_is_present, feat, 0.0)
        
    #     return batch


    def calculate_loss(
        self: "Model",
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


def generate_correlated_features(self: Model, batch_size, n_correlated_pairs) -> Float[Tensor, "batch_size instances features"]:
    '''
    Generates a batch of correlated features.
    Each output[i, j, 2k] and output[i, j, 2k + 1] are correlated, i.e. one is present iff the other is present.
    '''
    feat = t.rand((batch_size, self.cfg.n_instances, 2 * n_correlated_pairs), device=self.W.device)
    feat_set_seeds = t.rand((batch_size, self.cfg.n_instances, n_correlated_pairs), device=self.W.device)
    feat_set_is_present = feat_set_seeds <= self.feature_probability[:, [0]]
    feat_is_present = einops.repeat(feat_set_is_present, "batch instances features -> batch instances (features pair)", pair=2)
    return t.where(feat_is_present, feat, 0.0)


def generate_anticorrelated_features(self: Model, batch_size, n_anticorrelated_pairs) -> Float[Tensor, "batch_size instances features"]:
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


def generate_uncorrelated_features(self: Model, batch_size, n_uncorrelated) -> Float[Tensor, "batch_size instances features"]:
    '''
    Generates a batch of uncorrelated features.
    '''
    feat = t.rand((batch_size, self.cfg.n_instances, n_uncorrelated), device=self.W.device)
    feat_seeds = t.rand((batch_size, self.cfg.n_instances, n_uncorrelated), device=self.W.device)
    feat_is_present = feat_seeds <= self.feature_probability[:, [0]]
    return t.where(feat_is_present, feat, 0.0)


def generate_batch(self: Model, batch_size):
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


Model.generate_correlated_features = generate_correlated_features
Model.generate_anticorrelated_features = generate_anticorrelated_features
Model.generate_uncorrelated_features = generate_uncorrelated_features
Model.generate_batch = generate_batch


# ======================================================
# ! 3 - TMS: Superposition in a Privileged Basis
# ======================================================





class NeuronModel(Model):
    def __init__(
        self,
        cfg: Config,
        feature_probability: Optional[Tensor] = None,
        importance: Optional[Tensor] = None,
        device=device
    ):
        super().__init__(cfg, feature_probability, importance, device)

    def forward(
        self,
        features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        activations = F.relu(einops.einsum(
           features, self.W,
           "... instances features, instances hidden features -> ... instances hidden"
        ))
        out = F.relu(einops.einsum(
            activations, self.W,
            "... instances hidden, instances hidden features -> ... instances features"
        ) + self.b_final)
        return out
    



class NeuronComputationModel(Model):
    W1: Float[Tensor, "n_instances n_hidden n_features"]
    W2: Float[Tensor, "n_instances n_features n_hidden"]
    b_final: Float[Tensor, "n_instances n_features"]

    def __init__(
        self,
        cfg: Config,
        feature_probability: Optional[Tensor] = None,
        importance: Optional[Tensor] = None,
        device=device
    ):
        super().__init__(cfg, feature_probability, importance, device)

        del self.W
        self.W1 = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_hidden, cfg.n_features))))
        self.W2 = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_features, cfg.n_hidden))))
        self.to(device)


    def forward(
        self,
        features: Float[Tensor, "... instances features"]
    ) -> Float[Tensor, "... instances features"]:
        activations = F.relu(einops.einsum(
           features, self.W1,
           "... instances features, instances hidden features -> ... instances hidden"
        ))
        out = F.relu(einops.einsum(
            activations, self.W2,
            "... instances hidden, instances features hidden -> ... instances features"
        ) + self.b_final)
        return out
    

    def generate_batch(self, batch_size) -> Tensor:
        feat = 2 * t.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W1.device) - 1
        feat_seeds = t.rand((batch_size, self.cfg.n_instances, self.cfg.n_features), device=self.W1.device)
        feat_is_present = feat_seeds <= self.feature_probability
        batch = t.where(feat_is_present, feat, 0.0)
        return batch


    def calculate_loss(
        self,
        out: Float[Tensor, "batch instances features"],
        batch: Float[Tensor, "batch instances features"],
    ) -> Float[Tensor, ""]:
        error = self.importance * ((batch.abs() - out) ** 2)
        loss = einops.reduce(error, 'batch instances features -> instances', 'mean').sum()
        return loss
# ======================================================
# ! 5 - SAEs in Toy Models
# ======================================================

@dataclass
class AutoEncoderConfig:
    n_instances: int
    n_input_ae: int
    n_hidden_ae: int
    l1_coeff: float = 0.5
    tied_weights: bool = False


class AutoEncoder(nn.Module):
    W_enc: Float[Tensor, "n_instances n_input_ae n_hidden_ae"]
    W_dec: Float[Tensor, "n_instances n_hidden_ae n_input_ae"]
    b_enc: Float[Tensor, "n_instances n_hidden_ae"]
    b_dec: Float[Tensor, "n_instances n_input_ae"]

    def __init__(self, cfg: AutoEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.W_enc = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_input_ae, cfg.n_hidden_ae))))
        if not(cfg.tied_weights):
            self.W_dec = nn.Parameter(nn.init.xavier_normal_(t.empty((cfg.n_instances, cfg.n_hidden_ae, cfg.n_input_ae))))
        self.b_enc = nn.Parameter(t.zeros(cfg.n_instances, cfg.n_hidden_ae))
        self.b_dec = nn.Parameter(t.zeros(cfg.n_instances, cfg.n_input_ae))
        self.to(device)

    def forward(self, h: Float[Tensor, "batch_size n_instances n_hidden"]):

        # Compute activations
        h_cent = h - self.b_dec
        acts = einops.einsum(
            h_cent, self.W_enc,
            "batch_size n_instances n_input_ae, n_instances n_input_ae n_hidden_ae -> batch_size n_instances n_hidden_ae"
        )
        acts = F.relu(acts + self.b_enc)

        # Compute reconstructed input
        h_reconstructed = einops.einsum(
            acts, (self.W_enc.transpose(-1, -2) if self.cfg.tied_weights else self.W_dec),
            "batch_size n_instances n_hidden_ae, n_instances n_hidden_ae n_input_ae -> batch_size n_instances n_input_ae"
        ) + self.b_dec

        # Compute loss, return values
        l2_loss = (h_reconstructed - h).pow(2).mean(-1) # shape [batch_size n_instances]
        #l1_loss = acts.abs().sum(-1) # shape [batch_size n_instances]
        # TODO is maybe try some of the stuff below
        #l1_loss = acts.sqrt().sum(-1).square() # shape [batch_size n_instances]
        #l1_loss = acts.pow(1/4).sum(-1)
        # we are now using sum(sqrt(x)) as the loss
        l1_loss = acts.sqrt().sum()
        loss = (self.cfg.l1_coeff * l1_loss + l2_loss).mean(0).sum() # scalar

        return l1_loss, l2_loss, loss, acts, h_reconstructed

    @t.no_grad()
    def normalize_decoder(self) -> None:
        '''
        Normalizes the decoder weights to have unit norm. If using tied weights, we we assume W_enc is used for both.
        '''
        if self.cfg.tied_weights:
            self.W_enc.data = self.W_enc.data / self.W_enc.data.norm(dim=1, keepdim=True)
        else:
            self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=2, keepdim=True)

    @t.no_grad()
    def resample_neurons(
        self,
        h: Float[Tensor, "batch_size n_instances n_hidden"],
        frac_active_in_window: Float[Tensor, "window n_instances n_hidden_ae"],
        neuron_resample_scale: float,
    ) -> None:
        '''
        Resamples neurons that have been dead for `dead_neuron_window` steps, according to `frac_active`.
        '''
        _, l2_loss, _, _, _ = self.forward(h)

        # Create an object to store the dead neurons (this will be useful for plotting)
        dead_neurons_mask = t.empty((self.cfg.n_instances, self.cfg.n_hidden_ae), dtype=t.bool, device=self.W_enc.device)

        for instance in range(self.cfg.n_instances):

            # Find the dead neurons in this instance. If all neurons are alive, continue
            is_dead = (frac_active_in_window[:, instance].sum(0) < 1e-8)
            dead_neurons_mask[instance] = is_dead
            dead_neurons = t.nonzero(is_dead).squeeze(-1)
            alive_neurons = t.nonzero(~is_dead).squeeze(-1)
            n_dead = dead_neurons.numel()
            if n_dead == 0: continue
            
            # Compute L2 loss for each element in the batch
            l2_loss_instance = l2_loss[:, instance] # [batch_size]
            if l2_loss_instance.max() < 1e-6:
                continue # If we have zero reconstruction loss, we don't need to resample neurons
            
            # Draw `n_hidden_ae` samples from [0, 1, ..., batch_size-1], with probabilities proportional to l2_loss
            distn = Categorical(probs = l2_loss_instance / l2_loss_instance.sum())
            replacement_indices = distn.sample((n_dead,)) # shape [n_dead]

            # Index into the batch of hidden activations to get our replacement values
            replacement_values = (h - self.b_dec)[replacement_indices, instance] # shape [n_dead n_input_ae]

            # Get the norm of alive neurons (or 1.0 if there are no alive neurons)
            W_enc_norm_alive_mean = 1.0 if len(alive_neurons) == 0 else self.W_enc[instance, :, alive_neurons].norm(dim=0).mean().item()
            
            # Use this to renormalize the replacement values
            replacement_values = (replacement_values / (replacement_values.norm(dim=1, keepdim=True) + 1e-8)) * W_enc_norm_alive_mean * neuron_resample_scale

            # Lastly, set the new weights & biases
            self.W_enc.data[instance, :, dead_neurons] = replacement_values.T
            self.b_enc.data[instance, dead_neurons] = 0.0

        # Return data for visualising the resampling process
        colors = [["red" if dead else "black" for dead in dead_neuron_mask_inst] for dead_neuron_mask_inst in dead_neurons_mask]
        title = f"resampling {dead_neurons_mask.sum().item()}/{dead_neurons_mask.numel()} neurons (shown in red)"
        return colors, title

    def optimize(
        self,
        model: Model,
        batch_size: int = 1024,
        steps: int = 10_000,
        log_freq: int = 100,
        lr: float = 1e-3,
        lr_scale: Callable[[int, int], float] = constant_lr,
        neuron_resample_window: Optional[int] = None,
        dead_neuron_window: Optional[int] = None,
        neuron_resample_scale: float = 0.2,
    ):
        '''
        Optimizes the autoencoder using the given hyperparameters.

        This function should take a trained model as input.
        '''
        if neuron_resample_window is not None:
            assert (dead_neuron_window is not None) and (dead_neuron_window < neuron_resample_window)

        optimizer = t.optim.Adam(list(self.parameters()), lr=lr)
        frac_active_list = []
        progress_bar = tqdm(range(steps))

        # Create lists to store data we'll eventually be plotting
        data_log = {"values": [], "colors": [], "titles": [], "frac_active": []}
        colors = None
        title = "no resampling yet"

        l2_history, l1_history = [], []
        for step in progress_bar:

            # Normalize the decoder weights before each optimization step
            self.normalize_decoder()

            # Resample dead neurons
            if (neuron_resample_window is not None) and ((step + 1) % neuron_resample_window == 0):
                # Get the fraction of neurons active in the previous window
                frac_active_in_window = t.stack(frac_active_list[-neuron_resample_window:], dim=0)
                # Compute batch of hidden activations which we'll use in resampling
                batch = model.generate_batch(batch_size)
                h = einops.einsum(batch, model.W, "batch_size instances features, instances hidden features -> batch_size instances hidden")
                # Resample
                colors, title = self.resample_neurons(h, frac_active_in_window, neuron_resample_scale)

            # Update learning rate
            step_lr = lr * lr_scale(step, steps)
            for group in optimizer.param_groups:
                group['lr'] = step_lr

            # Get a batch of hidden activations from the model
            with t.inference_mode():
                features = model.generate_batch(batch_size)
                h = einops.einsum(features, model.W, "... instances features, instances hidden features -> ... instances hidden")

            # Optimize
            optimizer.zero_grad()
            l1_loss, l2_loss, loss, acts, _ = self.forward(h)
            l2_history.append(l2_loss.mean().item())
            l1_history.append(l1_loss.item())
            loss.backward()
            optimizer.step()

            # Calculate the sparsities, and add it to a list
            frac_active = einops.reduce((acts.abs() > 1e-8).float(), "batch_size instances hidden_ae -> instances hidden_ae", "mean")
            frac_active_list.append(frac_active)

            # Display progress bar, and append new values for plotting
            if step % log_freq == 0 or (step + 1 == steps):
                progress_bar.set_postfix(l1_loss=l1_loss.mean(0).sum().item(), l2_loss=l2_loss.mean(0).sum().item(), lr=step_lr)
                data_log["values"].append(self.W_enc.detach().cpu())
                data_log["colors"].append(colors)
                data_log["titles"].append(f"Step {step}/{steps}: {title}")
                data_log["frac_active"].append(frac_active.detach().cpu())
        fig = go.Figure(data=go.Scatter(y=l2_history, mode="lines", name="L2 loss"))
        fig.add_trace(go.Scatter(y=l1_history, mode="lines", name="L1 loss"))
        fig.show()

        return data_log

#-------------------- OUR CODE ---------------------#
def plot_W_and_b(W, b, show_b=False):
    with t.no_grad():
        WTW = W.T @ W
        plt.imshow(WTW.cpu())
        plt.colorbar()
        plt.show()
        if show_b:
            plt.imshow(b[None, :].cpu())
            plt.colorbar()
            plt.show()
# actually run the experiment
instances = 3
d_hidden = 2
d_features = 2
importance = 0.7
sparsity = 0.5
model_cfg = Config(instances, d_features, d_hidden, binary=True)
# importance_vec = (importance ** t.arange(0, d_features)).to(device)
importance_vec = t.ones(d_features).to(device)
model = Model(model_cfg, importance=importance_vec, feature_probability=1.0-sparsity, device=device)
model.optimize(batch_size=4096, lr=5e-3, steps=500)
# plot model weights
with t.no_grad():
    for i in range(instances):
        # print the sums of the rows
        plot_W_and_b(model.W[i], model.b_final[i])

ae_cfg = AutoEncoderConfig(3, d_hidden, 2 ** d_features, l1_coeff=0.00001) 
ae = AutoEncoder(ae_cfg)
res_ae = ae.optimize(model, neuron_resample_window=3_001, dead_neuron_window=400, steps=1_000, batch_size=8192)
# %%
for i in range(instances):
    with t.no_grad():
        ae_dec_cpu, w_i_cpu = ae.W_dec[i].cpu(), model.W[i].cpu()
        print(ae_dec_cpu.shape, w_i_cpu.shape)
        plt.imshow(ae_dec_cpu @ w_i_cpu)
        plt.show()
# %%
with t.no_grad():
    all_feature_combos = [(0, 1), (1, 0), (0, 0), (1, 1)]
    for features in all_feature_combos:
        top_feature = t.tensor(features, dtype=t.float32).to(device)
        top_feature_compressed = model.W[0] @ top_feature
        top_feature_compressed = top_feature_compressed[None, None, :]
        l1_loss, l2_loss, loss, feat_acts, reconstruction = ae(top_feature_compressed)
        print("Input features:", features)
        print("top feature compressed", top_feature_compressed, "reconstruction:", reconstruction.cpu())
        plt.imshow(feat_acts[0].cpu())
        feat_strs = [str(f) for f in features]
        plt.title(f"hidden acts of 3 SAEs when ground truth feats {','.join(feat_strs)} input")
        plt.colorbar()
        plt.show()
# %%
# okay above was visual. now let's actually try to quantify feature splitting

# set the noise_threshold just from glancing at visual results above
def calculate_l0_norm_remove_noise(x, noise_threshold=0.14):
    return (x.abs() > noise_threshold).float().sum(dim=-1)

n_features_to_use = 30 # just use the top 10 most important ones, for now. I suspect that importance kinda matters and we might want to include the same importance penalty in training the sae also.

# first let's baseline it by seeing what the feature splitting ratio is when we have a single feature
# it should be 1.0 (it's not)
# to make it more calibrated, let's give it a hand by normalizing it here and then using the same normalizing factor in the future for l0 norm


correct_l0_norm_for_single_feature = 1.0
mean_l0_norms = []
for i in range(n_features_to_use):
    with t.no_grad():
        features = [i]
        top_feature = one_hot_multi(d_features, *features).to(device)
        top_feature_compressed = model.W[0] @ top_feature
        top_feature_compressed = top_feature_compressed[None, None, :]
        l1_loss, l2_loss, loss, acts, compressed_through_sae = ae(top_feature_compressed)
        mean_l0_norms.append(calculate_l0_norm_remove_noise(acts)[0].mean().item())
        #print(f"mean l0 norm of hidden acts for feats {features}: {round(mean_l0_norms[-1], 2)} on 3 SAEs, correct l0 norm: {correct_l0_norm_for_single_feature}")
# normalizing_factor = correct_l0_norm_for_single_feature / np.mean(mean_l0_norms)
print("mean l0 norm for all single features:", np.mean(mean_l0_norms))

# now let's try for all combinations of 2 features in the top n_features_to_use
all_combos_of_2_features = [(i, j) for i in range(n_features_to_use) for j in range(i+1, n_features_to_use)]
mean_l0_norms_for_two_features = []
for features in all_combos_of_2_features:
    with t.no_grad():
        top_feature = one_hot_multi(d_features, *features).to(device)
        top_feature_compressed = model.W[0] @ top_feature
        top_feature_compressed = top_feature_compressed[None, None, :]
        l1_loss, l2_loss, loss, acts, compressed_through_sae = ae(top_feature_compressed)
        l0_norm = calculate_l0_norm_remove_noise(acts)[0].mean().item()
        mean_l0_norms_for_two_features.append(l0_norm)
        #print(f"mean l0 norm of hidden acts for feats {features}: {round(mean_l0_norms_for_two_features[-1], 2)} on 3 SAEs, correct l0 norm: {correct_l0_norm_for_single_feature}")
print("mean l0 norm for all 2 feature combos:", np.mean(mean_l0_norms_for_two_features))
# interesting, so this is actually pretty close to 1, which suggests that it is *not* doing feature splitting