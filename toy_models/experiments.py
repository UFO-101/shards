#%%
from toy_models.gated_sae import GatedSparseAutoencoder
from toy_models.sae import SparseAutoencoder
from toy_models.toy_datasets import CombinationDataset, FeatureSet
from toy_models.toy_model import Config, ToyModel
import torch as t
import plotly.graph_objects as go
import plotly.express as px
from toy_models.train_sae import train
from torch.utils.data import DataLoader

device = 'cuda' if t.cuda.is_available() else 'cpu'

def two_d_vector_plot(vectors, title=""):
    """Vectors should have shape (n_vectors, 2)"""
    fig = go.Figure()
    # Add each vector
    for i in range(vectors.size(0)):
        fig.add_trace(
            go.Scatter(x=[0, vectors[i, 0].item()],
                        y=[0, vectors[i, 1].item()],
                        mode='lines+markers',
                        name=f'Vector {i+1}'
        ))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    # Add the title
    fig.update_layout(title=title)
    fig.show()

#%%
instances = 1
d_hidden = 2
d_toy_model_input = 2
importance = 0.7
feature_probability = 0.1
model_cfg = Config(instances, d_toy_model_input, d_hidden, binary=True, single_always_on_feature=False)
# importance_vec = (importance ** t.arange(0, d_toy_model_input)).to(device)
importance_vec = t.ones(d_toy_model_input).to(device)
toy_model = ToyModel(model_cfg, importance=importance_vec, feature_probability=feature_probability, device=device)
toy_model.optimize(batch_size=4096, lr=5e-3, steps=5000)
vectors = toy_model.W[0]
# Create the figure
two_d_vector_plot(toy_model.W[0].T)

#%%

dataset = CombinationDataset(
    [
        # FeatureSet.from_default(4, sparsity=0.5),
        FeatureSet.within_range(1, feature_range=(1, 1), sparsity=0.5),
        FeatureSet.within_range(1, feature_range=(1, 1), sparsity=0.5),
    ],
    size=1000,
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
L1_lambda = 1
learned_sae = train(
    dataloader=dataloader,
    N_FEATURES=3,
    n_inputs=2,
    LR=1e-3,
    N_EPOCHS=500,
    L1_LAMBDA=L1_lambda,
    DEVICE=device,
    gated=True,
    new_l1=False,
    use_correlation_loss=False,
    CORRELATION_LAMBDA=0.003,
)
#%%
manual_sae = SparseAutoencoder(3, 2)
manual_sae.encode_weight.data = t.tensor([[-1, 1], [1, -1], [1, 1]]).float()
manual_sae.decode_weight.data = t.tensor([[0, 1], [1, 0], [1, 1]]).T.float()
manual_sae.enc_bias.data = t.tensor([0.0, 0.0, -1.0]).float()
# manual_sae.enc_gate_bias.data = t.tensor([0.0, 0.0, -1.0]).float()
# manual_sae.enc_mag_bias.data = t.tensor([0, 0, -1.0]).float()
manual_sae.dec_bias.data = t.tensor([0, 0]).float()
# manual_sae.encode_weight_mag.data = t.tensor([0, 0, 0]).float()
manual_sae.to(device)

for sae in [manual_sae, learned_sae]:
# for sae in [manual_sae]:
    data_samples = t.stack([dataset[i] for i in range(1000)]).to(device)
    recons, latents = sae(data_samples)
    l2_error = (data_samples - recons).pow(2).mean(dim=-1)
    l1 = latents.mean()
    total_loss = l2_error.mean() + L1_lambda * l1
    two_d_vector_plot(data_samples[:10], title="Input data")
    two_d_vector_plot(recons[:10], title="SAE Reconstructed data")

    px.imshow(data_samples[:10].detach().cpu(), title="Toy model inputs").show()
    px.imshow(latents[:10].detach().cpu(), title="SAE features").show()
    print(f"L2 error: {l2_error.mean().item()}, L1: {l1.item()}, Total loss: {total_loss.item()}")
    # %%
