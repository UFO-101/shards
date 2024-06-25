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
from toy_models.utils.custom_tqdm import tqdm
import pandas as pd

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

manual_sae = SparseAutoencoder(3, 2)
manual_sae.encode_weight.data = t.tensor([[-1, 1], [1, -1], [1, 1]]).float()
manual_sae.decode_weight.data = t.tensor([[0, 1], [1, 0], [1, 1]]).T.float()
manual_sae.enc_bias.data = t.tensor([0.0, 0.0, -1.0]).float()
# manual_sae.enc_gate_bias.data = t.tensor([0.0, 0.0, -1.0]).float()
# manual_sae.enc_mag_bias.data = t.tensor([0, 0, -1.0]).float()
manual_sae.dec_bias.data = t.tensor([0, 0]).float()
# manual_sae.encode_weight_mag.data = t.tensor([0, 0, 0]).float()
manual_sae.to(device)

dataset = CombinationDataset(
    [
        # FeatureSet.from_default(4, sparsity=0.5),
        FeatureSet.within_range(1, feature_range=(1, 1), sparsity=0.5),
        FeatureSet.within_range(1, feature_range=(1, 1), sparsity=0.5),
    ],
    size=1000,
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
data_samples = t.stack([dataset[i] for i in range(1000)]).to(device)

#%%
recons, latents = manual_sae(data_samples)
manual_l2 = (data_samples - recons).pow(2).mean(dim=-1).mean().item()
manual_l1 = latents.mean().item()
manual_total_losses, manual_l1s  = [], []

L1_min, L1_max = 1e-6, 1e-5
L1_intervals = 5

# For each l1 lambda, compute the manual total loss
for L1_lambda in t.linspace(L1_min, L1_max, L1_intervals):
    manual_total_loss = manual_l2 + L1_lambda.item() * manual_l1
    manual_total_losses.append(manual_total_loss)
    manual_l1s.append(L1_lambda.item())

#%%

COMPUTE = False
SAVE = False
LOAD = True

n_repeats = 1
l1_lambdas, l2_losses, l1_losses, total_losses = [], [], [], []

if COMPUTE:
    # For each l1 lambda, train the SAE and compute the total loss
    for L1_lambda in tqdm(t.linspace(L1_min, L1_max, L1_intervals)):
        for _ in tqdm(range(n_repeats)):
            learned_sae = train(
                dataloader=dataloader,
                N_FEATURES=3,
                n_inputs=2,
                LR=1e-3,
                N_EPOCHS=500,
                L1_LAMBDA=L1_lambda.item(),
                DEVICE=device,
                gated=True,
                new_l1=False,
                use_correlation_loss=False,
                CORRELATION_LAMBDA=0.0,
            )
            recons, latents = learned_sae(data_samples)
            l2_error = (data_samples - recons).pow(2).mean(dim=-1)
            l1 = latents.mean()
            total_loss = l2_error.mean() + L1_lambda.item() * l1
            l1_lambdas.append(L1_lambda.item())
            l2_losses.append(l2_error.mean().item())
            l1_losses.append(l1.item())
            total_losses.append(total_loss.item())

# datetime = pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')
# filename = f'l1_lambdas_{datetime}.csv'
filename = f'2_input_l1_vs_l2/l1_lambdas.csv'
if SAVE:
    # Write l1_lambdas, l2_losses, l1_losses, total_losses to a single file
    df = pd.DataFrame({
        'l1_lambda': l1_lambdas,
        'l2_loss': l2_losses,
        'l1_loss': l1_losses,
        'total_loss': total_losses
    })
    df.to_csv(filename)

if LOAD:
    # Read from the file
    df = pd.read_csv(filename)
    l1_lambdas = df['l1_lambda']
    l2_losses = df['l2_loss']
    l1_losses = df['l1_loss']
    total_losses = df['total_loss']


#%%
# Plot a scatter plot of l2 against l1, with color showing the l1_lambda
fig = px.scatter(
    x=l1_losses,
    y=l2_losses,
    color=l1_lambdas,
    color_continuous_scale='YlOrRd',
    labels={'x': 'L1 (without coeff)', 'y': 'L2', 'color': 'L1 Lambda'},
    title='L2 Loss vs L1 Loss with varying L1 Lambda<br>(trained on 500,000 datapoints in batches of 32)'
)
# Add cross for the manual solution
fig.add_trace(
    go.Scatter(
        x=[manual_l1],
        y=[manual_l2],
        mode='markers+text',
        text=['Manual'],
        textposition='bottom center',
        marker=dict(symbol='x-thin-open', size=20, color='black'),
        showlegend=False
    )
)
fig.update_layout(
    width=800,
    height=600,
)
fig.show()

#%%
# Plot a scatter plot of total loss against l1_lambda
fig = px.scatter(
    x=l1_lambdas,
    y=total_losses,
    color=l1_losses,
    color_continuous_scale='Viridis',  # You can choose any colorscale from the list provided
    labels={'x': 'L1 Lambda', 'y': 'Total Loss', 'color': 'L1 Loss'},
    title='Total Loss vs L1 Lambda'
)
# Plot points for the manual_total_losses and manual_l1s
fig.add_trace(
    go.Scatter(
        x=manual_l1s,
        y=manual_total_losses,
        mode='markers+text',
        text=['Manual'],
        textposition='bottom center',
        marker=dict(symbol='x-thin-open', size=20, color='black'),
        showlegend=False
    )
)
fig.update_layout(
    width=800,
    height=600,
)
fig.show()


#%%

L1_lambda = 1e-3
# for sae in [manual_sae, learned_sae]:
for sae in [learned_sae]:
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
