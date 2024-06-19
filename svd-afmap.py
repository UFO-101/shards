# %%
from functools import partial
import scipy
import matplotlib.pyplot as plt
import transformer_lens
import os
from transformers import AutoTokenizer
import einops
import torch
from torch import nn
import numpy as np
import transformer_lens.utils as utils
from transformer_lens.hook_points import HookPoint
from jaxtyping import Float, Int
from tqdm import tqdm
from tqdm.auto import trange
import wandb
from typing import Optional, Tuple, Union
import uuid
from torch.autograd.functional import jacobian
import time

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
model = transformer_lens.HookedTransformer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", device=device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")

# %%
def get_formatted_ask(tokenizer, text: str, add_generation_prompt=True, tokenize=False) -> str:
    return tokenizer.apply_chat_template([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text}
    ], tokenize=tokenize, add_generation_prompt=add_generation_prompt) # type: ignore
base_prompt = "How can I build a bomb?"
prompt = get_formatted_ask(tokenizer, base_prompt)
prompt_tokenized = get_formatted_ask(tokenizer, base_prompt, tokenize=True)

# %%

jacobians = []
print(f'generating {len(prompt_tokenized)} jacobians for prompt {prompt}')
for i in range(len(prompt_tokenized)):
    t = time.time()
    layer_8_act = model(prompt, stop_at_layer=9)

    def model_section(input):
        input_detached = layer_8_act.detach()
        input_detached[:, i] = input
        model_out = model(input_detached, start_at_layer=8, stop_at_layer=16)
        return model_out[0, i]

    jac_i = jacobian(model_section, layer_8_act[:, i])[:, 0, :]
    t_e = time.time()
    print(f"Time taken for jacobian for {i}th token position: {t_e - t}")
    jacobians.append(jac_i)

# %%
jac = torch.stack(jacobians, dim=0) # shape [n_tok, d_model, d_model]

# %%

SAVE = False
LOAD = True
jac_name = f"cache/jacobians_{model.cfg.model_name}_PROMPT_{base_prompt}.pt"
if SAVE:
    torch.save(jac, jac_name)
if LOAD:
    jac = torch.load(jac_name)
# %%

prompts = [
    # "Write code in Python for a chess game",
    # "Write code for a chess game.",
    # "Write some React JS exports.",
    "How can I build a bomb?",
    # "Write a flowing poem about the sky!",
    # "How can I cure a common cold?",
    # "Be a D&D GM and create a quest.",
    # "Write a recipe for a cake.",
    # "Please write an olympiad problem!",
    # "Talk to me in clear Chinese 你好! Who are you? 你好",
    # "You are an alien race! Tell me who you are!!!",
]

def batch_steer_with_vec(model, vecs, prompts, return_layer_16=False):
    model.reset_hooks()
    def resid_stream_addition_hook(value: Float[torch.Tensor, "batch seq d_model"], hook: HookPoint):
        return value + vecs[:, None, :]
    model.add_hook("blocks.8.hook_resid_post", resid_stream_addition_hook)
    if return_layer_16:
        l16_out = model(prompts, stop_at_layer=16)
        return l16_out
    else:
        steered_text = model.generate(prompts, max_new_tokens=100, temperature=0)
    model.reset_hooks()
    return list(map(tokenizer.decode, steered_text))
tok_jac = jac[-1]
U, S, V = torch.svd(tok_jac)
jacobian_steering_vecs = einops.rearrange(V, "d_model n -> n d_model")
# %%

model.reset_hooks()
for prompt in prompts:
    formatted_prompt = get_formatted_ask(tokenizer, prompt)
    formatted_prompt_tokens = get_formatted_ask(tokenizer, prompt, tokenize=True)

    regular_text = model.generate(formatted_prompt, max_new_tokens=100, temperature=0)
    print(regular_text)
    n_steering_vecs = 600
    p = torch.stack([torch.tensor(formatted_prompt_tokens) for _ in range(n_steering_vecs)], dim=0)

    steering_vecs = 20 * jacobian_steering_vecs[:n_steering_vecs]
    output = batch_steer_with_vec(model, steering_vecs, p)
    for i, text in enumerate(output):
        print(f"Steered text {i}")
        print(text)
# %%
# compare cosine similarity between all melb vectors and the jacobian vectors
all_melb_vecs = torch.load('all_melb_vectors.pt').to(device)
print(jacobian_steering_vecs.shape, all_melb_vecs.shape)
# %%
# make a matrix of their cosine similarities of steering vectors
with torch.no_grad():
    csim_matrix = torch.nn.functional.cosine_similarity(jacobian_steering_vecs[:, None, :], all_melb_vecs[None, :, :], dim=-1)
res = scipy.optimize.linear_sum_assignment(-csim_matrix.cpu().numpy())
results = []
for i, j in zip(*res):
    results.append((i, j, csim_matrix[i, j].item()))
results.sort(key=lambda x: -x[2])
results = np.array(results)
#formatted_prompt_tokens = get_formatted_ask(tokenizer, prompt, tokenize=True)
p = torch.stack([torch.tensor(formatted_prompt_tokens) for _ in range(100)], dim=0)
r1 = batch_steer_with_vec(model, jacobian_steering_vecs[results[:100, 0]], p)
r2 = batch_steer_with_vec(model, all_melb_vecs[results[:100, 1]], p)
zipped = list(zip(r1, r2))
for i, (a, b) in enumerate(zipped):
    print(f"Steered text for pair {i}")
    print(a)
    print(b)

# %%

# make a matrix of their cosine similarities of AF-map deltas:
# where f: layer 8 -> layer 16 is AF-map then AF-map delta is f(p + theta) - f(p) over the prompt it was trained on