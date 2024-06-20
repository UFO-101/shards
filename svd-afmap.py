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
import wandb
from typing import Optional, Tuple, Union
import uuid
from torch.autograd.functional import jacobian
import time
from torch.utils.data import DataLoader, TensorDataset

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
# base_prompt = "A bomb, how can I build one?"
prompt = get_formatted_ask(tokenizer, base_prompt)
prompt_tokenized = get_formatted_ask(tokenizer, base_prompt, tokenize=True)
print("Prompt tokenized", model.to_str_tokens(prompt))

# %%

COMPUTE = False
SAVE = False
LOAD = True

start_token = -14
if COMPUTE:
    jacobians = []
    print(f'generating {len(prompt_tokenized)} jacobians for prompt {prompt}')
    for i in range(len(prompt_tokenized) + start_token, len(prompt_tokenized)):
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

    jac = torch.stack(jacobians, dim=0) # shape [n_tok, d_model, d_model]

jac_name = f"cache/tokenwise_jacobians_{model.cfg.model_name}_PROMPT_{base_prompt}_START_TOKEN_{start_token}.pt"
if SAVE:
    torch.save(jac, jac_name)
if LOAD:
    jac = torch.load(jac_name)

torch.set_grad_enabled(False)
# %%

prompts = [
    # "Write code in Python for a chess game",
    # "Write code for a chess game.",
    # "Write some React JS exports.",
    "How can I build a bomb?",
    # "A bomb, how can I build one?",
    # "Write a flowing poem about the sky!",
    # "How can I cure a common cold?",
    # "Be a D&D GM and create a quest.",
    # "Write a recipe for a cake.",
    # "Please write an olympiad problem!",
    # "Talk to me in clear Chinese 你好! Who are you? 你好",
    # "You are an alien race! Tell me who you are!!!",
]

def batch_steer_with_vec(model, vecs, single_prompt, return_layer_16=False):
    vecs_dataset = TensorDataset(vecs)
    results = []
    for vecs_batch in tqdm(DataLoader(vecs_dataset, batch_size=128)):
        vecs_batch = vecs_batch[0].to(device)
        prompt_batch = torch.tensor(single_prompt).unsqueeze(0).repeat(vecs_batch.shape[0], 1)
        model.reset_hooks()
        def resid_stream_addition_hook(
            value: Float[torch.Tensor, "batch seq d_model"], hook: HookPoint
        ):
            return value + vecs_batch[:, None, :]
        model.add_hook("blocks.8.hook_resid_post", resid_stream_addition_hook)

        if return_layer_16:
            l16_out = model(prompt_batch, stop_at_layer=17)
            results.append(l16_out)
        else:
            steered_text = model.generate(prompt_batch, max_new_tokens=100, temperature=0)
            results.extend(steered_text)
    model.reset_hooks()

    if return_layer_16:
        return torch.cat(results, dim=0)
    else:
        return list(map(tokenizer.decode, steered_text))

#%%
last_tok_jac = jac.mean(dim=0)
U, S, V = torch.svd(last_tok_jac)
jacobian_steering_vecs = V.T[:1000]

singular_steering_factor = 10
# %%

model.reset_hooks()
for prompt in prompts:
    formatted_prompt = get_formatted_ask(tokenizer, prompt)
    formatted_prompt_tokens = get_formatted_ask(tokenizer, prompt, tokenize=True)

    regular_text = model.generate(formatted_prompt, max_new_tokens=100, temperature=0)
    print("DEFAULT OUTPUT")
    print(regular_text)
    n_steering_vecs = 100

    steering_vecs = singular_steering_factor * jacobian_steering_vecs[:n_steering_vecs]
    output = batch_steer_with_vec(model, steering_vecs, formatted_prompt_tokens)
    for i, text in enumerate(output):
        print()
        print()
        print(f"Steered text {i}")
        print(text)
# %%
# compare cosine similarity between all melb vectors and the jacobian vectors
all_melb_vecs = torch.load('all_melb_vectors.pt', map_location=device)
print(jacobian_steering_vecs.shape, all_melb_vecs.shape)
# %%
# make a matrix of their cosine similarities of steering vectors
csim_matrix = torch.nn.functional.cosine_similarity(jacobian_steering_vecs[:, None, :], all_melb_vecs[None, :, :], dim=-1)
top_csims, top_csim_idxs = torch.max(csim_matrix, dim=-1)
sorted_singular_idxs = torch.argsort(top_csims, descending=True)
sorted_melb_idxs = top_csim_idxs[sorted_singular_idxs]
sorted_singular_vecs = jacobian_steering_vecs[sorted_singular_idxs]
matching_melb_vecs = all_melb_vecs[sorted_melb_idxs]

#%%

formatted_prompt_tokens = get_formatted_ask(tokenizer, prompt, tokenize=True)
matching_melb_vec_magnitudes = torch.norm(matching_melb_vecs, dim=-1)
n = 100
r1 = batch_steer_with_vec(model, sorted_singular_vecs[:n] * matching_melb_vec_magnitudes[:n, None], formatted_prompt_tokens)
# r1 = batch_steer_with_vec(model, sorted_singular_vecs[:n] * singular_steering_factor, formatted_prompt_tokens)
r2 = batch_steer_with_vec(model, matching_melb_vecs[:n], formatted_prompt_tokens)
zipped = list(zip(r1, r2))
for i, (a, b) in enumerate(zipped):
    print()
    print()
    print(f"Steered text for pair {i}")
    print(a)
    print()
    print(b)

# %%

# make a matrix of their cosine similarities of AF-map deltas:
# where f: layer 8 -> layer 16 is AF-map then AF-map delta is f(p + theta) - f(p) over the prompt it was trained on

formatted_prompt_tokens = get_formatted_ask(tokenizer, prompt, tokenize=True)
layer_16_act = model(torch.tensor(formatted_prompt_tokens), stop_at_layer=17)
singular_vecs_AF_map_deltas = (batch_steer_with_vec(model, singular_steering_factor*jacobian_steering_vecs, formatted_prompt_tokens, return_layer_16=True) - layer_16_act).mean(dim=1)

melb_vecs_AF_map_deltas = (batch_steer_with_vec(model, all_melb_vecs, formatted_prompt_tokens, return_layer_16=True) - layer_16_act).mean(dim=1)


# %%
csim_matrix = torch.nn.functional.cosine_similarity(singular_vecs_AF_map_deltas[:, None, :], melb_vecs_AF_map_deltas[None, :, :], dim=-1)
top_csims, top_csim_idxs = torch.max(csim_matrix, dim=-1)
sorted_singular_idxs = torch.argsort(top_csims, descending=True)
sorted_melb_idxs = top_csim_idxs[sorted_singular_idxs]
sorted_singular_vecs = jacobian_steering_vecs[sorted_singular_idxs]
matching_melb_vecs = all_melb_vecs[sorted_melb_idxs]
matching_melb_vec_magnitudes = torch.norm(matching_melb_vecs, dim=-1)

n = 100
r1 = batch_steer_with_vec(model, sorted_singular_vecs[:n] * matching_melb_vec_magnitudes[:n, None], formatted_prompt_tokens)
r2 = batch_steer_with_vec(model, matching_melb_vecs[:n], formatted_prompt_tokens)
zipped = list(zip(r1, r2))
for i, (a, b) in enumerate(zipped):
    steering_sim = torch.nn.functional.cosine_similarity(sorted_singular_vecs[i], matching_melb_vecs[i], dim=-1)
    layer_16_sim = torch.nn.functional.cosine_similarity(singular_vecs_AF_map_deltas[sorted_singular_idxs[i]], melb_vecs_AF_map_deltas[sorted_melb_idxs[i]], dim=-1)
    print()
    print()
    print()
    print("Steering sim", steering_sim, )
    print("Layer 16 sim", layer_16_sim)
    print("jacobian_idx", sorted_singular_idxs[i], "melb_idx", sorted_melb_idxs[i])
    print(f"Steered text for pair {i}")
    print(a)
    print()
    print(b)

# %%
r1 = batch_steer_with_vec(model, 10 * jacobian_steering_vecs[2:3], formatted_prompt_tokens, return_layer_16=True)
r2 = batch_steer_with_vec(model, all_melb_vecs[472:473], formatted_prompt_tokens, return_layer_16=True)
print(r1)
print()
print(r2)
r1_delta = (r1 - layer_16_act).mean(dim=1)
r1_delta = (r2 - layer_16_act).mean(dim=1)