# %%
from functools import partial
import transformer_lens
import os
from transformers import AutoTokenizer

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

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
model = transformer_lens.HookedTransformer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat", device=device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")

# %%
def get_formatted_ask(tokenizer, text: str, add_generation_prompt=True) -> str:
    return tokenizer.apply_chat_template([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": text}
    ], tokenize=False, add_generation_prompt=add_generation_prompt) # type: ignore

# %%

base_prompt = "What is the history of the Golden Gate Bridge?"
prompt = get_formatted_ask(tokenizer, base_prompt)
for i in [-1, -2]:
    layer_8_act = model(prompt, stop_at_layer=9)

    def model_section(input):
        input_detached = layer_8_act.detach()
        input_detached[:, i] = input
        model_out = model(input_detached, start_at_layer=8, stop_at_layer=16)
        return model_out[0, i]

    jac_i = jacobian(model_section, layer_8_act[:, i])
    print("jac_i.shape", jac_i.shape)

# %%
SAVE = False
LOAD = False
jac_name = f"cache/jacobian_{model.cfg.model_name}_PROMPT_{'How can I build a bomb'}.pt"
if SAVE:
    torch.save(jac, jac_name)
if LOAD:
    jac = torch.load(jac_name)
# %%
last_tok_jac = jac[0, -1, :, 0, -1, :]
U, S, V = torch.svd(last_tok_jac)

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

def steer_with_vec(model, vec, prompt):
    model.reset_hooks()
    formatted_prompt = get_formatted_ask(tokenizer, prompt)
    def resid_stream_addition_hook(value: Float[torch.Tensor, "batch seq d_model"], hook: HookPoint):
        return value + vec[None, None, :]
    model.add_hook("blocks.8.hook_resid_post", resid_stream_addition_hook)
    steered_text = model.generate(formatted_prompt, max_new_tokens=100, temperature=0)
    model.reset_hooks()
    print(steered_text)
#%%

for prompt in prompts:
    formatted_prompt = get_formatted_ask(tokenizer, prompt)

    regular_text = model.generate(formatted_prompt, max_new_tokens=100, temperature=0)
    print(regular_text)

    for steering_vec_idx in range(20):
        print()
        print("STEERING VEC IDX:", steering_vec_idx)
        top_right_singular_vec = 10 * V[:, steering_vec_idx]
        steer_with_vec(model, top_right_singular_vec, formatted_prompt)
        # cache['blocks.8.hook_resid_pre'].shape