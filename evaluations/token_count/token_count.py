import os
import random
import time

import numpy as np
import torch
from accelerate import Accelerator
from rich.pretty import pprint
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from trainer.logger import WandbLogger, log_initial_info
from itertools import cycle
from omegaconf import DictConfig, OmegaConf, open_dict
from huggingface_hub import HfApi, login
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint, load_state_dict_from_zero_checkpoint
from contextlib import nullcontext
from transformers import AutoTokenizer

import json
import copy
import hydra
from trainer.dataloader import load_datasets
import deepspeed
from datasets import load_dataset


from trainer.trainer import RefuelTrainer, DPOTrainer

torch.set_printoptions(threshold=10_000)

def tokenize_messages( tokenizer, reject_mask, chosen_token, reject_token, method="REFUEL", system_message_length=0):
    """
    Convert messages to token IDs and attention masks, with token counting.
    
    Args:
        chosen: List of chosen messages
        reject: List of reject messages
        tokenizer: Tokenizer to use
        reject_mask: Mask for reject messages
        chosen_token: Array of chosen tokens
        reject_token: Array of reject tokens
        method: String indicating the method ("REFUEL" or "DPO")
        
    Returns:
        Total token count
    """


    token_count = 0
    pad_token_id = 128256

    # Count tokens in chosen_token until we hit only pad tokens
    for i, token in enumerate(chosen_token):
        j = len(chosen_token) - i

        if chosen_token[j-1] == pad_token_id:
            # Check if all remaining tokens are pad tokens
            pass
        else:
            token_count = j
            break

    print(f"Number of the tokens that are not pad tokens: {token_count}")
    print(f"Length of the system message: {system_message_length}")
    print(f"Added chosen tokens: {token_count - system_message_length}")
    token_count = token_count - system_message_length

    # Count tokens in reject_token until we hit only pad tokens
    reject_token_count = 0
    for i, token in enumerate(reject_token):
        j = len(reject_token) - i

        if reject_token[j-1] == pad_token_id:
            # Check if all remaining tokens are pad tokens
            pass
        else:
            reject_token_count = j
            break

    if method == "REFUEL":
        # For REFUEL, we only count the tokens after the first 1 in reject_mask
        first_one_index = np.where(reject_mask == 1)[0][0]
        print(f"Reject token count: {reject_token_count}")
        print(f"First one index: {first_one_index}")
        print(f"Added reject tokens: {reject_token_count - first_one_index}")
        token_count += reject_token_count - first_one_index

    if method == "DPO":
        token_count += reject_token_count - system_message_length
    
    return token_count



def set_seeds(local_seed: int) -> torch.Generator:
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    g = torch.Generator()
    g.manual_seed(local_seed)
    return g



def main():
    method = "REFUEL"
    dataset_name = "LuckyLukke/negotio_REFUEL_onesided_preprocessed"

    accelerator = Accelerator()

    local_seed = 555134 + accelerator.process_index * 100003  # Prime
    g = set_seeds(local_seed)


    device = accelerator.device

    #Cudnn will us deterministic algorithms, which means that the model will always produce the same output for the same input but it may slow down the training
    torch.backends.cudnn.deterministic = True

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset(dataset_name + '_', split='train')
    temp_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    MIN_LOGPROB = -1000.0
    def filter_by_logprob(example):
        # Return True if the example's chosen_logprob is above threshold
        return example["chosen_logprob"] > MIN_LOGPROB


    filtered_dataset = dataset.filter(filter_by_logprob)

    dataset = filtered_dataset

    # Log initial information
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, generator=g)
    
    data_iter = cycle(dataloader)
  
    torch.manual_seed(local_seed)  # reset the local seed again

    total_token_count = 0

    system_message = []
    dataset_system_message = dataset[0]["chosen"][0]
    if type(dataset_system_message["content"]) == list:
        system_message = dataset_system_message["content"][0]
        system_message_role = dataset_system_message["role"][0]
    else:
        system_message = dataset_system_message["content"]
        system_message_role = dataset_system_message["role"]

    system_message = tokenizer.apply_chat_template([{"role": system_message_role, "content": system_message}], tokenize=True, add_generation_prompt=False)
    system_message_length = len(system_message)

    print(f"System message length: {system_message_length}")

    for update in tqdm(range(1, 8000 + 1), disable= not accelerator.is_main_process):
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = cycle(dataloader)
            data = next(data_iter)

        chosen, reject = data["chosen"], data["reject"]
        reject_mask = data["reject_mask"]

        chosen_token = data["chosen_token"]
        chosen_token = torch.stack(chosen_token).squeeze()
        chosen_token = chosen_token.numpy()
        reject_token = data["reject_token"]
        reject_token = torch.stack(reject_token).squeeze()
        reject_token = reject_token.numpy()
        reject_mask = torch.stack(reject_mask).squeeze()
        reject_mask = reject_mask.numpy()

        # Tokenize and count tokens
        token_count = tokenize_messages(
           tokenizer, reject_mask, chosen_token, reject_token, method=method, system_message_length=system_message_length
        )

        total_token_count += token_count
        
        if accelerator.is_main_process and update % 250 == 0:
            print(f"Update {update}: Total tokens processed: {total_token_count}")

if __name__ == '__main__':
    main()