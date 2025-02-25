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
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

import json
import copy
import hydra
from trainer.dataloader import load_datasets

from trainer.trainer import RefuelTrainer, DPOTrainer, GRPOTrainer

torch.set_printoptions(threshold=10_000)



def set_seeds(local_seed: int) -> None:
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)

def set_arguments(args, num_processes):

    args.world_size = num_processes

    args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    args.batch_size = args.world_size * args.local_batch_size

    with open_dict(args):  # Temporarily allow adding attributes
        args.num_updates = args.total_episodes // args.batch_size
    return args



def upload_to_hf(args, model, tokenizer, accelerator, checkpoint=None):

    accelerator.wait_for_everyone()


    model_save_dir = os.path.join(args.output_dir, f"{args.exp_name}")


    if checkpoint is not None:
        model_save_dir += f"-{checkpoint}"

    os.makedirs(model_save_dir, exist_ok=True)

        
    accelerator.save_state(output_dir=model_save_dir)

    # https://huggingface.co/docs/accelerate/usage_guides/deepspeed#saving-and-loading

    # unwrapped_model = accelerator.unwrap_model(model)

    # fp32_model = load_state_dict_from_zero_checkpoint(unwrapped_model, checkpoint_dir)


    accelerator.wait_for_everyone()





@hydra.main(version_base=None, config_path="configs_training/experiment", config_name="${config-name}")
def main(cfg: DictConfig):

    args = cfg

    accelerator = Accelerator()

    local_seed = args.seed + accelerator.process_index * 100003  # Prime
    set_seeds(local_seed)

    args = set_arguments(args, accelerator.num_processes)

    device = accelerator.device

    if args.training_method == "REFUEL":
        trainer = RefuelTrainer(args=args, accelerator=accelerator)
    elif args.training_method == "DPO":
        trainer = DPOTrainer(args=args, accelerator=accelerator)
    elif args.training_method == "GRPO":
        trainer = GRPOTrainer(args=args, accelerator=accelerator)
    

    #Cudnn will us deterministic algorithms, which means that the model will always produce the same output for the same input but it may slow down the training
    torch.backends.cudnn.deterministic = True

    policy, tokenizer = trainer.setup_model()

    dataset, validation_dataset, temp_dataloader, recompute_log = load_datasets(args)

    MIN_LOGPROB = -500  # example threshold

    def filter_by_logprob(example):
        # Return True if the example's chosen_logprob is above threshold
        return example["chosen_logprob"] > MIN_LOGPROB

    filtered_dataset = dataset.filter(filter_by_logprob)

    # Do the same for validation_dataset if you want:
    filtered_validation_dataset = validation_dataset.filter(filter_by_logprob)

    if args.end_idx != -1:
        dataset = dataset.select(range(args.start_idx, args.end_idx))

    # Log initial information
    log_initial_info(args, accelerator, policy)

    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    optimizer = trainer.setup_optimizer(policy)

    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates Cosine Scheduler
    scheduler = trainer.setup_scheduler(optimizer)

    torch.manual_seed(args.seed)

    print("PREPARING DATALOADERS")

    dataloader = DataLoader(filtered_dataset, batch_size=args.local_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation_dataloader = DataLoader(filtered_validation_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False)

    print("PREPARING ACCELERATOR")
    # After setting up optimizer and scheduler
    policy, optimizer, scheduler, dataloader, validation_dataloader = accelerator.prepare(
        policy, optimizer, scheduler, dataloader, validation_dataloader
    )


    if recompute_log:
        #Add the logprobs of the initial model to the original dataset
        dataset, validation_dataset = trainer.add_original_logprobs_to_datasets(dataset, validation_dataset, policy, tokenizer)
    
    data_iter = cycle(dataloader)
  
    torch.manual_seed(local_seed)  # reset the local seed again
    start_time = time.time()


    policy.train()
    for update in tqdm(range(1, args.num_updates + 1), disable= not accelerator.is_main_process):

        # VALIDATION
        if (update - 1) % (args.print_sample_output_freq // args.batch_size) == 0 or update == args.num_updates:

            trainer.mode = "validation"
            if accelerator.is_main_process:
                print(f"STARTING VALIDATION at update {update}")
            policy.eval()
            with torch.no_grad():
                for data in tqdm(validation_dataloader, disable= not accelerator.is_main_process):
                    new_logprobs = trainer.compute_logprobs(policy, tokenizer, data, device)
                    old_logprobs = trainer.get_reference_logprobs(data)
                    loss = trainer.calculate_loss(new_logprobs, old_logprobs, data)

            trainer.wandb_logger.log_validation(step=(update-1))

            if update > 1:
                trainer.wandb_logger.log_training(step=(update-1))
            
            del new_logprobs, old_logprobs

            policy.train()
            torch.cuda.empty_cache()

        #TRAINING
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = cycle(dataloader)
            data = next(data_iter)

        with accelerator.accumulate(policy):
            
            # Move data to the appropriate device
            data = {key: value.to(device) for key, value in data.items() if isinstance(value, torch.Tensor)}

            new_logprobs = trainer.compute_logprobs(policy, tokenizer, data, device)
            old_logprobs = trainer.get_reference_logprobs(data)

            loss = trainer.calculate_loss(new_logprobs, old_logprobs, data)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss at update {update}: {loss}")
                continue

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


        #Upload the model at specified intervals
        if update % (args.upload_interval // args.batch_size) == 0:
            upload_to_hf(args, policy, tokenizer, accelerator, checkpoint=update*args.batch_size)
        
        torch.cuda.empty_cache()

    upload_to_hf(args, policy, tokenizer, accelerator)

   
    trainer.wandb_logger.finalize()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()