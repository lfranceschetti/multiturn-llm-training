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

import json
import copy
import hydra
from trainer.dataloader import load_datasets
import deepspeed


from trainer.trainer import RefuelTrainer, DPOTrainer

torch.set_printoptions(threshold=10_000)



def set_seeds(local_seed: int) -> torch.Generator:
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed_all(local_seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    g = torch.Generator()
    g.manual_seed(local_seed)
    return g

def set_arguments(args, num_processes):

    with open_dict(args):  # Temporarily allow adding attributes
        args.world_size = num_processes
        args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
        args.batch_size = args.world_size * args.local_batch_size
        args.num_updates = args.total_episodes // args.batch_size
        
    return args



def upload_to_hf(args, model, tokenizer, accelerator, checkpoint=None):

    accelerator.wait_for_everyone()


    model_save_dir = os.path.join(args.output_dir, f"{args.exp_name}")


    if checkpoint is not None:
        model_save_dir += f"-{checkpoint}"

    hub_model_name = f"LuckyLukke/{args.exp_name}-{checkpoint}"

    os.makedirs(model_save_dir, exist_ok=True)

    if args.use_peft:

        print("Merging adapter")
        # deepspeed_plugin = accelerator.state.deepspeed_plugin
        # zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        # gather_if_zero3 = deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext
        # print(f"Using DeepSpeed Zero Stage 3 {zero_stage_3}")
        # with torch.no_grad():
        #     with gather_if_zero3(list(model.parameters())):
        #         if accelerator.is_main_process:
        #             model.push_to_hub(hub_model_name, safe_serialization=True, use_temp_dir=True)
        #             tokenizer.push_to_hub(hub_model_name, safe_serialization=True, use_temp_dir=True)
        #         model.unmerge_adapter()
        unwrapped_model = accelerator.unwrap_model(model)

        unwrapped_model.save_pretrained(
            model_save_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )

    else:
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
    g = set_seeds(local_seed)

    args = set_arguments(args, accelerator.num_processes)

    device = accelerator.device

    if args.training_method == "REFUEL":
        trainer = RefuelTrainer(args=args, accelerator=accelerator)
    elif args.training_method == "DPO":
        trainer = DPOTrainer(args=args, accelerator=accelerator)

    

    #Cudnn will us deterministic algorithms, which means that the model will always produce the same output for the same input but it may slow down the training
    torch.backends.cudnn.deterministic = True

    
    model, tokenizer = trainer.setup_model()

    dataset, validation_dataset, temp_dataloader, recompute_log = load_datasets(args)

    # MIN_LOGPROB = -1000.0
    # def filter_by_logprob(example):
    #     if "chosen_logprob" in example:
    #         # Return True if the example's chosen_logprob is above threshold
    #         return example["chosen_logprob"] > MIN_LOGPROB
    #     else:
    #         return True

    # filtered_dataset = dataset.filter(filter_by_logprob)

    # # Do the same for validation_dataset if you want:
    # filtered_validation_dataset = validation_dataset.filter(filter_by_logprob)

    # dataset = filtered_dataset
    # validation_dataset = filtered_validation_dataset

    if args.end_idx != -1:
        dataset = dataset.select(range(args.start_idx, args.end_idx))

    # Log initial information
    log_initial_info(args, accelerator, model)

    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    optimizer = trainer.setup_optimizer(model)

    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates Cosine Scheduler
    scheduler = trainer.setup_scheduler(optimizer)


    print("PREPARING DATALOADERS")

    dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True, num_workers=4, pin_memory=True, generator=g)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, generator=g)

    print("PREPARING ACCELERATOR")
    # After setting up optimizer and scheduler
    model, optimizer, scheduler, dataloader, validation_dataloader = accelerator.prepare(
        model, optimizer, scheduler, dataloader, validation_dataloader
    )

    if recompute_log:
        print("ADDING REF LOGPROBS TO DATASET")
        #Add the logprobs of the initial model to the original dataset
        dataset, validation_dataset = trainer.add_original_logprobs_to_datasets(dataset, validation_dataset, model, tokenizer)
        dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True, num_workers=4, pin_memory=True, generator=g)
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, generator=g)
        model, optimizer, scheduler, dataloader, validation_dataloader = accelerator.prepare(
            model, optimizer, scheduler, dataloader, validation_dataloader
        )
    
    data_iter = cycle(dataloader)
  
    torch.manual_seed(local_seed)  # reset the local seed again
    start_time = time.time()


    model.train()
    for update in tqdm(range(1, args.num_updates + 1), disable= not accelerator.is_main_process):
        if update == 1:
            print("STARTING TRAINING")
        else:
            # VALIDATION
            if (update - 1) % (args.print_sample_output_freq // args.batch_size) == 0 or update == args.num_updates:

                trainer.mode = "validation"
                if accelerator.is_main_process:
                    print(f"STARTING VALIDATION at update {update}")
                model.eval()
                with torch.no_grad():
                    for data in tqdm(validation_dataloader, disable= not accelerator.is_main_process):
                        new_logprobs = trainer.compute_logprobs(model, tokenizer, data, device)
                        old_logprobs = trainer.get_reference_logprobs(data)
                        loss = trainer.calculate_loss(new_logprobs, old_logprobs, data)

                trainer.wandb_logger.log_validation(step=(update-1))

                if update > 1:
                    trainer.wandb_logger.log_training(step=(update-1))
                
                del new_logprobs, old_logprobs

                model.train()
                torch.cuda.empty_cache()

        
        trainer.mode = "training"
        #TRAINING
        try:
            data = next(data_iter)
        except StopIteration:
            data_iter = cycle(dataloader)
            data = next(data_iter)

        with accelerator.accumulate(model):


            
            # Move data to the appropriate device
            data = {key: value.to(device) for key, value in data.items() if isinstance(value, torch.Tensor)}

            new_logprobs = trainer.compute_logprobs(model, tokenizer, data, device)
            old_logprobs = trainer.get_reference_logprobs(data)

            loss = trainer.calculate_loss(new_logprobs, old_logprobs, data)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss at update {update}: {loss}")
                continue

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


        #Upload the model at specified intervals
        if update % (args.upload_interval // args.batch_size) == 0:
            print(f"Uploading model at update {update}")
            upload_to_hf(args, model, tokenizer, accelerator, checkpoint=update*args.batch_size)
        
        torch.cuda.empty_cache()

    upload_to_hf(args, model, tokenizer, accelerator)

   
    trainer.wandb_logger.finalize()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()