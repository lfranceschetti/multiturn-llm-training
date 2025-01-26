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
from helpers.logger import WandbLogger, log_initial_info
from itertools import cycle
from omegaconf import DictConfig, OmegaConf
import json
import hydra
from trainer.dataloader import load_datasets
from trainer.trainer import RefuelTrainer, DPOTrainer

torch.set_printoptions(threshold=10_000)



def set_seeds(local_seed: int) -> None:
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)

def set_arguments(args, num_processes):

    args.world_size = num_processes

    args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    args.batch_size = args.world_size * args.local_batch_size

    #Open question: What is whitening?
    if args.refuel.whiten_rewards:
        assert (args.local_batch_size >= 8), f"Per-rank minibatch size {args.local_batch_size} is insufficient for whitening"

    args.refuel.num_updates = args.total_episodes // args.batch_size
    return args



@hydra.main(version_base=None, config_path="configs_training/experiment", config_name="${config-name}")
def main(cfg: DictConfig):

    args = cfg

    accelerator = Accelerator()

    local_seed = args.seed + accelerator.process_index * 100003  # Prime
    set_seeds(local_seed)

    args = set_arguments(args, accelerator.num_processes)

    device = accelerator.device

    if args.training_method == "REFUEL":
        trainer = RefuelTrainer(args=args)
    elif args.training_method == "DPO":
        trainer = DPOTrainer(args=args)

    #Cudnn will us deterministic algorithms, which means that the model will always produce the same output for the same input but it may slow down the training
    torch.backends.cudnn.deterministic = True

    policy, tokenizer = trainer.setup_model()

    dataset, validation_dataset, temp_dataloader, recompute_log = load_datasets(args)

    if args.end_idx != -1:
        dataset = dataset.select(range(args.start_idx, args.end_idx))

    # Log initial information
    log_initial_info(args, accelerator, policy)

    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    optimizer = trainer.setup_optimizer(accelerator, policy)

    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates Cosine Scheduler
    scheduler = trainer.setup_scheduler(accelerator, optimizer)

    torch.manual_seed(args.seed)

    print("PREPARING DATALOADERS")

    dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False)

    print("PREPARING ACCELERATOR")
    # After setting up optimizer and scheduler
    policy, optimizer, scheduler, dataloader, validation_dataloader = accelerator.prepare(
        policy, optimizer, scheduler, dataloader, validation_dataloader
    )


    if recompute_log:
        #Add the logprobs of the initial model to the original dataset
        dataset, validation_dataset = trainer.add_original_logprobs_to_datasets(dataset, validation_dataset, accelerator, policy, tokenizer)
    
    data_iter = cycle(dataloader)
  
    torch.manual_seed(local_seed)  # reset the local seed again
    start_time = time.time()

    wandb_logger = WandbLogger(args, accelerator)

    policy.train()
    training_loss = []
    training_chosen_logprobs = []
    training_reject_logprobs = []
    for update in tqdm(range(1, args.refuel.num_updates + 1), disable= not accelerator.is_main_process):

        # VALIDATION
        if (update - 1) % args.print_sample_output_freq == 0 or update == args.refuel.num_updates:
            if accelerator.is_main_process:
                print(f"STARTING VALIDATION at update {update}")
            policy.eval()
            loss_list = []
            chosen_logprobs_list = []
            reject_logprobs_list = []
            with torch.no_grad():
                for data in tqdm(validation_dataloader, disable= not accelerator.is_main_process):
                    new_logprobs = trainer.compute_logprobs(policy, tokenizer, data, device)
                    old_logprobs = torch.cat((data["chosen_logprob"], data["reject_logprob"]), dim=0)
                    loss = trainer.calculate_loss(new_logprobs, old_logprobs, data)
                    loss = torch.tensor(loss, device=device)
                    loss_list.append(loss)
                    chosen_logprobs_list.append(new_logprobs[: len(new_logprobs) // 2])
                    reject_logprobs_list.append(new_logprobs[len(new_logprobs) // 2:])

            wandb_logger.log_validation(loss_list, chosen_logprobs_list, reject_logprobs_list, step=update)

            if update > 1:
                wandb_logger.log_training(training_loss, training_chosen_logprobs, training_reject_logprobs, step=update)
            
            del loss_list, chosen_logprobs_list, reject_logprobs_list, new_logprobs, old_logprobs

            policy.train()
            torch.cuda.empty_cache()

            training_loss, training_chosen_logprobs, training_reject_logprobs = [], [], []


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
            old_logprobs = torch.cat((data["chosen_logprob"], data["reject_logprob"]), dim=0)

            loss = trainer.calculate_loss(new_logprobs, old_logprobs, data)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss at update {update}: {loss}")
                continue

            training_loss.append(loss.item())
            training_chosen_logprobs.append(new_logprobs[: len(new_logprobs) // 2])
            training_reject_logprobs.append(new_logprobs[len(new_logprobs) // 2:])

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        torch.cuda.empty_cache()

    

    if args.output_dir:
        accelerator.wait_for_everyone()
        output_dir = os.path.join(args.output_dir, wandb_logger.run_name)
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        accelerator.save_state(output_dir=output_dir)
        accelerator.wait_for_everyone()
    wandb_logger.finalize()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()