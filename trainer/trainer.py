from typing import  Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import gather_object, DummyOptim, DummyScheduler
from datasets import load_dataset, DatasetDict, concatenate_datasets
from torch.utils.data import DataLoader
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from omegaconf import DictConfig
from trainer.utils import disable_dropout_in_model, get_cosine_schedule_with_warmup
from attr import define, field
from accelerate import Accelerator
from abc import abstractmethod
from tqdm import tqdm
from helpers.logger import WandbLogger, GRPO_Logger, REFUEL_Logger



def validate_inputs(input_ids, attention_mask):
    assert input_ids is not None, "Input IDs tensor is None."
    assert attention_mask is not None, "Attention Mask tensor is None."
    assert len(input_ids.shape) == 2, f"Expected input_ids to have 2 dimensions, got {input_ids.shape}."
    assert len(attention_mask.shape) == 2, f"Expected attention_mask to have 2 dimensions, got {attention_mask.shape}."
    assert input_ids.shape == attention_mask.shape, "Mismatch between input_ids and attention_mask shapes."
    assert input_ids.numel() > 0, "Input tensor is empty."

@define
class Trainer:
    args: DictConfig
    accelerator: Accelerator


    def __init__(self, args: DictConfig, accelerator: Accelerator):
        self.args = args
        self.accelerator = accelerator
        self.mode = "train"
        self.token_keys = ["chosen_token", "reject_token"]
        self.mask_keys = ["chosen_mask", "reject_mask"]
        self.reference_logprob_keys = ["chosen_logprob", "reject_logprob"]
        self.tokenized_logprobs = getattr(args, "tokenized_logprobs", False)

        self.wandb_logger = WandbLogger(args, accelerator)



    def setup_optimizer(self, policy: nn.Module) -> Union[torch.optim.Optimizer, DummyOptim]:
        print("Setting up optimizer...")
        optimizer_cls = (
            torch.optim.AdamW
            if self.accelerator.state.deepspeed_plugin is None
            or "optimizer" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
            else DummyOptim
        )

        optimizer = optimizer_cls(policy.parameters(), lr=self.args.lr, eps=self.args.eps, weight_decay=self.args.weight_decay)
        return optimizer


    def setup_scheduler(self, optimizer):
        print("Setting up scheduler...")
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            scheduler = get_cosine_schedule_with_warmup(optimizer, int(self.args.num_updates * self.args.warmup_ratio * self.args.world_size), self.args.num_updates * self.args.world_size)
        else:
            scheduler = DummyScheduler(
            optimizer, total_num_steps=self.args.num_updates * self.args.world_size, warmup_num_steps=int(self.args.num_updates * self.args.warmup_ratio * self.args.world_size))
        return scheduler

    def setup_model(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.args.base_model,
            padding_side='right',
            trust_remote_code=True,
        )
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        policy = AutoModelForCausalLM.from_pretrained(
            self.args.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            #"flash_attention_2" refers to a more efficient attention mechanism that can speed up training and inference. 
            #It’s a low-level optimization detail that uses a more memory- and compute-efficient algorithm for attention, if supported by the hardware and library.
            #RuntimeError: FlashAttention only supports Ampere GPUs or newer.
            # attn_implementation="flash_attention_2",
        )
        policy.config.use_cache = False
        policy.gradient_checkpointing_enable()

        disable_dropout_in_model(policy)

        # policy.gradient_checkpointing_enable()
        return policy, tokenizer
    
    def add_original_logprobs_to_datasets(self, dataset, validation_dataset, policy, tokenizer, batch_size=4, chunk_size=4):

        print("Adding original logprobs to dataset...")
        device = self.accelerator.device

        def process_and_update(dataset, chunk_size):
            num_chunks = (len(dataset) + chunk_size - 1) // chunk_size  # Calculate number of chunks
            updated_dataset = []

            with torch.no_grad():
                for i in tqdm(range(num_chunks)):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(dataset))

                    chunk = dataset.select(range(start_idx, end_idx))  # Select a chunk
                    dataloader = DataLoader(chunk, batch_size=batch_size, shuffle=False)
                    all_logprobs = []

                    for batch in dataloader:
                        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                        logprobs = self.compute_logprobs(self.accelerator.unwrap_model(policy), tokenizer, batch, device)
                        logprobs = logprobs.detach().cpu().float().tolist()

                        all_logprobs.extend(logprobs)
                        torch.cuda.empty_cache()

                    # Update chunk with logprobs
                    for i, logprob_key in enumerate(self.reference_logprob_keys):
                        log_probs = all_logprobs[i * len(chunk) : (i + 1) * len(chunk)]
                        chunk = chunk.add_column(logprob_key, log_probs)
                  
                    updated_dataset.append(chunk)

            return updated_dataset

        validation_dataset_chunks = process_and_update(validation_dataset, chunk_size)
        validation_dataset = concatenate_datasets(validation_dataset_chunks)

        train_dataset_chunks = process_and_update(dataset, chunk_size)
        dataset = concatenate_datasets(train_dataset_chunks)

        # Save updated datasets to hub if main process
        if self.accelerator.is_main_process:
            DatasetDict({"train": dataset, "test": validation_dataset}).push_to_hub(
                self.args.task.query_dataset + '_' + self.args.task.cluster
            )

        self.accelerator.wait_for_everyone()

        return dataset, validation_dataset
    
    def compute_logprobs(self, policy, tokenizer, data, device):
            results = []

            assert len(self.token_keys) == len(self.mask_keys), "Number of token_keys and mask_keys should be the same."
            
            for token_key, mask_key in zip(self.token_keys, self.mask_keys):
                tokens = data[token_key].to(device)
                masks = data[mask_key].to(device)

                attention_mask = tokens != tokenizer.pad_token_id

                #Replacing padding tokens with eos_token_id ensures that the model treats padding as if it has reached the end of the sequence
                input_ids = torch.masked_fill(tokens, ~attention_mask, tokenizer.eos_token_id)

                validate_inputs(input_ids, attention_mask)
            
                output = policy(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=False,
                )
                #The output.logits contains the unnormalized scores (logits) for each token.
                #Removes the logits corresponding to the last token in each sequence
                logits = output.logits[:, :-1]
                if self.args.task.temperature:
                    logits /= self.args.task.temperature + 1e-7


                # 1) Compute log-sum-exp across vocab at each position
                #    This is shape (B, L, 1).
                logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)

                # 2) Gather the logits corresponding to the *actual* next tokens
                #    input_ids[:, 1:] is shape (B, L)
                #    next_token_logits is (B, L, 1).
                next_token_logits = torch.gather(
                    logits, 
                    dim=2, 
                    index=input_ids[:, 1:].unsqueeze(-1)
                )

                # 3) Subtract log-sum-exp to get log-probs for those tokens
                #    shape (B, L)
                logprobs = (next_token_logits - logsumexp).squeeze(-1)


                # #all_logprobs contains the log-probabilities for every token in the vocabulary for each position in the sequence.
                # #has the shape (B, L-1, V)
                # all_logprobs = F.log_softmax(logits, dim=-1)
                # #extract the log-probabilities of the actual target tokens for each position in the sequence
                # #logprrobs afterwards has shape (B, L-1)
                # logprobs = torch.gather(all_logprobs, 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

                #Sets logprobs to 0 for tokens we want to ignore. Masks was originally (B, L). To match (B, L-1) in logprobs, you need masks[:, 1:]
                logprobs = logprobs * masks[:, 1:]

                if self.tokenized_logprobs:
                    #Resulting shape: (B, L-1)
                    pass
                else:
                    #Resulting shape: (B)
                    logprobs = logprobs.sum(-1)

                # Detach and delete tensors to free memory
                del output, logits, attention_mask, tokens, masks, input_ids, logsumexp, next_token_logits
                torch.cuda.empty_cache()

                results.append(logprobs)

            #Resulting shape: (N*B) if tokenized_logprobs is False, else (N*B, L-1)
            return torch.cat(results, dim=0)
    


    
    def get_reference_logprobs(self, data):
        results = []
        for key in self.reference_logprob_keys:
            results.append(data[key])

        #Resulting shape: (N*B) if a number is stored in each key, else (N*B, L-1)
        return torch.cat(results, dim=0)
    



    
    @abstractmethod
    def calculate_loss(self, new_logprobs, old_logprobs, data):
        pass


class RefuelTrainer(Trainer):

    def __init__(self, args, accelerator):
        super().__init__(args, accelerator)
        self.wandb_logger = REFUEL_Logger(args, accelerator)

    """Implementation of Trainer for Refuel algorithm."""
    def calculate_loss(self, new_logprobs, old_logprobs, data):
        ratio_logprob = new_logprobs - old_logprobs
        ratio_logprob = ratio_logprob[:len(new_logprobs) // 2] - ratio_logprob[len(new_logprobs) // 2:]


        if not self.args.refuel.eta:
            reg_diff = self.args.refuel.beta * ratio_logprob - (data["chosen_reward"] - data["reject_reward"])
        else:
            reg_diff = ratio_logprob - self.args.refuel.eta * (data["chosen_reward"] - data["reject_reward"])

        
        loss = (reg_diff ** 2).mean()

        if self.args.refuel.nll_term:
            loss = loss + (self.args.refuel.nll_weight * -new_logprobs[:len(new_logprobs) // 2].mean() / self.args.task.total_length)

        
        if self.mode == "training":
            self.wandb_logger.add_training_metrics(loss, chosen_logprobs=new_logprobs[:len(new_logprobs) // 2], reject_logprobs=new_logprobs[len(new_logprobs) // 2:])
        elif self.mode == "validation":
            self.wandb_logger.add_validation_metrics(loss, chosen_logprobs=new_logprobs[:len(new_logprobs) // 2], reject_logprobs=new_logprobs[len(new_logprobs) // 2:])

  
        return loss



class DPOTrainer(Trainer):

    
    """Implementation of Trainer for DPO algorithm."""
    def calculate_loss(self, new_logprobs, old_logprobs, data):
        # Separate chosen (winning) and rejected (losing) log probabilities
        pi_w = new_logprobs[:len(new_logprobs) // 2]      # π*(a^w | s^w)
        pi_l = new_logprobs[len(new_logprobs) // 2:]      # π*(a^l | s^l)
        pi_ref_w = old_logprobs[:len(old_logprobs) // 2]  # π_ref(a^w | s^w)
        pi_ref_l = old_logprobs[len(old_logprobs) // 2:]  # π_ref(a^l | s^l)

        # Compute log ratios for chosen (winning) and rejected (losing) trajectories
        log_ratio_chosen = pi_w - pi_ref_w  # log(π*(a^w | s^w) / π_ref(a^w | s^w))
        log_ratio_rejected = pi_l - pi_ref_l  # log(π*(a^l | s^l) / π_ref(a^l | s^l))

        # Compute the argument for the sigmoid function
        beta = self.args.dpo.beta
        sigmoid_arg = beta * (log_ratio_chosen.sum() - log_ratio_rejected.sum())

        # Optional offset adjustment
        if self.args.dpo.with_offset:
            sigmoid_arg = sigmoid_arg - (data["chosen_reward"] - data["reject_reward"])

        # Apply the sigmoid function
        sigmoid = torch.sigmoid(sigmoid_arg)

        # Compute the DPO loss
        loss = -torch.log(sigmoid).mean()

        return loss




class GRPOTrainer(Trainer):

    def __init__(self, args: DictConfig, accelerator: Accelerator):
        super().__init__(args, accelerator)
        self.token_keys =  ["token"]
        self.reward_key = "payoff"
        self.mask_keys =  ["mask"]
        self.reference_logprob_keys = ["logprob"]
        self.tokenized_logprobs = True
        self.wandb_logger = GRPO_Logger(args, accelerator)



    
    """Implementation of Trainer for GRPO algorithm."""
    def calculate_loss(self, new_logprobs, ref_logprobs, data):

        #Input shape of the logprobs (B*G, L-1)
        #I need to ensure with the choice of data that always G data points have the same original prompt.
        #Then I can send in B different packages like this
        #So i need the batch_size to be B*G
        #I can start out with B=1, G=8
        #This can easily be implemented if I just use the same prompt for all generations (always the same starting agent)

        G = self.args.grpo.G

        per_token_kl = torch.exp(ref_logprobs - new_logprobs) - (ref_logprobs - new_logprobs) - 1

        #(B = original number of prompts x G = number of generations per prompt).
        rewards = data[self.reward_key]

        # Compute grouped-wise rewards (B, G) and take mean -> (B)
        mean_grouped_rewards = rewards.view(-1, G).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, G).std(dim=1)

        
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(G, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(G, dim=0)
        #Shape (B x G)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        
        if self.args.grpo.online:
            #(B, L-1)
            # x - x.detach() allows for preserving gradients from x
            prob_ratio = torch.exp(new_logprobs - new_logprobs.detach())
        else:
            #(B, L-1)
            prob_ratio = torch.exp(new_logprobs - ref_logprobs)


        logprob_diff = new_logprobs - ref_logprobs
        print("Logprob differences:", logprob_diff.min().item(), logprob_diff.max().item(), logprob_diff.mean().item())
        print("prob_ratio min/max/mean:", prob_ratio.min().item(), prob_ratio.max().item(), prob_ratio.mean().item())
        print(new_logprobs)
        print(ref_logprobs)


        per_token_loss = prob_ratio * advantages.unsqueeze(1)

        masks = data[self.mask_keys[0]]


        per_token_loss = -(per_token_loss - self.args.grpo.beta * per_token_kl)
        loss = ((per_token_loss * masks[:, 1:]).sum(dim=1) / masks[:, 1:].sum(dim=1)).clamp(min=1).mean()

        #Logging
        mean_kl  = ((per_token_kl * masks[:, 1:]).sum(dim=1) / masks[:, 1:].sum(dim=1)).mean()
        reward_mean = rewards.mean()
        reward_std = std_grouped_rewards.mean()
        prob_ratio_mean = prob_ratio.mean()
        if self.mode == "training":
            self.wandb_logger.add_training_metrics(loss.detach(), reward_mean, reward_std, mean_kl, prob_ratio_mean)
        elif self.mode == "validation":
            self.wandb_logger.add_validation_metrics(loss.detach(), reward_mean, reward_std, mean_kl, prob_ratio_mean)

        return loss
    
    