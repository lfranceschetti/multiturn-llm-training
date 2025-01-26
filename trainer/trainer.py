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

    def setup_optimizer(self, accelerator: Accelerator, policy: nn.Module) -> Union[torch.optim.Optimizer, DummyOptim]:
        print("Setting up optimizer...")
        optimizer_cls = (
            torch.optim.AdamW
            if accelerator.state.deepspeed_plugin is None
            or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
            else DummyOptim
        )

        optimizer = optimizer_cls(policy.parameters(), lr=self.args.lr, eps=self.args.eps, weight_decay=self.args.weight_decay)
        return optimizer


    def setup_scheduler(self, accelerator, optimizer):
        print("Setting up scheduler...")
        if (
            accelerator.state.deepspeed_plugin is None
            or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            scheduler = get_cosine_schedule_with_warmup(optimizer, int(self.args.refuel.num_updates * self.args.warmup_ratio * self.args.world_size), self.args.refuel.num_updates * self.args.world_size)
        else:
            scheduler = DummyScheduler(
            optimizer, total_num_steps=self.args.refuel.num_updates * self.args.world_size, warmup_num_steps=int(self.args.refuel.num_updates * self.args.warmup_ratio * self.args.world_size))
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
        disable_dropout_in_model(policy)

        # policy.gradient_checkpointing_enable()
        return policy, tokenizer
    
    def add_original_logprobs_to_datasets(self, dataset, validation_dataset, accelerator, policy, tokenizer, batch_size=4, chunk_size=4):

        print("Adding original logprobs to dataset...")
        device = accelerator.device

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
                        logprobs = self.compute_logprobs(accelerator.unwrap_model(policy), tokenizer, batch, device)
                        all_logprobs.extend(logprobs)
                        torch.cuda.empty_cache()

                    # Update chunk with logprobs
                    chosen_logprobs = [float(x.cpu().item()) for x in all_logprobs[: len(chunk)]]
                    reject_logprobs = [float(x.cpu().item()) for x in all_logprobs[len(chunk):]]

                    chunk = chunk.add_column("chosen_logprob", chosen_logprobs)
                    chunk = chunk.add_column("reject_logprob", reject_logprobs)
                    updated_dataset.append(chunk)

            return updated_dataset

        validation_dataset_chunks = process_and_update(validation_dataset, chunk_size)
        validation_dataset = concatenate_datasets(validation_dataset_chunks)

        train_dataset_chunks = process_and_update(dataset, chunk_size)
        dataset = concatenate_datasets(train_dataset_chunks)

        # Save updated datasets to hub if main process
        if accelerator.is_main_process:
            DatasetDict({"train": dataset, "test": validation_dataset}).push_to_hub(
                self.args.task.query_dataset + '_' + self.args.task.cluster
            )

        accelerator.wait_for_everyone()

        return dataset, validation_dataset
    
    def compute_logprobs(self, policy, tokenizer, data, device):
            results = []
            for token_key, mask_key in [("chosen_token", "chosen_mask"), ("reject_token", "reject_mask")]:
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
                logits /= self.args.task.temperature + 1e-7

                #all_logprobs contains the log-probabilities for every token in the vocabulary for each position in the sequence.
                #has the shape (batch_size, sequence_length - 1, vocab_size)
                all_logprobs = F.log_softmax(logits, dim=-1)
                #extract the log-probabilities of the actual target tokens for each position in the sequence
                #logprrobs afterwards has shape (batch_size, sequence_length - 1)
                logprobs = torch.gather(all_logprobs, 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

                #Sets logprobs to 0 for padding tokens and sums the log-probabilities for each position in the sequence
                logprobs = logprobs * masks[:, 1:]
                #Resulting shape: (batch_size)
                logprobs = logprobs.sum(-1)

                # Detach and delete tensors to free memory
                del output, all_logprobs, logits, attention_mask, tokens, masks, input_ids
                torch.cuda.empty_cache()

                results.append(logprobs)
            return torch.cat(results, dim=0)

    
    @abstractmethod
    def calculate_loss(self, new_logprobs, old_logprobs, data):
        pass


class RefuelTrainer(Trainer):
    """Implementation of Trainer for Refuel algorithm."""
    def calculate_loss(self, new_logprobs, old_logprobs, data):
        ratio_logprob = new_logprobs - old_logprobs
        ratio_logprob = ratio_logprob[:len(new_logprobs) // 2] - ratio_logprob[len(new_logprobs) // 2:]

        reg_diff = ratio_logprob - self.args.refuel.eta * (data["chosen_reward"] - data["reject_reward"])
        loss = (reg_diff ** 2).mean()

        if self.args.refuel.nll_term:
            loss = loss + (self.args.refuel.nll_weight * -new_logprobs[:len(new_logprobs) // 2].mean() / self.args.task.total_length)

  
        return loss



class DPOTrainer(Trainer):

    
    """Implementation of Trainer for DPO algorithm."""
    def calculate_loss(self, new_logprobs, old_logprobs, data):
        # Separate chosen and rejected log probabilities
        pi_w = new_logprobs[:len(new_logprobs) // 2]
        pi_l = new_logprobs[len(new_logprobs) // 2:]
        pi_ref_w = old_logprobs[:len(old_logprobs) // 2]
        pi_ref_l = old_logprobs[len(old_logprobs) // 2:]

        # Compute log ratio for chosen and rejected trajectories
        log_ratio_chosen = pi_w - pi_ref_w
        log_ratio_rejected = pi_l - pi_ref_l

        # Compute the sigmoid term
        beta = self.args.dpo.beta
        sigmoid_arg = beta * (log_ratio_chosen - log_ratio_rejected)

        if self.args.dpo.with_offset:
            sigmoid_arg = sigmoid_arg - (data["chosen_reward"] - data["reject_reward"])

        sigmoid = torch.sigmoid(sigmoid_arg)

        # Compute the DPO loss
        loss = -torch.log(sigmoid).mean()

        return loss