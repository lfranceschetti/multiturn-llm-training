from typing import Callable, Optional, Union, Any, List

from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    is_wandb_available,
    Trainer,
)
from transformers.utils import is_peft_available


from pathlib import Path
import sys
import os
# Add the parent directory to the Python path to access the sister directory
# At the top of grpo_multi.py
import sys
print("Python path:", sys.path)
import trl
print("Using trl from:", trl.__file__)



# Force a fresh import
import trl
print(f"Imported trl module: {trl}")
print(f"trl module path: {getattr(trl, '__file__', 'No __file__ attribute')}")

from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import apply_chat_template, maybe_apply_chat_template
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)


from .environment import Environment
from .GRPOEnvLogger import print_prompt_completions_sample

if is_peft_available():
    from peft import PeftConfig # type: ignore

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class GRPOMultiTrainer(GRPOTrainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
            peft_config: Optional["PeftConfig"] = None,
            **kwargs,
    ):
        if not args.use_vllm: # type: ignore
            raise ValueError("vLLM must be enabled for GRPOMultiTrainer")
        if not (callable(reward_funcs) or (isinstance(reward_funcs, list) and all(callable(f) for f in reward_funcs))): 
            raise ValueError("reward_funcs must be a function or a list of functions. Use vLLM to host neural reward models.")
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )

    def tokenize_messages(self, messages, max_length=2048):
        """
        Convert messages to token IDs and attention masks, identifying which tokens are
        from the assistant (for reward calculation and loss computation).
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (token_ids, attention_mask, assistant_mask)
        """
        # Get the full tokenized conversation

        print("GETTING TOKEN IDS for msgs", messages)
        token_ids = self.processing_class.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=False, 
            padding='max_length', 
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Create the attention mask (1 for tokens, 0 for padding)
        attention_mask = (token_ids != self.processing_class.pad_token_id).int()
        
        # Create assistant mask (1 for assistant tokens, 0 for others)
        assistant_mask = torch.zeros_like(token_ids)
        
        # Process message by message to identify assistant portions
        current_position = 0


        for msg in messages:
            # Tokenize just this message with the chat template
            msg_tokens = self.processing_class.apply_chat_template(
                [msg], 
                tokenize=True, 
                add_generation_prompt=False,
                return_tensors="pt"
            )
            msg_length = msg_tokens.size(1)

            print("MSG LENGTH", msg_length)
            
            # If we would exceed the max length, truncate
            if current_position + msg_length > max_length:
                msg_length = max_length - current_position
                if msg_length <= 0:
                    break
            
            # Mark assistant tokens
            if msg["role"] == "assistant":
                assistant_mask[0, current_position:current_position + msg_length] = 1
            
            current_position += msg_length
            if current_position >= max_length:
                break
        
        return token_ids, attention_mask, assistant_mask
    
    @profiling_decorator
    def _get_per_token_logps(self, model, input_ids, attention_mask, assistant_mask):
        """
        Compute the log probabilities for tokens generated by the assistant.
        
        Args:
            model: The model to compute log probabilities for
            input_ids: Token IDs of shape [B, L]
            attention_mask: Attention mask of shape [B, L]
            assistant_mask: Binary mask indicating which tokens were generated by the assistant [B, L]
            
        Returns:
            Log probabilities for the assistant-generated tokens
        """
        # Get logits from the model for the entire sequence
        print("Using model", model)

        rank = self.accelerator.process_index
    
        # Add rank to all print statements
        print(f"[Rank {rank}] Entering _get_per_token_logps")
        print(f"[Rank {rank}] Using model {model}")

        # First synchronize before tiny test
        if torch.distributed.is_initialized():
            print(f"[Rank {rank}] Before tiny test barrier")
            torch.distributed.barrier()
            print(f"[Rank {rank}] After tiny test barrier")

        with torch.no_grad():
            tiny_input = torch.ones((1, 10), dtype=torch.long, device="cuda:0")
            tiny_mask = torch.ones((1, 10), device="cuda:0")
            try:
                tiny_output = model(input_ids=tiny_input, attention_mask=tiny_mask)
                print("Tiny test passed")
                print(tiny_output)
            except Exception as e:
                print("Tiny test failed")
                print(e)

        print("input_ids shape:", input_ids.shape)
        print("attention_mask shape:", attention_mask.shape)
        print("input_ids device:", input_ids.device)
        print("attention_mask device:", attention_mask.device)
        print("input_ids dtype:", input_ids.dtype)
        print("attention_mask dtype:", attention_mask.dtype)
        print("First few values of input_ids:", input_ids[0, :10])  # First 10 tokens of first item in batch
        print("First few values of attention_mask:", attention_mask[0, :10])  # First 10 mask values of first item in batch

        model_device = next(model.parameters()).device
        print(f"Model is on device: {model_device}")
        print(f"input_ids is on device: {input_ids.device}")
        print(f"attention_mask is on device: {attention_mask.device}")
        print(f"assistant_mask is on device: {assistant_mask.device}")

        if input_ids.device != model_device:
            print(f"Moving input_ids from {input_ids.device} to {model_device}")
            input_ids = input_ids.to(model_device)
        
        if attention_mask.device != model_device:
            print(f"Moving attention_mask from {attention_mask.device} to {model_device}")
            attention_mask = attention_mask.to(model_device)
        
        # Get logits from the model for the entire sequence
        print("Starting model forward pass")
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        print("Completed model forward pass")
        logits = outputs.logits  # Shape: [B, L, V]

        print("LOGITS", logits)
        
        # Shift logits left and input_ids right to align
        # This matches each token's logit with the next token's ID (standard LM setup)
        logits = logits[:, :-1, :]  # Shape: [B, L-1, V]
        shifted_input_ids = input_ids[:, 1:]  # Shape: [B, L-1]
        shifted_assistant_mask = assistant_mask[:, 1:]  # Shape: [B, L-1]
        
        # Divide logits by temperature
        logits = logits / self.temperature
        
        print("Calculating per token")
        # Compute log probabilities for each token
        per_token_logps = selective_log_softmax(logits, shifted_input_ids)

        print("PER TOKEN LOGPROBs", per_token_logps)

        

        # Apply the assistant mask to keep only assistant-generated tokens
        # Replace zeros from mask with -inf to properly ignore them in calculations
        masked_log_probs = per_token_logps.masked_fill(shifted_assistant_mask == 0, float('-inf'))

        print("MASKED LOGPROBS", masked_log_probs)
        
        return masked_log_probs

    def _generate_and_score_completions(
         self, inputs: dict[str, Union[torch.Tensor, Any]]   
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs] # type: ignore
        # prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs] # type: ignore
        # prompt_inputs = self.processing_class(
        #     prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False # type: ignore
        # ) # type: ignore
        # prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs) # type: ignore
        # prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # if self.max_prompt_length is not None:
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Gather the original prompts in message dict form, not the text form
        all_prompts_text = gather_object(prompts)

        rank = self.accelerator.process_index
    
        # Add rank to all print statements
        print(f"[Rank {rank}] Entering generating and scoring completions")

        if self.accelerator.is_main_process:

            ordered_set_of_prompts = all_prompts_text[:: self.num_generations]

            with profiling_context(self, "vLLM.generate"):
                full_conversations = self.vllm_client.generate(
                    prompts=ordered_set_of_prompts,
                    n=self.num_generations,
                    repetition_penalty=self.repetition_penalty,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=-1 if self.top_k is None else self.top_k,
                    min_p=0.0 if self.min_p is None else self.min_p,
                    max_tokens=self.max_completion_length,
                    guided_decoding_regex=self.guided_decoding_regex,
            )
                
                print("FULL CONVERSATIONS", full_conversations)

        else:
            full_conversations = [None] * len(all_prompts_text)

        print(f"[Rank {rank}] full_conversations length: {len(full_conversations)}, example: {full_conversations[:1]}")

        print(f"[Rank {rank}] Before broadcast")

        full_conversations = broadcast_object_list(full_conversations, from_process=0)

        print(f"[Rank {rank}] After broadcast")

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        full_conversations = full_conversations[process_slice]

        #PRINT STARTING TOKENIZATION IN RANK x
        print("Starting tokenization in rank", self.accelerator.process_index)

        token_ids_list = []
        attention_mask_list = []
        assistant_mask_list = []

        for i, conversation in enumerate(full_conversations):
            print("Tokenizing conversation", i)
            token_ids, attention_mask, assistant_mask = self.tokenize_messages(conversation)

            print("TOKEN IDS")
            print(token_ids)
            print("ATTENTION MASK")
            print(attention_mask)
            print("ASSISTANT MASK")
            print(assistant_mask)
            token_ids_list.append(token_ids)
            attention_mask_list.append(attention_mask)
            assistant_mask_list.append(assistant_mask)

        print("TOKENIZATION complete")
        # Stack all tensors
        token_ids = torch.cat(token_ids_list, dim=0).to(device)
        attention_mask = torch.cat(attention_mask_list, dim=0).to(device)
        assistant_mask = torch.cat(assistant_mask_list, dim=0).to(device)

        
        print("Calculating logprobs")
        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, token_ids, attention_mask, assistant_mask
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, token_ids, attention_mask, assistant_mask
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, token_ids, attention_mask, assistant_mask
                    )

        print("Calculating rewards")
        # use message dicts for reward function inputs
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]] # type: ignore
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys} # type: ignore
            output_reward_func = reward_func(prompts=prompts, completions=full_conversations, **reward_kwargs) # type: ignore
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1) # type: ignore
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) # type: ignore

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)


        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if mode == "train":
            self._total_train_tokens += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        completion_length = self.accelerator.gather_for_metrics(assistant_mask.sum(1)).float().mean().item() # type: ignore
        self._metrics[mode]["completion_length"].append(completion_length)

        reward_per_func = rewards_per_func.mean(0) # type: ignore
        for i, reward_func in enumerate(self.reward_funcs):
            if reward_func.__name__:
                reward_func_name = reward_func.__name__ # type: ignore
            else:
                reward_func_name = f"reward_func_{i}"
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts)
            completions_to_log = gather_object(full_conversations)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                        self.num_completions_to_print,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None: # type: ignore
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)}) # type: ignore

        print("TOKEN IDS")
        print(token_ids)
        print("ATTENTION MASK")
        print(attention_mask)
        print("ASSISTANT MASK")
        print(assistant_mask)
        print("OLD PER TOKEN LOGPS")
        print(old_per_token_logps)
        print("REF PER TOKEN LOGPS")
        print(ref_per_token_logps)
        print("ADVANTAGES")
        print(advantages)

        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "assistant_mask": assistant_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
    
    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        input_ids = inputs["token_ids"]
        attention_mask = inputs["attention_mask"]
        assistant_mask = inputs["assistant_mask"]

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, assistant_mask)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * assistant_mask).sum() / assistant_mask.sum()

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * assistant_mask).sum() / assistant_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (coef_1 < (1 - self.epsilon_low)) | (coef_1 > (1 + self.epsilon_high))
        clip_ratio = (is_clipped * assistant_mask).sum() / assistant_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())


        print("Calculated loss:", loss)
        return loss
    
