from functools import partial
import torch
from torch.optim.lr_scheduler import LambdaLR
import math
from trl import GRPOConfig
from typing import List, Optional

def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)



def get_default_grpo_config(run_name: str, vllm_server_host: str, vllm_server_port: int, num_gpus: int = 1) -> GRPOConfig:
    return GRPOConfig(
        output_dir=f"outputs/{run_name}",
        run_name=run_name,
        learning_rate=1e-6,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=20,
        num_train_epochs=1,
        bf16=True,
        adam_beta1=0.9,
        adam_beta2=0.99,
        max_grad_norm=0.1,
        num_iterations=1,
        max_prompt_length=1024,
        max_completion_length=200,
        per_device_train_batch_size=2,
        num_generations=(2 * num_gpus - 2 if num_gpus > 1 else 2),
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=100,
        save_only_model=True,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.7 if num_gpus > 1 else 0.3,
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        report_to="wandb",
        vllm_server_host=vllm_server_host,
        vllm_server_port=vllm_server_port,
        deepspeed="/cluster/home/fraluca/negotio2/multiturn-llm-training/deepspeed_config.json",
    )


