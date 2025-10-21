import sys 
import os 
import json 
import argparse
notebook_dir = os.getcwd() 
sys.path.append(os.path.abspath(os.path.join(notebook_dir, '..', 'llm-negotiations'))) 
from envs.negotiation.env import NegotiationEnv 
from trainer.LAGRPOTrainer import LAGRPOTrainer
from trl import GRPOConfig
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf 
from envs.negotiation.games import Game
from evaluator.utils import unpack_nested_yaml, fill_defaults, get_inference_root_overrides
import torch
from envs.negotiation.games import Game
from transformers import BitsAndBytesConfig
from peft import LoraConfig

#AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


#https://github.com/huggingface/peft/blob/main/examples/sft/README.md

#Current training settings are following the following tutorial:
#http://github.com/huggingface/peft/blob/main/examples/sft/utils.py

@hydra.main(version_base=None, config_path="../llm-negotiations/configs", config_name="inference_root")
def main(cfg: DictConfig):
    with open_dict(cfg['experiment']):
        # unpack nested yaml files
        _ = unpack_nested_yaml(cfg['experiment'])
        # check if any keys are missing and update default run-time overrides
        overrides = get_inference_root_overrides(cfg, "/cluster/home/mgiulianelli/code/negotio2/llm-negotiations/configs/inference_root.yaml")
        _ = fill_defaults(cfg['experiment'], root_overrides=overrides, defaults_file="/cluster/home/mgiulianelli/code/negotio2/llm-negotiations/configs/negotiation_defaults.yaml")
        # unpack default yaml files (if any)
        _ = unpack_nested_yaml(cfg['experiment'])
        # update model constructors in case of model overrides

    config = cfg.experiment

    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    config_dict = OmegaConf.to_container(config, resolve=True)
    print("Config:\n", json.dumps(config_dict, indent=4))

    negotiation_env = NegotiationEnv(config, game_type="multi-game")

    print("Negotiation Environment created")

    train_dataset = negotiation_env.create_dataset(size=4000)
    eval_dataset = negotiation_env.create_dataset(size=92)

    reward_functions = negotiation_env.get_reward_functions()

    # notable defaults: lr = 1e-6, max_grad_norm = 0.01, constant lr 10 warmup steps, 1024 tokens in+out
    run_name = "grpo_turn_level_multi_game_3"
    num_gpus = torch.cuda.device_count()

    vllm_server_host = os.environ.get("VLLM_SERVER_HOST", "0.0.0.0")
    vllm_server_port = int(os.environ.get("VLLM_SERVER_PORT", 8000))

    training_args = GRPOConfig(
        output_dir=f"/cluster/scratch/mgiulianelli/huggingface/models/{run_name}",
        run_name=run_name,
        learning_rate=5e-5,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        num_train_epochs=1,
        bf16=True,
        num_iterations=1,
        max_prompt_length=1600,
        max_completion_length=200,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_generations=8,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=200,
        save_only_model=True,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.7 if num_gpus > 1 else 0.3,
        logging_steps=20,
        log_on_each_node=False,
        log_completions=True,
        report_to="wandb",
        vllm_server_host=vllm_server_host,
        vllm_server_port=vllm_server_port,
        eval_strategy="steps",
        eval_steps=100,
        eval_on_start=False,
        push_to_hub=True,
        beta=0.08,
        hub_strategy="all_checkpoints",
        
    )

    bnb_config = None

    print("Training Args:\n", training_args)


    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
       )

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )

    trainer = LAGRPOTrainer(
        model=model,
        reward_funcs=reward_functions, 
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        turn_level_sampling=True
    )

    trainer.train()

    trainer.save_model()


if __name__ == "__main__":
    main()