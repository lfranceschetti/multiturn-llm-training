import sys 
import os 
import json 
import argparse
from envs.negotiation.env import NegotiationEnv 
from multiturn_llm_training.grpo.lagrpo_trainer import LAGRPOTrainer
from trl import GRPOConfig
import hydra
from envs.negotiation.games import Game
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from datasets import Dataset
#AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


#https://github.com/huggingface/peft/blob/main/examples/sft/README.md

#Current training settings are following the following tutorial:
#http://github.com/huggingface/peft/blob/main/examples/sft/utils.py

def get_test_reward_functions():
    """
    Returns a list of reward functions to be used by the GRPO trainer.
    """
    
    def test_reward_len(completions,**kwargs):
        return [-abs(20 - len(completion)) for completion in completions]

    
    return [test_reward_len]

def get_test_datasets(train_size, eval_size):
    """
    Returns a train and eval dataset to be used by the GRPO trainer.
    """
    prompt = "Please generate a random number between 0 and 100. Only respond with the number, nothing else."
    prompt_2 = "Please generate a random number between 0 and 100. Only respond with the number, nothing else."
    game_config = { }
    train_samples = []
    eval_samples = []
    for _ in range(train_size):
        train_samples.append({
            "prompt": prompt,
            "prompt_2": prompt_2,
            "game_config": game_config
        })
    for _ in range(eval_size):
        eval_samples.append({
            "prompt": prompt,
            "prompt_2": prompt_2,
            "game_config": game_config
        })
    train_dataset = Dataset.from_list(train_samples)
    eval_dataset = Dataset.from_list(eval_samples)  
    return train_dataset, eval_dataset

def main(args):

    print("Training Args:\n", args)


    os.makedirs(f"{args.output_dir}/{args.run_name}", exist_ok=True)


    if args.test_env:
        reward_functions = get_test_reward_functions()
        train_dataset, eval_dataset = get_test_datasets(args.train_size, args.eval_size)
        print("Test datasets created")
    else:
        negotiation_env = NegotiationEnv(game_type="multi-game")
        print("Negotiation Environment created")
        train_dataset = negotiation_env.create_dataset(size=args.train_size)
        eval_dataset = negotiation_env.create_dataset(size=args.eval_size)
        reward_functions = negotiation_env.get_reward_functions()


    # notable defaults: lr = 1e-6, max_grad_norm = 0.01, constant lr 10 warmup steps, 1024 tokens in+out
    num_gpus = torch.cuda.device_count()

    vllm_server_host = os.environ.get("VLLM_SERVER_HOST", "0.0.0.0")
    vllm_server_port = int(os.environ.get("VLLM_SERVER_PORT", 8000))

    training_args = GRPOConfig(
        output_dir=f"{args.output_dir}/{args.run_name}",
        run_name=args.run_name,
        learning_rate=5e-5,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        num_train_epochs=1,
        bf16=True,
        num_iterations=1,
        max_prompt_length=1600,
        max_completion_length=200,
        # Note: The effective batch size (per_device_train_batch_size * num_gpus) must be divisible
        # by num_generations to ensure proper grouping for reward normalization in GRPO.
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

    if args.quantized:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
    else:
        bnb_config = None

    print("Training Args:\n", training_args)


    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--eval-size", type=int, default=10)
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--run-name", type=str, default="grpo_test_1")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "..", "output"))
    parser.add_argument("--quantized", type=bool, default=False)
    parser.add_argument("--test-env", type=bool, default=False)

    args = parser.parse_args()
    
    main(args)