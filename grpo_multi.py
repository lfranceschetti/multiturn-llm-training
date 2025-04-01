import sys 
import os 
import json 
notebook_dir = os.getcwd() 
sys.path.append(os.path.abspath(os.path.join(notebook_dir, '..', 'llm-negotiations'))) 
from envs.negotiation_env import NegotiationEnv 
from trainer.GRPOMultiTrainer import GRPOMultiTrainer
from trainer.utils import get_default_grpo_config 
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf 
from simulator.games import Game
from helpers.utils import unpack_nested_yaml, fill_defaults, get_inference_root_overrides
import torch
from simulator.games import Game
#AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


@hydra.main(version_base=None, config_path="../llm-negotiations/configs", config_name="inference_root")
def main(cfg: DictConfig):

    with open_dict(cfg['experiment']):
        # unpack nested yaml files
        _ = unpack_nested_yaml(cfg['experiment'])
        # check if any keys are missing and update default run-time overrides
        overrides = get_inference_root_overrides(cfg, "/cluster/home/fraluca/negotio2/llm-negotiations/configs/inference_root.yaml")
        _ = fill_defaults(cfg['experiment'], root_overrides=overrides, defaults_file="/cluster/home/fraluca/negotio2/llm-negotiations/configs/negotiation_defaults.yaml")
        # unpack default yaml files (if any)
        _ = unpack_nested_yaml(cfg['experiment'])
        # update model constructors in case of model overrides

    config = cfg.experiment

    config_dict = OmegaConf.to_container(config, resolve=True)
    print("Config:\n", json.dumps(config_dict, indent=4))


    negotiation_env = NegotiationEnv(config)

    print("Negotiation Environment created")

    train_dataset = negotiation_env.create_dataset(size=8000)
    eval_dataset = negotiation_env.create_dataset(size=400)

    reward_functions = negotiation_env.get_reward_functions()

    # notable defaults: lr = 1e-6, max_grad_norm = 0.01, constant lr 10 warmup steps, 1024 tokens in+out
    run_name = "grpo_negotiation_test_1"
    num_gpus = torch.cuda.device_count()

    vllm_server_host = os.environ.get("VLLM_SERVER_HOST", "0.0.0.0")
    vllm_server_port = int(os.environ.get("VLLM_SERVER_PORT", 8000))    
    training_args = get_default_grpo_config(run_name, vllm_server_host, vllm_server_port, num_gpus=num_gpus)

    print("Training Args:\n", training_args)

    #Model that should be trained
    model = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        # quantization_config=bnb_config,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        # device_map=device_map,
       )

    trainer = GRPOMultiTrainer(
        model=model,
        reward_funcs=reward_functions, 
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer

    )

    trainer.train()

if __name__ == "__main__":
    main()