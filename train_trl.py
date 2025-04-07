import sys 
import os 
import json 
notebook_dir = os.getcwd() 
sys.path.append(os.path.abspath(os.path.join(notebook_dir, '..', 'llm-negotiations'))) 
from envs.negotiation_env import NegotiationEnv 
from trainer.GRPOEnvTrainer import GRPOEnvTrainer 
from trainer.utils import get_default_grpo_config 
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf 
from simulator.games import Game
from helpers.utils import unpack_nested_yaml, fill_defaults, get_inference_root_overrides
import torch

@hydra.main(version_base=None, config_path="../llm-negotiations/configs", config_name="inference_root")
def main(cfg: DictConfig):

    with open_dict(cfg['experiment']):
        # unpack nested yaml files
        _ = unpack_nested_yaml(cfg['experiment'])
        # check if any keys are missing and update default run-time overrides
        overrides = get_inference_root_overrides(cfg)
        _ = fill_defaults(cfg['experiment'], root_overrides=overrides)
        # unpack default yaml files (if any)
        _ = unpack_nested_yaml(cfg['experiment'])
        # update model constructors in case of model overrides

    config = cfg.experiment

    config_dict = OmegaConf.to_container(config, resolve=True)
    print("Config:\n", json.dumps(config_dict, indent=4))


    negotiation_env = NegotiationEnv(config)

    print("Negotiation Environment created")

    train_dataset = negotiation_env.get_dataset(size=2000)
    eval_dataset = negotiation_env.get_dataset(size=200)

    reward_functions = negotiation_env.get_reward_functions()

    # notable defaults: lr = 1e-6, max_grad_norm = 0.01, constant lr 10 warmup steps, 1024 tokens in+out
    run_name = "grpo_negotiation_test_1"
    num_gpus = torch.cuda.device_count()
    training_args = get_default_grpo_config(run_name=run_name, num_gpus=num_gpus)

    print("Training Args:\n", json.dumps(training_args, indent=4))

    #Model that should be trained
    model = negotiation_env.agent_1.model.model
    tokenizer = negotiation_env.agent_1.model.tokenizer



    trainer = GRPOEnvTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions, 
        env=negotiation_env,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()

if __name__ == "__main__":
    main()