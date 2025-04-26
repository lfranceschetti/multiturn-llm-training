import sys 
import os 
import json 
import argparse
notebook_dir = os.getcwd() 
sys.path.append(os.path.abspath(os.path.join(notebook_dir, '..', 'llm-negotiations'))) 
from envs.negotiation_env import NegotiationEnv 
from trainer.GRPOMultiTrainer import GRPOMultiTrainer
from trl import GRPOConfig
import hydra
from omegaconf import DictConfig, open_dict, OmegaConf 
from simulator.games import Game
from helpers.utils import unpack_nested_yaml, fill_defaults, get_inference_root_overrides
import torch
from simulator.games import Game
from transformers import BitsAndBytesConfig
from peft import LoraConfig

#AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM


#https://github.com/huggingface/peft/blob/main/examples/sft/README.md

#Current training settings are following the following tutorial:
#http://github.com/huggingface/peft/blob/main/examples/sft/utils.py

def main():

        # update model constructors in case of model overrides


    model_name = "meta-llama/Llama-3.2-1B-Instruct"

    config_dict = OmegaConf.to_container(config, resolve=True)
    print("Config:\n", json.dumps(config_dict, indent=4))

    negotiation_env = NegotiationEnv(config)

    print("Negotiation Environment created")

    train_dataset = negotiation_env.create_dataset(size=500)
 




if __name__ == "__main__":
    main()