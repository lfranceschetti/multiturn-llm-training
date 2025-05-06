import sys 
import os 
import json 
import argparse
notebook_dir = os.getcwd() 
sys.path.append(os.path.abspath(os.path.join(notebook_dir, '..', 'llm-negotiations'))) 
from envs.negotiation_env import NegotiationEnv 


#https://github.com/huggingface/peft/blob/main/examples/sft/README.md

#Current training settings are following the following tutorial:
#http://github.com/huggingface/peft/blob/main/examples/sft/utils.py

def main():

        # update model constructors in case of model overrides
    model_name = "meta-llama/Llama-3.2-1B-Instruct"


    negotiation_env = NegotiationEnv(game_type="multi-game")

    print("Negotiation Environment created")

    train_dataset = negotiation_env.create_dataset(size=47, game_type="multi-game")
    print(train_dataset)




if __name__ == "__main__":
    main()