import sys 
import os 
import json 
import argparse
notebook_dir = os.getcwd() 
sys.path.append(os.path.abspath(os.path.join(notebook_dir, '..', 'llm-negotiations'))) 
from envs.negotiation.env import NegotiationEnv 


#https://github.com/huggingface/peft/blob/main/examples/sft/README.md

#Current training settings are following the following tutorial:
#http://github.com/huggingface/peft/blob/main/examples/sft/utils.py

def main():

        # update model constructors in case of model overrides
    model_name = "meta-llama/Llama-3.2-1B-Instruct"


    negotiation_env = NegotiationEnv(game_type="multi-game")

    print("Negotiation Environment created")

    train_dataset = negotiation_env.create_dataset(size=46)
    
    game_names_and_descriptions = {}
    game_names_and_issues = {}


    #json dump every element in the dataset
    for i, game in enumerate(train_dataset):
        name = game["game_config"]["name"]
        description = game["game_config"]["description"]
        if name not in game_names_and_descriptions.keys():
            game_names_and_descriptions[name] = description

        issues = game["game_config"]["issues"]
        for issue in issues:
            if issue not in game_names_and_issues.keys():
                game_names_and_issues[issue] = []
            game_names_and_issues[issue].append(name)
        game_names_and_descriptions.append({"name": name, "description": description})

    print(game_names_and_descriptions)
    print(game_names_and_issues)    





if __name__ == "__main__":
    main()