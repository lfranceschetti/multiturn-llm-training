import time
import numpy as np
import torch
from typing import Any, Dict, List, Sequence, Optional, Union
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from simulator.protocols import NegotiationProtocol
from simulator.games import Game
from simulator.agents import NegotiationAgent
from evaluator.evaluator import Evaluator
from datasets import Dataset
from models.openai_model import OpenAIModel
import itertools
import os
import yaml
import random
import json


SCALE = [100, 100]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class NegotiationEnv:
    def __init__(
        self,
        game_type: str = "generic-rental-agreement",
        seed: int = 42,
        **kwargs
    ):
        """
        Environment for negotiation simulations between two agents.
        
        Args:
            config: Optional configuration for the negotiation environment. If not provided, default values will be used.
            **kwargs: Additional keyword arguments to override config values
        """

        self.game_type = game_type
        self.seed = seed
        self.set_seed(self.seed)

        self.games_path = "/cluster/home/fraluca/negotio2/multiturn-llm-training/configs/games/"
        self.rules_path = self.games_path + "general_game_rules.yaml"


    def get_prompts_from_game(self, game: Game, max_rounds: int = 5):
        prompts = game.get_system_game_msg(agent_id=0)["content"]
        prompts_2 = game.get_system_game_msg(agent_id=1)["content"]
        prompts += f"You have {max_rounds} rounds to reach an agreement; failure to do so will result in a total payoff of 0 for both parties, so it is crucial to find a mutually acceptable solution within these rounds. Also keep your answer as short and concise as possible. Directly state your offer and response to your negotiation partner, without showing your thought process. Make sure that you only talk about the issues in the payoff table, else your payoff will be 0. Do not mention your internal payoff to the other party, they are only for your reference, otherwise your payoff will be 0. "
        prompts_2 += f"You have {max_rounds} rounds to reach an agreement; failure to do so will result in a total payoff of 0 for both parties, so it is crucial to find a mutually acceptable solution within these rounds. Also keep your answer as short and concise as possible. Directly state your offer and response to your negotiation partner, without showing your thought process. Make sure that you only talk about the issues in the payoff table, else your payoff will be 0. Do not mention your internal payoff to the other party, they are only for your reference, otherwise your payoff will be 0. "
        return prompts, prompts_2
        


    def create_dataset(self, size=2000) -> Dataset:
        # For now, we will use the same prompt for all games
        with open(self.rules_path, "r") as f:
            rules = yaml.safe_load(f)


        if self.game_type == "generic-rental-agreement":
            games_config = {
                "game": "generic-rental-agreement.yaml",
                "issues": ["gen-ra-rent.yaml"],
                "issue_weights": [[1], [1]],
                "scale": SCALE,
                **rules,
            }

            games_config = self.add_game_info_to_game_config(games_config)

            game = Game(**games_config)
            
            prompts, prompts_2 = self.get_prompts_from_game(game)

            # Create samples with the correct structure
            samples = []
            for i in range(size):
                samples.append({
                    "prompt": prompts,  # This should be a list of message dicts
                    "prompt_2": prompts_2,
                    "game_config": games_config,
                    "starting_agent": i % 2 == 0,
                    "game_type": self.game_type,
                    "negotiation_role": 1
                })
        
            # Create a Hugging Face Dataset
            try:
                ds = Dataset.from_list(samples)
                _ = ds[0]  # force materialize
            except Exception:
                ds = Dataset.from_dict({"text": [p1] * size})
            return ds
            

        elif self.game_type in {"multi-game", "out-of-domain"}:
            if self.game_type == "multi-game":
                games_used = [
                    {"game": "generic-rental-agreement.yaml",
                    "issues": ["gen-ra-deposit.yaml","gen-ra-duration-distributive.yaml","gen-ra-duration.yaml","gen-ra-rent.yaml"]},
                    {"game": "generic-loan-agreement.yaml",
                    "issues": ["gen-la-amount.yaml","gen-la-duration.yaml","gen-la-fees.yaml","gen-la-rate.yaml"]},
                    {"game": "generic-merger.yaml",
                    "issues": ["gen-m-benefits.yaml", "gen-m-ownership.yaml"]},
                ]
            else:
                games_used = [
                    {"game": "rio_copa.yaml",
                    "issues": ["rp_contingent_liability.yaml","rp_family_employees.yaml","rp_financing.yaml","rp_non_compete_period.yaml"]},
                ]

            #Get additional game configs from their respective yaml files
            for gd in games_used:

                gd = add_game_info_to_game_config(gd)

           
            issue_weights_multiple_possibilites = [90, 50, 10]

            #Create the different game configs except for the issue weights distribution
            game_configs = []
            for gd in games_used:
                issues = gd["issues"]
                game_info = gd["game_info"]
                #First add all single issue games
                for issue in issues:
                    game_configs.append({
                        "name": issue,
                        "issues": [issue],
                        "scale": SCALE,
                        **rules,
                        **game_info
                    })

                #Then make all combinations of issues for the game
                combos = list(itertools.combinations(issues, 2))

                for combo in combos:
                    #Sample a random issue weight from the list do it with numpy
                    game_configs.append({
                        "name": combo,
                        "issues": list(combo),
                        "scale": SCALE,
                        **rules,
                        **game_info
                    })

                # distribute samples evenly across combos
            
            print("Number of game configs: ", len(game_configs))
            #Shuffle the game configs so that it doesnt optimize on the first few games or single-issue games
            game_configs = np.random.permutation(game_configs)
            
            samples = []
            for i in range(size//2):
                game_config = game_configs[i % len(game_configs)]
                if len(game_config["issues"]) == 1:
                    game_config["issue_weights"] = [[1], [1]]
                else:
                    #Sample a random issue weight from the list do it with numpy
                    iw_1 = np.random.choice(issue_weights_multiple_possibilites)
                    iw_2 = np.random.choice(issue_weights_multiple_possibilites)
                    game_config["issue_weights"] = [[iw_1, 100-iw_1], [iw_2, 100-iw_2]]

                game = Game(**game_config)
                prompt1, prompt2 = self.get_prompts_from_game(game)
                samples.append({
                    "prompt": prompt1,
                    "prompt_2": prompt2,
                    "game_config": game_config,
                    "starting_agent": (i // len(game_configs)) % 2 == 0,
                    "game_type": self.game_type,
                    "negotiation_role": 1
                })

                samples.append({
                    "prompt": prompt2,
                    "prompt_2": prompt1,
                    "game_config": game_config,
                    "starting_agent": (i // len(game_configs)) % 2 == 0,
                    "game_type": self.game_type,
                    "negotiation_role": 2
                })


            dataset = Dataset.from_list(samples)
            return dataset

        else:
            raise ValueError(f"Game type {self.game_type} not supported")



    def get_reward_functions(self):
        """
        Returns a list of reward functions to be used by the GRPO trainer.
        """
        
        def negotiation_payoff_reward(prompts, completions, get_full_info=False, game_config=None, negotiation_roles=None, **kwargs):
            rewards = []
            evaluations = [] if get_full_info else None
            
            for i, messages in enumerate(completions):
                messages = messages[1:]
                # try:
                    # Determine starting agent
                starting_agent = 0
                if messages and messages[0]["role"] == "user":
                    starting_agent = 1

                # Create a new game instance for each evaluation using the provided config
                current_game_config = game_config[i] if isinstance(game_config, list) else game_config
                game = Game(**current_game_config)


                evaluation_model = OpenAIModel(model_provider="openai", model_name="gpt-4o-mini")
                evaluator = Evaluator(model=evaluation_model, game=game, game_type=self.game_type)

                # Use the environment's evaluator
                evaluation = evaluator.evaluate(
                    messages, 
                    starting_agent=starting_agent, 
                    get_payoffs=True
                )

                    
                if negotiation_roles is None or not isinstance(negotiation_roles, list) or len(negotiation_roles) == 0:
                    negotiation_roles = [1] * len(completions)
                # Get the payoff for the agent we're training (Agent 1)
                if negotiation_roles[i] == 1 or negotiation_roles[i] == None:
                    payoff = evaluation["payoffs"]["Agent 1"]
                else:
                    payoff = evaluation["payoffs"]["Agent 2"]

                rewards.append(payoff)
                
                if get_full_info:
                    evaluations.append(evaluation)
                    
                # except Exception as e:
                #     print(f"Error computing reward: {e}")
                #     rewards.append(0.0)
                #     if get_full_info:
                #         evaluations.append(None)
            
            if get_full_info:
                return rewards, evaluations
            return rewards
        
        return [negotiation_payoff_reward]
          


    def add_game_info_to_game_config(self, game_config: dict):
        game_filename = game_config["game"]
        with open(os.path.join(self.games_path, game_filename), "r") as f:
            game_dict = yaml.safe_load(f)
        game_config.update(game_dict)
        return game_config


    def set_seed(self, seed: int):
        """
        Set the seed for reproducibility across random, numpy, and PyTorch.
        """
        np.random.seed(seed)  # NumPy
        torch.manual_seed(seed)  # PyTorch CPU
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)  # PyTorch GPU
        torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior for CuDNN
        torch.backends.cudnn.benchmark = False