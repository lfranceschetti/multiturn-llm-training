#!/usr/bin/env python3
import sys 
import os 
import json 
import argparse
notebook_dir = os.getcwd() 
sys.path.append(os.path.abspath(os.path.join(notebook_dir, '..', '..', 'llm-negotiations')))
from envs.negotiation.env import NegotiationEnv 
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import tqdm
from datasets import Dataset
from huggingface_hub import HfApi


def sample_geometric_bounded(p, max_value):
    while True:
        sample = np.random.geometric(p) - 1
        if sample <= max_value:
            return sample

def main(args):
    """Main function that executes the program logic.
    
    Args:
        args: The parsed command-line arguments
    """
    print(f"Running with the following arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    if args.game_type == "generic-rental-agreement":
        game = {
            "name": "generic-rental-agreement",
            "issues": [
                "gen-ra-rent.yaml"
            ],
            "issue_weights": [
                [
                    1
                ],
                [
                    1
                ]
            ],
            "scale": [
                100,
                100
            ],
            "description": "A landlord and a prospective tenant are negotiating a rental agreement.",
            "sides": [
                "You are an advisor representing the best interests of the landlord. Your main goal is to negotiate the best possible agreement for the landlord based on the information in the payoff tables. The numbers in the payoff tables show how valuable each outcome is to you. You can trust that the payoffs assigned to the different options in your table are accurate. Do not bring up any issues that are not specifically noted in your payoff table. It is possible that there is only 1 issue.",
                "You are an advisor representing the best interests of the tenant. Your main goal is to negotiate the best possible agreement for the tenant based on the information in the payoff tables. The numbers in the payoff tables show how valuable each outcome is to you. You can trust that the payoffs assigned to the different options in your table are accurate. Do not bring up any issues that are not specifically noted in your payoff table. It is possible that there is only 1 issue."
            ],
            "parties": [
                "Landlord",
                "Tenant"
            ],
            "rules_prompt": "Never forget the following negotiation rules:",
            "rules": [
                "Your total payoff is the sum of your payoffs on all issues. Higher payoffs are better than lower payoffs.",
                "A valid agreement occurs only when all issues are decided. Partial agreements result in a total payoff to you of zero.",
                "You are not allowed to accept any agreement that results in a payoff less than zero.",
                "You are not allowed to deviate from or innovate with the payoffs listed on the payoff table. In other words, you cannot change your payoffs.",
                "No side payments are allowed. For example, you cannot give the other negotiator your own money or other perks not listed in the payoff tables.",
                "You may describe issues and elaborate on them as you see fit. However, you are not allowed to invent additional issues.",
                "Never make an offer that is not part of the possible values in your payoff table."
            ]
        }

        config = {
            "game": game,
            "seed": 42
        }
        config = DictConfig(config)
        env = NegotiationEnv(config, game_type=args.game_type)

    elif args.game_type == "multi-game":
        #Maybe change this to 0.0 later
        env = NegotiationEnv(game_type=args.game_type)
    elif args.game_type == "out-of-domain":
        env = NegotiationEnv(game_type=args.game_type)


    # Create dataset with num_samples examples
    dataset = env.create_dataset(args.num_samples)

    # Filter the dataset to only include examples where starting_agent is True
    dataset_starting_agent = [d for d in dataset if d["starting_agent"]]
    dataset_responder = [d for d in dataset if not d["starting_agent"]]

    reward_functions = env.get_reward_functions()

    vllm_server_host = os.environ.get("VLLM_SERVER_HOST", "0.0.0.0")
    vllm_server_port = int(os.environ.get("VLLM_SERVER_PORT", 8000))
    vllm_client = VLLMClient(vllm_server_host, vllm_server_port, connection_timeout=120.0)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize results data structure
    results = {
        "game": [],
        "issues": [],
        "starting_agent": [],
        "negotiation_role": [],
        "sampled_h": [],
        "chosen": [],
        "reject": [],
        "chosen_token": [],
        "reject_token": [],
        "chosen_mask": [],
        "reject_mask": [],
        "chosen_reward": [],
        "reject_reward": [],
    }

    total_token_count = 0

    
    def process_sample(item, is_starting_agent, sample_num, total_samples, num_tries=0):
        """Process a single example and return the results.
        
        Args:
            item: The example to process (dictionary)
            is_starting_agent: Whether this example is for the starting agent
            sample_num: Current sample number
            total_samples: Total number of samples
        """
        print(f"Processing {'starting agent' if is_starting_agent else 'responder'} sample {sample_num}/{total_samples}")
        if num_tries > 5:
            print(f"Skipping sample {sample_num} after {num_tries} tries")
            return
        
        # Extract prompt and other data from the item
        prompt = item["prompt"]
        prompt_2 = item.get("prompt_2")
        negotiation_role = item.get("negotiation_role")
        game_config = item.get("game_config")

        sampled_h = sample_geometric_bounded(p=0.3, max_value=4)

        prompts = [prompt, prompt]
        prompts_2 = [prompt_2, prompt_2] if prompt_2 else None
        game_configs = [game_config, game_config] if game_config else None

        client_response = vllm_client.generate(
            prompts=prompts,
            prompts_2=prompts_2,
            n=1,
            temperature=1.0,
            top_p=args.top_p if hasattr(args, 'top_p') else 1.0,
            top_k=args.top_k if hasattr(args, 'top_k') else 50,
            max_tokens=args.max_tokens if hasattr(args, 'max_tokens') else 200,
            starting_agent=is_starting_agent,
            sampled_h=sampled_h
        )
        full_conversations = client_response["conversations"]
        token_ids_list = client_response["token_ids"]
        assistant_mask_list = client_response["assistant_masks"]
        token_count = client_response["total_token_count"][0]
        nonlocal total_token_count
        total_token_count += token_count



        rewards = []
        
        # Process reward
        for reward_func in reward_functions:
            for i in range(2):
                reward_kwargs = {k: v for k, v in item.items() if k not in ["prompt", "completion"]}
                output_reward_func, evaluations = reward_func(
                    prompts=[prompt],
                    completions=[full_conversations[i]],
                    get_full_info=True,
                    negotiation_roles=[negotiation_role],
                    game_configs=[game_config],
                    **reward_kwargs
                )
                reward = output_reward_func[0]  # Extract single value from list
                rewards.append(reward)

        # Only store results if there's a clear winner
        if rewards[0] > rewards[1]:
            results["chosen"].append(full_conversations[0])
            results["chosen_token"].append(token_ids_list[0])
            results["chosen_mask"].append(assistant_mask_list[0])
            results["chosen_reward"].append(rewards[0])
            results["reject"].append(full_conversations[1])
            results["reject_token"].append(token_ids_list[1])
            results["reject_mask"].append(assistant_mask_list[1])
            results["reject_reward"].append(rewards[1])
            
            # Store conversation metadata only when we have a valid result
            results["game"].append(game_config["name"])
            results["issues"].append(game_config["issues"])
            results["starting_agent"].append(is_starting_agent)
            results["negotiation_role"].append(negotiation_role)
            results["sampled_h"].append(sampled_h)
            
            if sample_num % 100 == 0:
                print(f"Current number of tokens: {total_token_count}")
                
            print(f"Processed {'starting agent' if is_starting_agent else 'responder'} sample {sample_num}. Rewards: {rewards[0]:.4f}, {rewards[1]:.4f}")
        elif rewards[0] < rewards[1]:
            results["chosen"].append(full_conversations[1])
            results["chosen_token"].append(token_ids_list[1])
            results["chosen_mask"].append(assistant_mask_list[1])
            results["chosen_reward"].append(rewards[1])
            results["reject"].append(full_conversations[0])
            results["reject_token"].append(token_ids_list[0])
            results["reject_mask"].append(assistant_mask_list[0])
            results["reject_reward"].append(rewards[0])
            
            # Store conversation metadata only when we have a valid result
            results["game"].append(game_config["name"])
            results["issues"].append(game_config["issues"])
            results["starting_agent"].append(is_starting_agent)
            results["negotiation_role"].append(negotiation_role)
            results["sampled_h"].append(sampled_h)
            
            if sample_num % 100 == 0:
                print(f"Current number of tokens: {total_token_count}")
                
            print(f"Processed {'starting agent' if is_starting_agent else 'responder'} sample {sample_num}. Rewards: {rewards[0]:.4f}, {rewards[1]:.4f}")
        else:
            # Try again if rewards are equal
            process_sample(item, is_starting_agent, sample_num, total_samples, num_tries=num_tries+1)
            return
    
    # Process each example individually - first starting agent, then responder
    
    # Process starting agent dataset
    print("\nProcessing starting agent dataset...")
    for i, item in enumerate(tqdm.tqdm(dataset_starting_agent, desc="Starting Agent Samples")):
        process_sample(item, True, i+1, len(dataset_starting_agent))

    # Process responder dataset
    print("\nProcessing responder dataset...")
    for i, item in enumerate(tqdm.tqdm(dataset_responder, desc="Responder Samples")):
        process_sample(item, False, i+1, len(dataset_responder))
    

    
    print(f"\nGeneration complete. Total token count: {total_token_count}")
    
    
    # Create dictionary to save
    dict_to_save = {
        "model": args.model,
        "game_type": args.game_type,
        "starting_agent": results["starting_agent"],
        "negotiation_role": results["negotiation_role"],
        "sampled_h": results["sampled_h"],
        "chosen": results["chosen"],
        "reject": results["reject"],
        "chosen_token": results["chosen_token"],
        "reject_token": results["reject_token"],
        "chosen_mask": results["chosen_mask"],
        "reject_mask": results["reject_mask"],
        "chosen_reward": results["chosen_reward"],
        "reject_reward": results["reject_reward"],
    }


    # # Save to local JSON file for backup
    # output_path = os.path.join("/cluster/home/mgiulianelli/code/negotio2/multiturn-llm-training/dataset_generation/results.json")
    # with open(output_path, 'w') as f:
    #     json.dump(dict_to_save, f, indent=2)
    # print(f"Results saved locally to {output_path}")
    
    # Upload to Hugging Face
    if args.hf_repo:
        try:
            # Convert to dataset format
            dataset_dict = {}
            for key, value in dict_to_save.items():
                if isinstance(value, list):
                    dataset_dict[key] = value
            
            # Create a Hugging Face dataset
            hf_dataset = Dataset.from_dict(dataset_dict)
            
            # Upload to Hugging Face
            dataset_name = f"{args.game_type}_{args.model.replace('/', '_')}"
            print(f"Uploading to Hugging Face repository: {args.hf_repo}, dataset: {dataset_name}")
            
            api = HfApi()
            api_token = os.environ.get("HF_TOKEN")
            if not api_token:
                print("Warning: HF_TOKEN environment variable not set. Using anonymous upload.")
            
            # Upload the dataset
            hf_dataset.push_to_hub(
                args.hf_repo,
                dataset_name,
                token=api_token,
            )
            
            print(f"Dataset successfully uploaded to: https://huggingface.co/datasets/{args.hf_repo}/{dataset_name}")
        except Exception as e:
            print(f"Error uploading to Hugging Face: {str(e)}")

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Evaluation script for multiturn LLM training")
    
    # Add arguments
    parser.add_argument("--model", type=str, required=True, help="Model to evaluate")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--game-type", type=str, default="generic-rental-agreement", help="Type of game to evaluate")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum number of tokens to generate")
    parser.add_argument("--hf-repo", type=str, help="Hugging Face repository ID to upload the dataset (e.g., 'username/dataset-name')")


    # Parse arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args)
