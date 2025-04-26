#!/usr/bin/env python3
import argparse
import sys 
import os 
import json 
import argparse
notebook_dir = os.getcwd() 
sys.path.append(os.path.abspath(os.path.join(notebook_dir, '..', 'llm-negotiations'))) 
from envs.negotiation_env import NegotiationEnv 
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.extras.vllm_client import VLLMClient
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import tqdm



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
        temperature = 1.0

    elif args.game_type == "multi-game":
        temperature = 0.0

    config = {
        "game": game,
        "seed": 42
    }

    config = DictConfig(config)

    env = NegotiationEnv(config)
    #Divided by two because each prompt is used for two games (one as starting agent and one as responder)
    dataset = env.create_dataset(args.num_samples // 2)
    reward_functions = env.get_reward_functions()

    vllm_server_host = os.environ.get("VLLM_SERVER_HOST", "0.0.0.0")
    vllm_server_port = int(os.environ.get("VLLM_SERVER_PORT", 8000))
    vllm_client = VLLMClient(vllm_server_host, vllm_server_port, connection_timeout=120.0, enable_weight_sync=False)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize results data structure
    results = {
        "conversations": [],
        "rewards": [],
        "evaluations": []
    }
    
    # Process the dataset in batches
    batch_size = args.batch_size
    for i in range(0, len(dataset), batch_size):
        for starting_agent in [True, False]:
            batch = dataset[i:min(i+batch_size, len(dataset))]

            print(f"Batch: {batch}")
            print(f"Processing batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size}")
            
            # Prepare prompts
            prompts = batch["prompt"]
            prompts_2 = batch["prompt_2"] if "prompt_2" in batch else None

            #Here startng_agent is a list, i.e can be different for each example in the batch
            
            
            client_response = vllm_client.generate(
                prompts=prompts,
                prompts_2=prompts_2,
                n=1,  # Single generation per prompt for evaluation
                temperature=temperature,
                top_p=args.top_p if hasattr(args, 'top_p') else 1.0,
                top_k=args.top_k if hasattr(args, 'top_k') else 50,
                max_tokens=args.max_tokens if hasattr(args, 'max_tokens') else 200,
                starting_agent=starting_agent,
                sampled_h=None
            )
            full_conversations = client_response["conversations"]
        
            # Calculate rewards for each conversation
            batch_rewards = []
            for reward_func in reward_functions:
                # Prepare kwargs for reward function
                keys = [key for key in batch if key not in ["prompt", "completion"]]
                reward_kwargs = {key: batch[key] for key in keys}
                
                # Calculate rewards
                output_reward_func, evaluations = reward_func(prompts=prompts, completions=full_conversations, get_full_info=True, **reward_kwargs)
                batch_rewards.append(output_reward_func)
                results["evaluations"].append(evaluations)
                
            # Convert rewards to tensor and get sum
            rewards_per_func = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
            rewards = rewards_per_func.sum(dim=0).tolist()
            
            # Store conversations and rewards
            for conv, reward in zip(full_conversations, rewards):
                results["conversations"].append(conv)
                results["rewards"].append(reward)
                
            print(f"Processed batch with {len(batch)} examples. Average reward: {np.mean(rewards):.4f}")
    
    # Calculate overall statistics
    all_rewards = results["rewards"]
    mean_reward = np.mean(all_rewards)
    std_reward = np.std(all_rewards)
    
    print(f"\nEvaluation complete. Total conversations: {len(results['conversations'])}")
    print(f"Mean reward: {mean_reward:.4f}")
    print(f"Standard deviation: {std_reward:.4f}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save results to a JSON file
    output_path = os.path.join(args.output_dir, f"{args.model.replace('/', '_')}_results.json")
    with open(output_path, 'w') as f:
        json.dump({
            "model": args.model,
            "stats": {
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
                "num_samples": len(results["conversations"])
            },
            "game_type": args.game_type,
            "conversations": results["conversations"],
            "rewards": results["rewards"],
            "evaluations": results["evaluations"]
        }, f, indent=2)
    
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Evaluation script for multiturn LLM training")
    
    # Add arguments
    parser.add_argument("--model", type=str, required=True, help="Model to evaluate")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--game-type", type=str, default="generic-rental-agreement", help="Type of game to evaluate")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum number of tokens to generate")

    # Parse arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args)
