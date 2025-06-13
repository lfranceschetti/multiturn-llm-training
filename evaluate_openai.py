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


    #Divided by two because each prompt is used for two games (one as starting agent and one as responder)
    dataset = env.create_dataset(args.num_samples)

    #Filter the dataset to only include examples where starting_agent is True
    dataset_starting_agent = [d for d in dataset if d["starting_agent"]]
    dataset_responder = [d for d in dataset if not d["starting_agent"]]

    reward_functions = env.get_reward_functions()

    vllm_server_host = os.environ.get("VLLM_SERVER_HOST", "0.0.0.0")
    vllm_server_port = int(os.environ.get("VLLM_SERVER_PORT", 8000))
    vllm_client = VLLMClient(vllm_server_host, vllm_server_port, connection_timeout=120.0, enable_weight_sync=False)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize results data structure
    results = {
        "conversations": [],
        "rewards": [],
        "evaluations": [],
        "starting_agent": [],
        "negotiation_role": []
    }

    
    def process_batch(batch, is_starting_agent, batch_num, total_batches):
        """Process a batch of examples and return the results.
        
        Args:
            batch: The batch of examples to process (list of dictionaries)
            is_starting_agent: Whether this batch is for the starting agent
            batch_num: Current batch number
            total_batches: Total number of batches
        """
        print(f"Processing {'starting agent' if is_starting_agent else 'responder'} batch {batch_num}/{total_batches}")
        
        # Extract prompts and other data from the batch
        prompts = [item["prompt"] for item in batch]
        prompts_2 = [item.get("prompt_2") for item in batch] if any("prompt_2" in item for item in batch) else None
        negotiation_roles = [item.get("negotiation_role") for item in batch] if any("negotiation_role" in item for item in batch) else None
        game_configs = [item.get("game_config") for item in batch] if any("game_config" in item for item in batch) else None

        client_response = vllm_client.generate(
            prompts=prompts,
            prompts_2=prompts_2,
            n=1,
            temperature=1.0,
            top_p=args.top_p if hasattr(args, 'top_p') else 1.0,
            top_k=args.top_k if hasattr(args, 'top_k') else 50,
            max_tokens=args.max_tokens if hasattr(args, 'max_tokens') else 200,
            starting_agent=is_starting_agent,
            sampled_h=None
        )
        full_conversations = client_response["conversations"]
        
        batch_rewards = []
        for reward_func in reward_functions:
            # Process each item in the batch separately
            for item in batch:
                reward_kwargs = {k: v for k, v in item.items() if k not in ["prompt", "completion"]}

                output_reward_func, evaluations = reward_func(
                    prompts=[item["prompt"]],
                    completions=[full_conversations[batch.index(item)]],
                    get_full_info=True,
                    negotiation_roles=[item.get("negotiation_role")],
                    game_configs=[item.get("game_config")],
                    **reward_kwargs
                )
                batch_rewards.append(output_reward_func[0])  # Extract single value from list
                results["evaluations"].extend(evaluations)
            
        # Convert rewards to tensor and get sum for each item
        rewards_per_func = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
        rewards = rewards_per_func.tolist()  # Keep as list of rewards
        
        # Store conversations and rewards
        for conv, reward, item in zip(full_conversations, rewards, batch):
            results["conversations"].append(conv)
            results["rewards"].append(reward)
            results["starting_agent"].append(is_starting_agent)
            results["negotiation_role"].append(item.get("negotiation_role"))

            
        print(f"Processed {'starting agent' if is_starting_agent else 'responder'} batch with {len(batch)} examples. Average reward: {np.mean(rewards):.4f}")
    
    # Process the dataset in batches - first starting agent, then responder
    batch_size = args.batch_size
    
    # Process starting agent dataset
    print("\nProcessing starting agent dataset...")
    total_starting_batches = (len(dataset_starting_agent) + batch_size - 1) // batch_size
    for i in tqdm.tqdm(range(0, len(dataset_starting_agent), batch_size), 
                      total=total_starting_batches,
                      desc="Starting Agent Batches"):
        batch = dataset_starting_agent[i:min(i+batch_size, len(dataset_starting_agent))]
        process_batch(batch, True, i//batch_size + 1, total_starting_batches)

    # Process responder dataset
    print("\nProcessing responder dataset...")
    total_responder_batches = (len(dataset_responder) + batch_size - 1) // batch_size
    for i in tqdm.tqdm(range(0, len(dataset_responder), batch_size),
                      total=total_responder_batches,
                      desc="Responder Batches"):
        batch = dataset_responder[i:min(i+batch_size, len(dataset_responder))]
        process_batch(batch, False, i//batch_size + 1, total_responder_batches)
    
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
        dict_to_save = {
            "model": args.model,
            "stats": {
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
                "num_samples": len(results["conversations"])
            },
            "game_type": args.game_type,
            "conversations": results["conversations"],
            "rewards": results["rewards"],
            "evaluations": results["evaluations"],
            "starting_agent": results["starting_agent"],
            "negotiation_role": results["negotiation_role"]
        }


        json.dump(dict_to_save, f, indent=2)

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
