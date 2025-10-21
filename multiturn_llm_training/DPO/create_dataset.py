#!/usr/bin/env python3
import sys 
import os 
import json 
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

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


@dataclass
class ProcessingConfig:
    """Configuration for processing parameters."""
    max_retries: int = 5
    progress_report_interval: int = 100


class ComparisonResult(Enum):
    """Enum for reward comparison results."""
    FIRST_BETTER = "first_better"
    SECOND_BETTER = "second_better"
    EQUAL = "equal"


@dataclass
class GenerationResult:
    """Data class to hold generation results."""
    conversations: List[str]
    token_ids: List[List[int]]
    assistant_masks: List[List[bool]]
    token_count: int


@dataclass
class DPOExample:
    """Data class for a single DPO training example."""
    game: str
    issues: List[str]
    starting_agent: bool
    negotiation_role: str
    chosen: str
    reject: str
    chosen_token: List[int]
    reject_token: List[int]
    chosen_mask: List[bool]
    reject_mask: List[bool]
    chosen_reward: float
    reject_reward: float




def sample_geometric_bounded(p: float, max_value: int) -> int:
    """Sample from geometric distribution with upper bound."""
    while True:
        sample = np.random.geometric(p) - 1
        if sample <= max_value:
            return sample

def setup_environment(args) -> Tuple[NegotiationEnv, VLLMClient, List[Any]]:
    """Set up the negotiation environment and VLLM client."""
    env = NegotiationEnv(game_type=args.game_type)
    print("Environment created")
    reward_functions = env.get_reward_functions()
    print("Reward functions created")
    vllm_server_host = os.environ.get("VLLM_SERVER_HOST", "0.0.0.0")
    vllm_server_port = int(os.environ.get("VLLM_SERVER_PORT", 8000))
    print(f"VLLM server host: {vllm_server_host}, VLLM server port: {vllm_server_port}")
    vllm_client = VLLMClient(vllm_server_host, vllm_server_port, connection_timeout=120.0)
    print("VLLM client created")
    return env, vllm_client, reward_functions


def generate_conversations(vllm_client: VLLMClient, item: Dict[str, Any], args, is_starting_agent: bool) -> GenerationResult:
    """Generate conversations using the VLLM client."""
    prompt = item["prompt"]
    prompt_2 = item.get("prompt_2")
    game_config = item.get("game_config")
    
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
        sampled_h=None
    )
    
    return GenerationResult(
        conversations=client_response["conversations"],
        token_ids=client_response["token_ids"],
        assistant_masks=client_response["assistant_masks"],
        token_count=client_response["total_token_count"][0]
    )


def calculate_rewards(reward_functions: List[Any], item: Dict[str, Any], conversations: List[str]) -> List[float]:
    """Calculate rewards for the generated conversations."""
    rewards = []
    
    # Process reward for each conversation (2 conversations)
    for i in range(2):
        conversation_rewards = []
        for reward_func in reward_functions:
            reward_kwargs = {k: v for k, v in item.items() if k not in ["prompt", "completion"]}
            output_reward_func, evaluations = reward_func(
                prompts=[item["prompt"]],
                completions=[conversations[i]],
                get_full_info=True,
                negotiation_roles=[item.get("negotiation_role")],
                game_configs=[item.get("game_config")],
                **reward_kwargs
            )
            reward = output_reward_func[0]  # Extract single value from list
            conversation_rewards.append(reward)
        
        # Sum all reward functions for this conversation
        total_reward = sum(conversation_rewards)
        rewards.append(total_reward)
    
    return rewards


def compare_rewards(rewards: List[float]) -> ComparisonResult:
    """Compare two rewards and return the comparison result."""
    if len(rewards) != 2:
        raise ValueError("Expected exactly 2 rewards for comparison")
    
    if rewards[0] > rewards[1]:
        return ComparisonResult.FIRST_BETTER
    elif rewards[0] < rewards[1]:
        return ComparisonResult.SECOND_BETTER
    else:
        return ComparisonResult.EQUAL


def create_dpo_example(item: Dict[str, Any], generation_result: GenerationResult, 
                      rewards: List[float], comparison_result: ComparisonResult, 
                      is_starting_agent: bool) -> DPOExample:
    """Create a DPO training example from the results."""
    game_config = item.get("game_config", {})
    
    if comparison_result == ComparisonResult.FIRST_BETTER:
        chosen_idx, reject_idx = 0, 1
    else:  # SECOND_BETTER
        chosen_idx, reject_idx = 1, 0
    
    return DPOExample(
        game=game_config.get("name", ""),
        issues=game_config.get("issues", []),
        starting_agent=is_starting_agent,
        negotiation_role=item.get("negotiation_role", ""),
        chosen=generation_result.conversations[chosen_idx],
        reject=generation_result.conversations[reject_idx],
        chosen_token=generation_result.token_ids[chosen_idx],
        reject_token=generation_result.token_ids[reject_idx],
        chosen_mask=generation_result.assistant_masks[chosen_idx],
        reject_mask=generation_result.assistant_masks[reject_idx],
        chosen_reward=rewards[chosen_idx],
        reject_reward=rewards[reject_idx]
    )


def process_sample(item: Dict[str, Any], vllm_client: VLLMClient, reward_functions: List[Any], 
                  args, is_starting_agent: bool, sample_num: int, total_samples: int, 
                  config: ProcessingConfig, num_tries: int = 0) -> Tuple[Optional[DPOExample], int]:
    """Process a single sample and return a DPO example and token count if successful."""
    print(f"Processing {'starting agent' if is_starting_agent else 'responder'} sample {sample_num}/{total_samples}")
    
    if num_tries > config.max_retries:
        print(f"Skipping sample {sample_num} after {num_tries} tries")
        return None, 0
    
    try:
        # Generate conversations
        generation_result = generate_conversations(vllm_client, item, args, is_starting_agent)
        
        # Calculate rewards
        rewards = calculate_rewards(reward_functions, item, generation_result.conversations)
        
        # Compare rewards
        comparison_result = compare_rewards(rewards)
        
        if comparison_result == ComparisonResult.EQUAL:
            # Retry if rewards are equal
            return process_sample(item, vllm_client, reward_functions, args, is_starting_agent, 
                               sample_num, total_samples, config, num_tries + 1)
        
        # Create DPO example
        dpo_example = create_dpo_example(item, generation_result, rewards, comparison_result, is_starting_agent)
        
        print(f"Processed {'starting agent' if is_starting_agent else 'responder'} sample {sample_num}. "
              f"Rewards: {rewards[0]:.4f}, {rewards[1]:.4f}")
        
        return dpo_example, generation_result.token_count
        
    except Exception as e:
        print(f"Error processing sample {sample_num}: {str(e)}")
        if num_tries < config.max_retries:
            return process_sample(item, vllm_client, reward_functions, args, is_starting_agent, 
                               sample_num, total_samples, config, num_tries + 1)
        return None, 0


def process_dataset(dataset: List[Dict[str, Any]], vllm_client: VLLMClient, reward_functions: List[Any], 
                   args, is_starting_agent: bool, config: ProcessingConfig) -> Tuple[List[DPOExample], int]:
    """Process a dataset and return DPO examples and total token count."""
    dpo_examples = []
    total_token_count = 0
    
    for i, item in enumerate(tqdm.tqdm(dataset, desc=f"{'Starting Agent' if is_starting_agent else 'Responder'} Samples")):
        dpo_example, token_count = process_sample(item, vllm_client, reward_functions, args, is_starting_agent, 
                                   i + 1, len(dataset), config)
        
        if dpo_example:
            dpo_examples.append(dpo_example)
            total_token_count += token_count
        
        if (i + 1) % config.progress_report_interval == 0:
            print(f"Current number of tokens: {total_token_count}")
    
    return dpo_examples, total_token_count


def convert_to_dict_format(dpo_examples: List[DPOExample], args) -> Dict[str, Any]:
    """Convert DPO examples to the dictionary format for saving."""
    return {
        "model": args.model,
        "game_type": args.game_type,
        "starting_agent": [ex.starting_agent for ex in dpo_examples],
        "negotiation_role": [ex.negotiation_role for ex in dpo_examples],
        "chosen": [ex.chosen for ex in dpo_examples],
        "reject": [ex.reject for ex in dpo_examples],
        "chosen_token": [ex.chosen_token for ex in dpo_examples],
        "reject_token": [ex.reject_token for ex in dpo_examples],
        "chosen_mask": [ex.chosen_mask for ex in dpo_examples],
        "reject_mask": [ex.reject_mask for ex in dpo_examples],
        "chosen_reward": [ex.chosen_reward for ex in dpo_examples],
        "reject_reward": [ex.reject_reward for ex in dpo_examples],
    }


def upload_to_huggingface(dict_to_save: Dict[str, Any], args) -> None:
    """Upload the dataset to Hugging Face."""
    if not args.hf_repo:
        return
    
    try:
        # Convert to dataset format
        dataset_dict = {key: value for key, value in dict_to_save.items() if isinstance(value, list)}
        hf_dataset = Dataset.from_dict(dataset_dict)
        
        # Upload to Hugging Face
        dataset_name = f"{args.game_type}_{args.model.replace('/', '_')}"
        print(f"Uploading to Hugging Face repository: {args.hf_repo}, dataset: {dataset_name}")
        
        api = HfApi()
        api_token = os.environ.get("HF_TOKEN")
        if not api_token:
            print("Warning: HF_TOKEN environment variable not set. Using anonymous upload.")
        
        hf_dataset.push_to_hub(args.hf_repo, dataset_name, token=api_token)
        print(f"Dataset successfully uploaded to: https://huggingface.co/datasets/{args.hf_repo}/{dataset_name}")
        
    except Exception as e:
        print(f"Error uploading to Hugging Face: {str(e)}")


def main(args):
    """Main function that executes the program logic."""
    print(f"Running with the following arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Setup
    env, vllm_client, reward_functions = setup_environment(args)
    print("Environment setup complete")
    config = ProcessingConfig()

    
    # Create and filter datasets
    dataset = env.create_dataset(args.num_samples)
    print("Dataset created")
    dataset_starting_agent = [d for d in dataset if d["starting_agent"]]
    dataset_responder = [d for d in dataset if not d["starting_agent"]]
    print("Datasets filtered")
    # Process datasets
    print("\nProcessing starting agent dataset...")
    starting_agent_examples, starting_agent_tokens = process_dataset(dataset_starting_agent, vllm_client, reward_functions, 
                                             args, True, config)

    print("\nProcessing responder dataset...")
    responder_examples, responder_tokens = process_dataset(dataset_responder, vllm_client, reward_functions, 
                                       args, False, config)
    
    # Combine results
    all_examples = starting_agent_examples + responder_examples
    total_token_count = starting_agent_tokens + responder_tokens
    print(f"\nGeneration complete. Total examples: {len(all_examples)}")
    print(f"Total token count: {total_token_count}")
    
    # Convert and save
    dict_to_save = convert_to_dict_format(all_examples, args)
    upload_to_huggingface(dict_to_save, args)

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Evaluation script for multiturn LLM training")
    
    # Add arguments
    parser.add_argument("--model", type=str, required=True, help="Model to evaluate")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--game-type", type=str, default="generic-rental-agreement", help="Type of game to evaluate")
    parser.add_argument("--hf-repo", type=str, help="Hugging Face repository ID to upload the dataset (e.g., 'username/dataset-name')")


    # Parse arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    main(args)
