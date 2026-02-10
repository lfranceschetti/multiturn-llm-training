#!/usr/bin/env python3
"""REFUEL dataset creation script."""
import sys 
import os 
import argparse
import time
from typing import List, Dict, Any

notebook_dir = os.getcwd() 
sys.path.append(os.path.abspath(os.path.join(notebook_dir, '..', '..', 'llm-negotiations')))
from trl.extras.vllm_client import VLLMClient

from multiturn_llm_training.utils.create_offline_dataset import (
    ProcessingConfig,
    GenerationResult,
    Sample,
    sample_geometric_bounded,
    setup_environment,
    calculate_rewards,
    upload_to_huggingface,
    convert_to_dict_format,
    process_dataset
)


# ============================================================================
# REFUEL-Specific Functions
# ============================================================================

def generate_conversations(vllm_client: VLLMClient, item: Dict[str, Any], args, is_starting_agent: bool) -> GenerationResult:
    """Generate conversations using the VLLM client (REFUEL: samples sampled_h)."""
    prompt = item["prompt"]
    prompt_2 = item.get("prompt_2")

    sampled_h = sample_geometric_bounded(p=0.3, max_value=4)
    
    prompts = [prompt, prompt]
    prompts_2 = [prompt_2, prompt_2] if prompt_2 else None

    
    client_response = vllm_client.generate(
        prompts=prompts,
        prompts_2=prompts_2,
        top_p=args.top_p if hasattr(args, 'top_p') else 1.0,
        top_k=args.top_k if hasattr(args, 'top_k') else 50,
        max_completion_length=args.max_tokens if hasattr(args, 'max_tokens') else 200,
        starting_agent=is_starting_agent,
        sampled_h=sampled_h  # REFUEL: use sampled h
    )
    
    return GenerationResult(
        conversations=client_response["conversations"],
        token_ids=client_response["token_ids"],
        assistant_masks=client_response["assistant_masks"],
        generated_tokens_agent=client_response["generated_tokens_agent"],
        generated_tokens_opp=client_response["generated_tokens_opp"],
        sampled_h=sampled_h  # REFUEL: store sampled_h
    )


# ============================================================================
# Main Processing
# ============================================================================

def main(args):
    """Main function that executes the program logic."""
    start_time = time.perf_counter()
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
    starting_agent_samples, starting_agent_discarded_samples = process_dataset(
        dataset_starting_agent, vllm_client, reward_functions, args, True, config, generate_conversations
    )

    print("\nProcessing responder dataset...")
    responder_samples, responder_discarded_samples = process_dataset(
        dataset_responder, vllm_client, reward_functions, args, False, config, generate_conversations
    )
    
    # Combine results
    all_samples = starting_agent_samples + responder_samples
    
    # Calculate total tokens from samples
    total_token_count = sum(s.generated_tokens_agent + s.generated_tokens_opp for s in all_samples)
    print(f"\nGeneration complete. Total samples: {len(all_samples)}")
    print(f"Total token count: {total_token_count}")
    
    # Calculate execution time before upload
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    args.elapsed_time_seconds = elapsed_time  # Store in args for metadata upload
    print(f"\n{'='*60}")
    print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"{'='*60}")
    
    # Convert and save
    print(f"\nConverting {len(all_samples)} samples to dictionary format...")
    dict_to_save = convert_to_dict_format(all_samples, args)
    print(f"Dictionary created with keys: {list(dict_to_save.keys())}")
    print(f"Dictionary values lengths: {[(k, len(v) if isinstance(v, list) else 'N/A') for k, v in dict_to_save.items()]}")
    print(f"\nAttempting to upload to Hugging Face...")
    print(f"HF_REPO argument: {args.hf_repo}")
    upload_to_huggingface(dict_to_save, args)
    print("Upload function completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REFUEL dataset creation script")
    parser.add_argument("--model", type=str, required=True, help="Model to evaluate")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--game-type", type=str, default="generic-rental-agreement", help="Type of game to evaluate")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum number of tokens to generate")
    parser.add_argument("--hf-repo", type=str, help="Hugging Face repository ID to upload the dataset")
    args = parser.parse_args()
    main(args)
