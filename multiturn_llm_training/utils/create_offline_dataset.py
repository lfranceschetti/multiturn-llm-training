#!/usr/bin/env python3
"""Common utilities for dataset creation in DPO and REFUEL training."""
import os
import json
import tempfile
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Union, Optional, Callable, TypeVar, Protocol
from enum import Enum
import numpy as np
import tqdm

from envs.negotiation.env import NegotiationEnv
from trl.extras.vllm_client import VLLMClient
from datasets import Dataset
from huggingface_hub import HfApi

# ============================================================================
# Configuration and Types
# ============================================================================

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
    """Data class to hold generation results.
    
    Works for both DPO (sampled_h=None) and REFUEL (sampled_h=int).
    """
    conversations: List[str]
    token_ids: List[List[int]]
    assistant_masks: List[List[bool]]
    generated_tokens: List[int]
    sampled_h: Optional[int] = None  # REFUEL-specific, None for DPO


@dataclass
class Sample:
    """Unified sample class for both DPO and REFUEL training.
    
    sampled_h is None for DPO samples and an int for REFUEL samples.
    """
    game: str
    issues: List[str]
    game_settings: str
    issue_weights: List[List[int]]
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
    chosen_generated_tokens: int
    reject_generated_tokens: int
    sampled_h: Optional[int] = None  # REFUEL-specific, None for DPO


# ============================================================================
# Environment Setup
# ============================================================================

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


# ============================================================================
# Reward Calculation
# ============================================================================

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


# ============================================================================
# Sample Creation
# ============================================================================

def sample_geometric_bounded(p: float, max_value: int) -> int:
    """Sample from geometric distribution with upper bound.
    
    Args:
        p: Probability parameter for geometric distribution
        max_value: Maximum value to return
        
    Returns:
        Sampled integer value between 0 and max_value (inclusive)
    """
    while True:
        sample = np.random.geometric(p) - 1
        if sample <= max_value:
            return sample


def create_sample(item: Dict[str, Any], generation_result: GenerationResult, 
                  rewards: List[float], comparison_result: ComparisonResult, 
                  is_starting_agent: bool) -> Sample:
    """Create a Sample from generation results.
    
    Works for both DPO and REFUEL. If generation_result has sampled_h, it will be included.
    
    Args:
        item: Original item with game_config and negotiation_role
        generation_result: GenerationResult with conversations, tokens, etc.
        rewards: List of rewards for the two conversations
        comparison_result: Which conversation is better
        is_starting_agent: Whether this is for the starting agent
        
    Returns:
        Sample object (DPO if sampled_h is None, REFUEL if sampled_h is set)
    """
    game_config = item.get("game_config", {})
    
    # Extract game_settings and remove .yaml extension if present
    game_settings = game_config.get("game_settings", "")
    if game_settings.endswith(".yaml"):
        game_settings = game_settings[:-5]  # Remove .yaml
    
    if comparison_result == ComparisonResult.FIRST_BETTER:
        chosen_idx, reject_idx = 0, 1
    else:  # SECOND_BETTER
        chosen_idx, reject_idx = 1, 0
    
    return Sample(
        game=game_config.get("name", ""),
        issues=game_config.get("issues", []),
        game_settings=game_settings,
        issue_weights=game_config.get("issue_weights", []),
        starting_agent=is_starting_agent,
        negotiation_role=item.get("negotiation_role", ""),
        chosen=generation_result.conversations[chosen_idx],
        reject=generation_result.conversations[reject_idx],
        chosen_token=generation_result.token_ids[chosen_idx],
        reject_token=generation_result.token_ids[reject_idx],
        chosen_mask=generation_result.assistant_masks[chosen_idx],
        reject_mask=generation_result.assistant_masks[reject_idx],
        chosen_reward=rewards[chosen_idx],
        reject_reward=rewards[reject_idx],
        chosen_generated_tokens=generation_result.generated_tokens[chosen_idx],
        reject_generated_tokens=generation_result.generated_tokens[reject_idx],
        sampled_h=generation_result.sampled_h  # Will be None for DPO, int for REFUEL
    )


# ============================================================================
# Dataset Processing
# ============================================================================

def process_sample(
    item: Dict[str, Any],
    vllm_client: VLLMClient,
    reward_functions: List[Any],
    args: Any,
    is_starting_agent: bool,
    sample_num: int,
    total_samples: int,
    config: ProcessingConfig,
    generate_fn: Callable[[VLLMClient, Dict[str, Any], Any, bool], GenerationResult],
    num_tries: int = 0
) -> Optional[Sample]:
    """Process a single sample and return a sample if successful.
    
    Works for both DPO and REFUEL. The generate_fn determines whether sampled_h is used.
    
    Args:
        item: The sample item to process
        vllm_client: VLLM client for generation
        reward_functions: List of reward functions
        args: Arguments object
        is_starting_agent: Whether this is for the starting agent
        sample_num: Current sample number
        total_samples: Total number of samples
        config: Processing configuration
        generate_fn: Function to generate conversations (returns GenerationResult)
        num_tries: Number of retry attempts
        
    Returns:
        Sample object or None if failed after max retries
    """
    print(f"Processing {'starting agent' if is_starting_agent else 'responder'} sample {sample_num}/{total_samples}")
    
    if num_tries > config.max_retries:
        print(f"Skipping sample {sample_num} after {num_tries} tries")
        return None
    
    try:
        # Generate conversations
        generation_result = generate_fn(vllm_client, item, args, is_starting_agent)
        
        # Calculate rewards
        rewards = calculate_rewards(reward_functions, item, generation_result.conversations)
        print(f"Rewards: {rewards}")

        # Compare rewards
        comparison_result = compare_rewards(rewards)
        
        if comparison_result == ComparisonResult.EQUAL:
            # Rewards are equal, retry generation
            print(f"Rewards are equal ({rewards[0]:.4f}), retrying generation...")
            return process_sample(
                item, vllm_client, reward_functions, args, is_starting_agent,
                sample_num, total_samples, config, generate_fn, num_tries + 1
            )
        
        # Create sample (rewards are different)
        sample = create_sample(item, generation_result, rewards, comparison_result, is_starting_agent)
        
        print(f"Processed {'starting agent' if is_starting_agent else 'responder'} sample {sample_num}. "
              f"Rewards: {rewards[0]:.4f}, {rewards[1]:.4f}")
        
        return sample
        
    except Exception as e:
        print(f"Error processing sample {sample_num}: {str(e)}")
        if num_tries < config.max_retries:
            return process_sample(
                item, vllm_client, reward_functions, args, is_starting_agent,
                sample_num, total_samples, config, generate_fn, num_tries + 1
            )
        return None


def process_dataset(
    dataset: List[Dict[str, Any]],
    vllm_client: VLLMClient,
    reward_functions: List[Any],
    args: Any,
    is_starting_agent: bool,
    config: ProcessingConfig,
    generate_fn: Callable[[VLLMClient, Dict[str, Any], Any, bool], GenerationResult]
) -> List[Sample]:
    """Process a dataset and return samples.
    
    Works for both DPO and REFUEL. The generate_fn determines whether sampled_h is used.
    
    Args:
        dataset: List of sample items
        vllm_client: VLLM client for generation
        reward_functions: List of reward functions
        args: Arguments object
        is_starting_agent: Whether this is for the starting agent
        config: Processing configuration
        generate_fn: Function to generate conversations (returns GenerationResult)
        
    Returns:
        List of Sample objects
    """
    samples = []
    
    for i, item in enumerate(tqdm.tqdm(dataset, desc=f"{'Starting Agent' if is_starting_agent else 'Responder'} Samples")):
        sample = process_sample(
            item, vllm_client, reward_functions, args, is_starting_agent,
            i + 1, len(dataset), config, generate_fn
        )
        
        if sample:
            samples.append(sample)
        
        if (i + 1) % config.progress_report_interval == 0:
            # Calculate total tokens from samples if needed for reporting
            total_tokens = sum(s.chosen_generated_tokens + s.reject_generated_tokens for s in samples)
            print(f"Processed {len(samples)} samples. Total tokens: {total_tokens}")
    
    return samples


# ============================================================================
# Data Format Conversion
# ============================================================================

def convert_to_dict_format(samples: List[Sample], args) -> Dict[str, Any]:
    """Convert training samples to the dictionary format for saving.
    
    This function works with both DPO and REFUEL samples. If samples have a 
    'sampled_h' attribute (REFUEL), it will be included in the output.
    
    Args:
        samples: List of training samples (Sample objects)
        args: Arguments object containing model and game_type
        
    Returns:
        Dictionary with all sample data formatted for saving
    """
    # NOTE: The order of keys in this dictionary determines the column order in the HuggingFace dataset table.
    # Reorder the keys below to change the column order in the uploaded dataset.
    result = {
        "model": args.model,
        "chosen": [s.chosen for s in samples],
        "reject": [s.reject for s in samples],
    }
    
    # Include sampled_h if any samples have it (REFUEL-specific)
    # Placed after chosen and reject as requested
    if samples and samples[0].sampled_h is not None:
        result["sampled_h"] = [s.sampled_h for s in samples]
    
    # Continue with remaining fields
    result.update({
        "game_type": args.game_type,
        "starting_agent": [s.starting_agent for s in samples],
        "negotiation_role": [s.negotiation_role for s in samples],
        "game_settings": [s.game_settings for s in samples],
        "issues": [s.issues for s in samples],
        "issue_weights": [s.issue_weights for s in samples],
        "chosen_token": [s.chosen_token for s in samples],
        "reject_token": [s.reject_token for s in samples],
        "chosen_mask": [s.chosen_mask for s in samples],
        "reject_mask": [s.reject_mask for s in samples],
        "chosen_reward": [s.chosen_reward for s in samples],
        "reject_reward": [s.reject_reward for s in samples],
        "chosen_generated_tokens": [s.chosen_generated_tokens for s in samples],
        "reject_generated_tokens": [s.reject_generated_tokens for s in samples],
    })
    
    return result


# ============================================================================
# Hugging Face Upload
# ============================================================================

def upload_to_huggingface(dict_to_save: Dict[str, Any], args) -> None:
    """Upload the dataset to Hugging Face."""
    print(f"[DEBUG] upload_to_huggingface called")
    print(f"[DEBUG] args.hf_repo = {args.hf_repo}")
    print(f"[DEBUG] bool(args.hf_repo): {bool(args.hf_repo)}")
    
    if not args.hf_repo:
        print("[DEBUG] No HF_REPO provided, skipping upload")
        return
    
    print(f"[DEBUG] HF_REPO provided: {args.hf_repo}, proceeding with upload")
    
    try:
        # Convert to dataset format
        print(f"[DEBUG] Filtering dictionary to list values only...")
        dataset_dict = {key: value for key, value in dict_to_save.items() if isinstance(value, list)}
        print(f"[DEBUG] Filtered dataset_dict keys: {list(dataset_dict.keys())}")
        print(f"[DEBUG] Filtered dataset_dict values lengths: {[(k, len(v)) for k, v in dataset_dict.items()]}")
        
        print(f"[DEBUG] Creating Hugging Face Dataset object...")
        hf_dataset = Dataset.from_dict(dataset_dict)
        print(f"[DEBUG] Dataset created with {len(hf_dataset)} examples")
        
        # Upload to Hugging Face
        dataset_name = f"{args.game_type}_{args.model.replace('/', '_')}"
        print(f"[DEBUG] Dataset name: {dataset_name}")
        print(f"[DEBUG] Uploading to Hugging Face repository: {args.hf_repo}, dataset: {dataset_name}")
        
        api = HfApi()
        api_token = os.environ.get("HF_TOKEN")
        print(f"[DEBUG] HF_TOKEN from environment: {'SET' if api_token else 'NOT SET'}")
        if not api_token:
            print("Warning: HF_TOKEN environment variable not set. Using anonymous upload.")
        
        print(f"[DEBUG] Calling push_to_hub...")
        hf_dataset.push_to_hub(args.hf_repo, dataset_name, token=api_token)
        print(f"[DEBUG] push_to_hub completed successfully")
        
        # Create and upload generation settings metadata file
        print(f"[DEBUG] Creating generation settings metadata file...")
        generation_settings = {
            "model": args.model,
            "game_type": args.game_type,
            "generation_settings": {
                "temperature": getattr(args, 'temperature', 1.0),
                "top_p": getattr(args, 'top_p', 1.0),
                "top_k": getattr(args, 'top_k', 50),
                "max_tokens": getattr(args, 'max_tokens', 200),
            },
            "num_samples": getattr(args, 'num_samples', None),
        }
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(generation_settings, f, indent=2)
            temp_file_path = f.name
        
        try:
            # Upload metadata file to dataset repository
            metadata_path = f"{dataset_name}/generation_settings.json"
            print(f"[DEBUG] Uploading metadata file to {metadata_path}...")
            api.upload_file(
                path_or_fileobj=temp_file_path,
                path_in_repo=metadata_path,
                repo_id=args.hf_repo,
                repo_type="dataset",
                token=api_token
            )
            print(f"[DEBUG] Metadata file uploaded successfully")
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
        
        print(f"Dataset successfully uploaded to: https://huggingface.co/datasets/{args.hf_repo}/{dataset_name}")
        print(f"Generation settings metadata available at: {metadata_path}")
        
    except Exception as e:
        print(f"[DEBUG] Exception caught in upload_to_huggingface")
        print(f"Error uploading to Hugging Face: {str(e)}")
        import traceback
        print(f"[DEBUG] Full traceback:")
        traceback.print_exc()
