#!/usr/bin/env python3
"""Test script to verify that masking correctly identifies tokens for loss calculation.

This script loads data from a HuggingFace dataset, applies the masking as done in
the trainer (masks[:, 1:]), and decodes the remaining tokens to verify they
correspond to the correct parts of the conversation.

Usage:
    # Test first 5 samples from a dataset
    python tests/masking.py --dataset-name "username/dataset_name" --model-name "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    # Test specific samples
    python tests/masking.py --dataset-name "username/dataset_name" --model-name "meta-llama/Meta-Llama-3.1-8B-Instruct" --indices 0 1 2
    
    # With dataset config
    python tests/masking.py --dataset-name "username/dataset_name" --dataset-config "config_name" --model-name "meta-llama/Meta-Llama-3.1-8B-Instruct"
"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def apply_mask_to_tokens(tokens, mask):
    """
    Apply mask to tokens as done in trainer.py (line 282-283).
    
    The mask is applied starting from position 1 (masks[:, 1:]) because
    logprobs are computed for next token predictions (positions 1 to L-1).
    
    Args:
        tokens: List of token IDs (length L)
        mask: List of mask values (length L), 1 = keep, 0 = ignore
        
    Returns:
        List of token IDs that are kept (positions 1+ where mask[1:] == 1)
    """
    # Convert to tensors if needed
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.tensor(tokens, dtype=torch.long)
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.long)
    
    # Ensure mask is integer type (0 or 1)
    mask = mask.long()
    
    # Apply mask starting from position 1 (matching trainer.py masks[:, 1:])
    # tokens[1:] are the tokens we predict (next tokens)
    # mask[1:] are the mask values for those positions
    tokens_to_predict = tokens[1:]  # Shape: (L-1,)
    mask_for_predictions = mask[1:]  # Shape: (L-1,)
    
    # Keep only tokens where mask is 1 (assistant tokens)
    kept_tokens = tokens_to_predict[mask_for_predictions == 1]
    
    return kept_tokens.tolist()


def decode_masked_tokens(tokenizer, tokens, mask, show_full=False):
    """
    Decode tokens with masking applied.
    
    Args:
        tokenizer: Tokenizer to use for decoding
        tokens: List of token IDs
        mask: List of mask values
        show_full: If True, also show full conversation for context
        
    Returns:
        Dictionary with decoded text and token information
    """
    # Apply mask to get tokens that contribute to loss
    kept_tokens = apply_mask_to_tokens(tokens, mask)
    
    # Decode the kept tokens
    decoded_text = tokenizer.decode(kept_tokens, skip_special_tokens=False)
    
    result = {
        "kept_tokens": kept_tokens,
        "decoded_text": decoded_text,
        "num_kept_tokens": len(kept_tokens),
        "total_tokens": len(tokens),
        "mask_sum": int(sum(mask[1:])) if len(mask) > 1 else 0
    }
    
    if show_full:
        # Also decode full sequence for comparison
        full_decoded = tokenizer.decode(tokens, skip_special_tokens=False)
        result["full_decoded"] = full_decoded
    
    return result


def test_masking(dataset_name, dataset_config=None, model_name=None, num_samples=5, indices=None):
    """
    Test masking on samples from a HuggingFace dataset.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        dataset_config: Optional dataset configuration name
        model_name: Model name to load tokenizer (if None, tries to infer from dataset)
        num_samples: Number of samples to test (if indices is None)
        indices: Specific sample indices to test (overrides num_samples)
    """
    print(f"Loading dataset: {dataset_name}" + (f" (config: {dataset_config})" if dataset_config else ""))
    
    # Load dataset
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split="train")
    else:
        dataset = load_dataset(dataset_name, split="train")
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Get model name from dataset if not provided
    if model_name is None:
        if "model" in dataset[0]:
            model_name = dataset[0]["model"]
            print(f"Inferred model name from dataset: {model_name}")
        else:
            raise ValueError("model_name must be provided if not in dataset")
    
    # Load tokenizer
    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Determine which samples to test
    if indices is not None:
        sample_indices = indices
    else:
        sample_indices = list(range(min(num_samples, len(dataset))))
    
    print(f"\nTesting {len(sample_indices)} samples: {sample_indices}")
    print("=" * 80)
    
    for idx in sample_indices:
        if idx >= len(dataset):
            print(f"Warning: Index {idx} out of range (dataset has {len(dataset)} samples)")
            continue
            
        sample = dataset[idx]
        print(f"\n{'='*80}")
        print(f"SAMPLE {idx}")
        print(f"{'='*80}")
        
        # Extract data
        chosen_tokens = sample.get("chosen_token", [])
        reject_tokens = sample.get("reject_token", [])
        chosen_mask = sample.get("chosen_mask", [])
        reject_mask = sample.get("reject_mask", [])
        
        # Check if we have the required fields
        if not all([chosen_tokens, reject_tokens, chosen_mask, reject_mask]):
            print(f"Warning: Sample {idx} missing required fields")
            print(f"  chosen_token: {bool(chosen_tokens)}, reject_token: {bool(reject_tokens)}")
            print(f"  chosen_mask: {bool(chosen_mask)}, reject_mask: {bool(reject_mask)}")
            continue
        
        # Print sample metadata
        if "starting_agent" in sample:
            print(f"Starting agent: {sample['starting_agent']}")
        if "negotiation_role" in sample:
            print(f"Negotiation role: {sample['negotiation_role']}")
        if "sampled_h" in sample:
            print(f"Sampled h (REFUEL): {sample['sampled_h']}")
        
        # Test chosen conversation
        print(f"\n--- CHOSEN CONVERSATION ---")
        print(f"Total tokens: {len(chosen_tokens)}, Mask sum (positions 1+): {sum(chosen_mask[1:]) if len(chosen_mask) > 1 else 0}")
        chosen_result = decode_masked_tokens(tokenizer, chosen_tokens, chosen_mask, show_full=True)
        print(f"Tokens kept for loss: {chosen_result['num_kept_tokens']}")
        print(f"\nTokens that contribute to loss (decoded):")
        print(f"  {chosen_result['decoded_text']}")
        
        # Test reject conversation
        print(f"\n--- REJECT CONVERSATION ---")
        print(f"Total tokens: {len(reject_tokens)}, Mask sum (positions 1+): {sum(reject_mask[1:]) if len(reject_mask) > 1 else 0}")
        reject_result = decode_masked_tokens(tokenizer, reject_tokens, reject_mask, show_full=True)
        print(f"Tokens kept for loss: {reject_result['num_kept_tokens']}")
        print(f"\nTokens that contribute to loss (decoded):")
        print(f"  {reject_result['decoded_text']}")
        
        # Verify mask alignment
        print(f"\n--- MASK VERIFICATION ---")
        if len(chosen_tokens) != len(chosen_mask):
            print(f"  WARNING: chosen_tokens length ({len(chosen_tokens)}) != chosen_mask length ({len(chosen_mask)})")
        if len(reject_tokens) != len(reject_mask):
            print(f"  WARNING: reject_tokens length ({len(reject_tokens)}) != reject_mask length ({len(reject_mask)})")
        
        # Show mask statistics
        chosen_mask_ones = sum(chosen_mask[1:]) if len(chosen_mask) > 1 else 0
        reject_mask_ones = sum(reject_mask[1:]) if len(reject_mask) > 1 else 0
        print(f"  Chosen: {chosen_mask_ones} tokens contribute to loss out of {len(chosen_tokens) - 1} predictable tokens")
        print(f"  Reject: {reject_mask_ones} tokens contribute to loss out of {len(reject_tokens) - 1} predictable tokens")
        
        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test masking on HuggingFace dataset to verify loss calculation tokens"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the HuggingFace dataset"
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default=None,
        help="Optional dataset configuration name"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for tokenizer (if not provided, tries to infer from dataset)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to test"
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Specific sample indices to test (overrides --num-samples)"
    )
    
    args = parser.parse_args()
    
    test_masking(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        model_name=args.model_name,
        num_samples=args.num_samples,
        indices=args.indices
    )


if __name__ == "__main__":
    main()

