#!/usr/bin/env python3
import os
import argparse

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, login
import torch
from peft import PeftModel
from transformers import Trainer, TrainingArguments

def main():
    parser = argparse.ArgumentParser(description="Upload DeepSpeed checkpoint to Hugging Face Hub")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory containing model checkpoints")
    parser.add_argument("--base_model", type=str, required=True, help="Base model name from Hugging Face")
    parser.add_argument("--model_names", nargs='+', required=True, help="List of model names to upload")
    parser.add_argument("--hf_username", type=str, default="LuckyLukke", help="HuggingFace username")
    parser.add_argument("--token", type=str, help="HuggingFace token (optional, can use HF_TOKEN env var)")
    args = parser.parse_args()
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    
    # Login to Hugging Face if token is provided
    if args.token:
        login(token=args.token)
    elif "HF_TOKEN" in os.environ:
        login(token=os.environ["HF_TOKEN"])
    
    # Initialize accelerator
    accelerator = Accelerator()

    if isinstance(args.model_names, str):
        args.model_names = [args.model_names]
    
    for name in args.model_names:
        # Construct full path to model directory
        model_dir = os.path.join(args.output_dir, name)
        print(f"Processing model: {name}")
        print(f"Loading from directory: {model_dir}")
        
        changed_name = name.replace("/checkpoint", "")
        # Construct repo name
        repo_name = f"{args.hf_username}/{changed_name}"
        
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            padding_side='right',
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )   

        # Use the full model_dir path directly
        model = PeftModel.from_pretrained(model, model_dir)

        print("Merging and unloading model")
        model = model.merge_and_unload()

        print(f"Pushing model to hub: {repo_name}")
        
        model.push_to_hub(repo_name, safe_serialization=True, use_temp_dir=True)
        tokenizer.push_to_hub(repo_name, safe_serialization=True, use_temp_dir=True)
        print(f"✅ Model successfully uploaded to https://huggingface.co/{repo_name}")


if __name__ == "__main__":
    main()