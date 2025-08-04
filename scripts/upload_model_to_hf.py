
#!/usr/bin/env python3
import os
import argparse

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, login
import torch
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

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
    
    for name in args.model_names:
        # Construct full path to model directory
        model_dir = os.path.join(args.output_dir, name)
        print(f"Processing model: {name}")
        print(f"Loading from directory: {model_dir}")
        
        # Construct repo name
        repo_name = f"{args.hf_username}/{name}"
        
        try:
            # Load the model architecture from the base model
            print(f"Loading base model: {args.base_model}")
            model = AutoModelForCausalLM.from_pretrained(args.base_model)
            model = accelerator.unwrap_model(model)
            
            # Load the training state
            print(f"Loading checkpoint from: {model_dir}")
            model = load_state_dict_from_zero_checkpoint(model, model_dir)
            
            # Load tokenizer
            print("Loading tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(args.base_model)
            
            # Upload model and tokenizer to Hub
            print(f"Pushing model to {repo_name}")
            model.push_to_hub(repo_name)
            tokenizer.push_to_hub(repo_name)
            
            print(f"✅ Model successfully uploaded to https://huggingface.co/{repo_name}")

            del model
            del tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error processing {name}: {str(e)}")

if __name__ == "__main__":
    main()