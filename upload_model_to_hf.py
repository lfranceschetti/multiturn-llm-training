import os
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, login
import torch
import json

# Replace with your Hugging Face token
HF_TOKEN = "***REMOVED***"
login(HF_TOKEN)

# Initialize accelerator

print(torch.cuda.is_available())
print(torch.cuda.device_count())


accelerator = Accelerator()


# Paths and repository details
output_dir = "/cluster/scratch/fraluca/huggingface/models/REFUEL-1B-test-2"
  # Replace with the path to your saved model directory
base_model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Replace with the base model name (e.g., bert-base-uncased)
repo_name = "LuckyLukke/REFUEL-1B-test-2"  # Replace with your desired repo name


# Load the model architecture from the base model
model = AutoModelForCausalLM.from_pretrained(base_model_name)

# Load the training state
accelerator.load_state(output_dir)

# Unwrap the model from the accelerator
model = accelerator.unwrap_model(model)



tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = model.to(torch.float16)
model.push_to_hub(repo_name)
tokenizer.push_to_hub(repo_name)

print(f"Model successfully uploaded to https://huggingface.co/{repo_name}")

