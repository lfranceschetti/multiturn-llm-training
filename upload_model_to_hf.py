import os
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, login
import torch
import json

# Replace with your Hugging Face token

# Initialize accelerator

print(torch.cuda.is_available())
print(torch.cuda.device_count())


accelerator = Accelerator()


# Paths and repository details
output_dir = "/cluster/scratch/mgiulianelli/huggingface/models/REFUEL-onesided-beta-0.01-1250"

names = [
  "REFUEL-onesided-beta-0.01-1250",
  "REFUEL-onesided-beta-0.01-2500",
  "REFUEL-onesided-beta-0.01-3750",
  "REFUEL-onesided-beta-0.1-1250",
  "REFUEL-onesided-beta-0.1-2500",
  "REFUEL-onesided-beta-0.1-3750",
  "REFUEL-onesided-beta-0.1-5000"
]


for name in names:
  # Replace with the path to your saved model directory

  
  output_dir = "/cluster/scratch/mgiulianelli/huggingface/models/" + name
  base_model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Replace with the base model name (e.g., bert-base-uncased)
  repo_name = "LuckyLukke/" + name


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

