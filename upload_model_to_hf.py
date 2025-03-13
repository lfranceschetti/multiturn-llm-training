import os
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, login
import torch
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint

import json

# Replace with your Hugging Face token

# Initialize accelerator

print(torch.cuda.is_available())
print(torch.cuda.device_count())


accelerator = Accelerator()


names = [
  "clembench_REFUEL_base_1-4000",
  "clembench_REFUEL_SFT_1-4000"

  # "wordle_REFUEL_turnwise-0.001",
]


for name in names:
  # Replace with the path to your saved model directory

  
  output_dir = "/cluster/scratch/fraluca/huggingface/models/" + name

  # if len(name.split("-")) == 5:
  #   #Multiply the checkpoint by 2
  #   checkpoint = int(name.split("-")[-1]) * 2
  #   name = "-".join(name.split("-")[:-1])
  #   name += "-" + str(checkpoint)

  base_model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Replace with the base model name (e.g., bert-base-uncased)
  repo_name = "LuckyLukke/" + name


  # Load the model architecture from the base model
  model = AutoModelForCausalLM.from_pretrained(base_model_name)

  model = accelerator.unwrap_model(model)

  # Load the training state

  model = load_state_dict_from_zero_checkpoint(model, output_dir)


  tokenizer = AutoTokenizer.from_pretrained(base_model_name)
  model.push_to_hub(repo_name)
  tokenizer.push_to_hub(repo_name)

  print(f"Model successfully uploaded to https://huggingface.co/{repo_name}")

