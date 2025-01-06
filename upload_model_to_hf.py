import os
from accelerate import Accelerator
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import HfApi, login
import torch

# Replace with your Hugging Face token
HF_TOKEN = "***REMOVED***"
login(HF_TOKEN)

# Initialize accelerator
accelerator = Accelerator(deepspeed_plugin=None)

print(torch.cuda.is_available())
print(torch.cuda.device_count())

if not torch.cuda.is_available():
    print("CUDA not available. Forcing CPU mode.")
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Paths and repository details
output_dir = "/cluster/scratch/fraluca/huggingface/models/8B_7000_1_555134_1735558948"  # Replace with the path to your saved model directory
base_model_name = "meta-llama/Llama-3.2-1B-Instruct"  # Replace with the base model name (e.g., bert-base-uncased)
repo_name = "LuckyLukke/meta-negotio-1B-1"  # Replace with your desired repo name


# Ensure the directory exists
if not os.path.exists(output_dir):
    raise FileNotFoundError(f"Output directory {output_dir} does not exist.")

# Load the model state
accelerator.load_state(output_dir)

# Load the model and tokenizer
model = AutoModel.from_pretrained(base_model_name)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)



def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# Assuming `policy` is your model
total_params, trainable_params = count_model_parameters(model)
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")

# Push model to Hugging Face
print("Pushing model to Hugging Face Hub...")
model.push_to_hub(repo_name, use_temp_dir=True)
tokenizer.push_to_hub(repo_name, use_temp_dir=True)


# Optional: Create a model card
model_card_content = f"""
# {repo_name}

This model was fine-tuned using Accelerate. Below are the details:

## Model Details
- **Base Model**: {base_model_name}
- **Trained Using**: Hugging Face Accelerate
- **Purpose**: Specify the use case
"""
readme_path = os.path.join(output_dir, "README.md")
with open(readme_path, "w") as f:
    f.write(model_card_content)

# Upload the README.md to the repository
print("Uploading README.md...")
api = HfApi()
api.upload_file(
    path_or_fileobj=readme_path,
    path_in_repo="README.md",
    repo_id=repo_name,
    token=HF_TOKEN,
)

print(f"Model successfully uploaded to https://huggingface.co/{repo_name}")

# Optional: Clear GPU cache
import torch
torch.cuda.empty_cache()
print("GPU cache cleared.")
