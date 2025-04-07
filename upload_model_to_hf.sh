#!/bin/bash
#SBATCH --time=01:00:00          # Max runtime
#SBATCH --mem-per-cpu=100G         # Memory per CPU
#SBATCH --nodes=1                # Number of nodes
#SBATCH --gpus=a100-pcie-40gb:2
#SBATCH --output=upload.out    # Output log file
#SBATCH --error=upload.err    # Error log file

# Load necessary modules
module load stack/2024-05  
module load gcc/13.2.0
module load python/3.11.6_cuda
module load eth_proxy

source ~/my_env.sh

# Define variables
OUTPUT_DIR="/cluster/scratch/mgiulianelli/huggingface/models"
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct" 
MODEL_NAMES=(
    "clembench_REFUEL_base_3-4000"
    "clembench_REFUEL_base_3-8000"
)

# Convert array to space-separated string
MODEL_NAMES_STR="${MODEL_NAMES[*]}"

# Execute Python script with arguments
python ./upload_model_to_hf.py \
    --output_dir "$OUTPUT_DIR" \
    --base_model "$BASE_MODEL" \
    --model_names $MODEL_NAMES_STR
    
echo "Model upload job completed"