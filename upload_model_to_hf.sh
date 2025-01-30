#!/bin/bash
#SBATCH --time=00:30:00          # Max runtime
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

export HF_HOME="/cluster/scratch/fraluca/huggingface"
export HF_TOKEN="***REMOVED***"

# MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Execute Python script
python ./upload_model_to_hf.py
