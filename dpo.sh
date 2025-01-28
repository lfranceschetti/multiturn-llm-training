#!/bin/bash
#SBATCH --time=3:59:00          # Max runtime
#SBATCH --mem-per-cpu=50G         # Memory per CPU
#SBATCH --nodes=1                # Number of nodes
#SBATCH --gpus=a100-pcie-40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=2
#SBATCH --output=dpo40s.out    # Output log file
#SBATCH --error=dpo40s.err    # Error log file


# Load necessary modules
module load stack/2024-05  
module load gcc/13.2.0
module load python/3.11.6_cuda
module load eth_proxy

export HF_HOME="/cluster/scratch/fraluca/huggingface"
export HF_TOKEN="***REMOVED***"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



# MODEL="/cluster/scratch/fraluca/huggingface/models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
# Execute Python script
#/cluster/home/fraluca/negotio/NEGOTIO/deepspeed_config.json
accelerate launch --config_file=/cluster/home/fraluca/.cache/huggingface/accelerate/default_config.yaml --main_process_port=29503 train.py --config-name 8B_DPO_1
 