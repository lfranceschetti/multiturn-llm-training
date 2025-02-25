#!/bin/bash
#SBATCH --time=23:59:00          # Max runtime
#SBATCH --mem-per-cpu=50G         # Memory per CPU
#SBATCH --nodes=1                # Number of nodes
#SBATCH --gpus=a100_80gb:2
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=2
#SBATCH --output=grpo_online.out    # Output log file
#SBATCH --error=grpo_online.err    # Error log file


# Load necessary modules
module load stack/2024-05  
module load gcc/13.2.0
module load python/3.11.6_cuda
module load eth_proxy

source ~/my_env.sh


# Execute Python script
accelerate launch --config_file=$ACCELERATE_CONFIG --main_process_port=29501 train.py --config-name 8B_GRPO_1
 