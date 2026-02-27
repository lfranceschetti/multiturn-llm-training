#!/bin/bash
#SBATCH --time=0:10:00          # Max runtime
#SBATCH --mem-per-cpu=50G         # Memory per CPU
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks=2
#SBATCH --output=dpo_test.out    # Output log file
#SBATCH --error=dpo_test.err    # Error log file

chmod +x bash_variables.sh
# chmod +x start_api.sh
source ./bash_variables.sh
# source ./start_api.sh

python -m multiturn_llm_training.DPO.create_dataset --model meta-llama/Llama-3.1-8B-Instruct --num-samples 50 --game-type generic-rental-agreement