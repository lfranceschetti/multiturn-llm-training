#!/bin/bash
#SBATCH --time=64:00:00          # Max runtime
#SBATCH --mem-per-cpu=50G         # Memory per CPU
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks=1               # Single task for dataset creation
#SBATCH --output=refuel_multi_game.out    # Output log file
#SBATCH --error=refuel_multi_game.err    # Error log file
#SBATCH --gpus=rtx_4090:2

# Set model name if not already set (default value)
if [ -z "$MODEL_NAME" ]; then
    export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
fi

chmod +x bash_variables.sh
chmod +x start_api.sh
source ./bash_variables.sh
source ./start_api.sh

# Run dataset creation script directly (no need for accelerate launch)
python multiturn_llm_training/REFUEL/create_dataset.py --model "$MODEL_NAME" --num-samples 8000 --game-type multi-game --hf-repo LuckyLukke/negotio_REFUEL