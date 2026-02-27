#!/bin/bash
#SBATCH --time=5:00:00          # Max runtime
#SBATCH --mem-per-cpu=50G         # Memory per CPU
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks=1               # Single task for dataset creation
#SBATCH --output=grpo_test_qlora.out    # Output log file
#SBATCH --error=grpo_test_qlora.err    # Error log file
#SBATCH --gpus=rtx_4090:4

# Set model name if not already set (default value)
if [ -z "$MODEL_NAME" ]; then
    export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
fi


chmod +x bash_variables.sh
chmod +x start_api.sh
source ./bash_variables.sh
source ./start_api.sh

#Override the ACCELERATE_CONFIG to use the correct config
export ACCELERATE_CONFIG="/cluster/home/fraluca/.cache/huggingface/accelerate/default_config.yaml"

if [ -z "$RUN_NAME" ]; then
    export RUN_NAME="grpo_test_8B_qlora"
fi

CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
  --config_file=$ACCELERATE_CONFIG \
  multiturn_llm_training/grpo/grpo.py \
  --output-dir=/cluster/scratch/fraluca/huggingface/models \
  --model-name=$MODEL_NAME \
  --run-name=$RUN_NAME \
  --test-env

exit 0