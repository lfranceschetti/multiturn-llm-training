#!/usr/bin/env bash

# Load necessary modules
module load stack/2024-05  
module load gcc/13.2.0
module load python/3.11.6_cuda
module load eth_proxy
unset LD_LIBRARY_PATH

export PYTHONPATH="$(pwd)/envs/negotiation:${PYTHONPATH}"

# Path to local TRL
TRL_PATH="/cluster/home/fraluca/negotio2"
export PYTHONPATH="$TRL_PATH:$PYTHONPATH"
export PYTHONPATH="$TRL_PATH/trl:$PYTHONPATH"

#export ACCELERATE_CONFIG="/cluster/home/fraluca/.cache/huggingface/accelerate/default_config.yaml"
export ACCELERATE_CONFIG="/cluster/home/fraluca/.cache/huggingface/accelerate/dataset_creation_config.yaml"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORK_DIR="/cluster/home/fraluca/negotio2/multiturn-llm-training"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_DIR="${WORK_DIR}/logs"
RUN_DIR="${LOG_DIR}/${TIMESTAMP}"

# Create directories
mkdir -p "$RUN_DIR"
echo "Logs will be stored in: $RUN_DIR"


export NODE="$(hostname -f)"     # or: "$(hostname)"

# Find a free port starting from a given base port.
# Checks both ss (socket stats) and curl to ensure no service is listening.
find_free_port() {
    local start_port=$1
    local port=$start_port
    while ss -tlnp 2>/dev/null | grep -q ":${port} " || \
          curl -s --max-time 1 --noproxy "*" "http://$(hostname):${port}/" >/dev/null 2>&1; do
        port=$((port + 1))
        if [ $port -gt 65535 ]; then
            echo "ERROR: Could not find a free port starting from $start_port" >&2
            exit 1
        fi
    done
    echo $port
}

# Use dynamic ports based on SLURM_JOB_ID to avoid conflicts between concurrent jobs
# Verifies each port is actually free before using it
if [ -n "$SLURM_JOB_ID" ]; then
    BASE_PORT=$((8000 + ($SLURM_JOB_ID % 1000) * 2))
else
    BASE_PORT=8000
fi

export MODEL0_PORT=$(find_free_port $BASE_PORT)
export MODEL1_PORT=$(find_free_port $((MODEL0_PORT + 1)))
echo "Using ports: MODEL0_PORT=$MODEL0_PORT, MODEL1_PORT=$MODEL1_PORT"

# Also set unique accelerate main_process_port if using distributed training
if [ -n "$SLURM_JOB_ID" ]; then
    export ACCELERATE_MAIN_PROCESS_PORT=$((29500 + ($SLURM_JOB_ID % 1000)))
else
    export ACCELERATE_MAIN_PROCESS_PORT=29500
fi
echo "Accelerate main_process_port: $ACCELERATE_MAIN_PROCESS_PORT"

# keep your existing RUN_DIR lines
export RUN_DIR LOG_DIR TIMESTAMP