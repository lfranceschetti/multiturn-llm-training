#!/bin/bash
# Launch all three v2 slice runners concurrently on a single multi-GPU pod.
#
# Pins each runner to its own GPU via CUDA_VISIBLE_DEVICES, staggers the
# launches by 60s so simultaneous full-precision model loads don't spike
# CPU RAM, and writes per-runner logs.
#
# Prereqs (run before this):
#   bash evaluations/runpod_setup.sh
#   bash evaluations/download_v2_checkpoints_all.sh
#   # Optional but recommended: pre-warm the HF cache for Qwen3-14B base
#   python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
#              AutoTokenizer.from_pretrained('OpenPipe/Qwen3-14B-Instruct'); \
#              AutoModelForCausalLM.from_pretrained('OpenPipe/Qwen3-14B-Instruct')"
#
# Usage:
#   tmux new -s eval
#   bash evaluations/run_v2_all_gpus.sh
#   # Detach: Ctrl+B then D. Reattach: tmux attach -t eval.
#   # Logs in ./eval_logs/gpu{1,2,3}.log; tail -f to watch live.
set -e

cd "$(dirname "$0")/.."
mkdir -p eval_logs
export PYTHONUNBUFFERED=1

echo "=== Launching v2 eval across 3 GPUs ==="
date

# GPU 0: pod 1's models (3 models, 1080 games)
CUDA_VISIBLE_DEVICES=0 bash evaluations/run_v2_full_gpu1.sh > eval_logs/gpu1.log 2>&1 &
PID1=$!
echo "[GPU 0] launched: run_v2_full_gpu1.sh (pid $PID1) -> eval_logs/gpu1.log"

sleep 60
# GPU 1: pod 2's models (3 models, 1080 games)
CUDA_VISIBLE_DEVICES=1 bash evaluations/run_v2_full_gpu2.sh > eval_logs/gpu2.log 2>&1 &
PID2=$!
echo "[GPU 1] launched: run_v2_full_gpu2.sh (pid $PID2) -> eval_logs/gpu2.log"

sleep 60
# GPU 2: pod 3's models (2 models, 720 games -- finishes earlier)
CUDA_VISIBLE_DEVICES=2 bash evaluations/run_v2_full_gpu3.sh > eval_logs/gpu3.log 2>&1 &
PID3=$!
echo "[GPU 2] launched: run_v2_full_gpu3.sh (pid $PID3) -> eval_logs/gpu3.log"

echo ""
echo "All three runners launched. Waiting for completion..."
echo "Tail logs with:  tail -f eval_logs/gpu1.log eval_logs/gpu2.log eval_logs/gpu3.log"
echo "Check GPU usage: nvidia-smi"
echo ""

wait $PID1 && echo "[GPU 0] DONE" || echo "[GPU 0] FAILED"
wait $PID2 && echo "[GPU 1] DONE" || echo "[GPU 1] FAILED"
wait $PID3 && echo "[GPU 2] DONE" || echo "[GPU 2] FAILED"

echo ""
echo "=== All runs complete ==="
date
ls -la evaluations/results/negotiation/v2/*_samples10-27.json 2>/dev/null
