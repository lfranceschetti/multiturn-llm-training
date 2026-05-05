#!/bin/bash
# One-shot setup for a fresh RunPod pod that will run a v2 eval slice.
# Idempotent — safe to re-run.
#
# Prerequisites (on RunPod):
#   - Pod started with PyTorch container, A100 80GB or RTX 5090 32GB+.
#   - Volume / disk >= 60GB (model + checkpoints + cache).
#   - You have OPENAI_API_KEY and HF_TOKEN (HuggingFace read token) ready.
#
# Usage:
#   bash setup.sh                 # base deps (existing repo script)
#   bash evaluations/runpod_setup.sh
#   # then export OPENAI_API_KEY=... HF_TOKEN=... and re-run, or manually edit secrets.json
set -e

cd /workspace/multiturn-llm-training

# 1. Base dependencies (delegated to repo's existing setup.sh).
if [ ! -f .runpod_setup_done ]; then
    echo "[1/4] Running base setup.sh..."
    bash setup.sh
    touch .runpod_setup_done
else
    echo "[1/4] Base setup already done (.runpod_setup_done exists)."
fi

# 2. secrets.json for OpenAI judge (gpt-4o-mini).
if [ -f secrets.json ]; then
    echo "[2/4] secrets.json already exists."
else
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "[2/4] WARNING: OPENAI_API_KEY not set."
        echo "      Create secrets.json manually with:"
        echo '      echo {\"openai\": {\"api_key\": \"sk-...\"}} > secrets.json'
    else
        cat > secrets.json <<EOF
{"openai": {"api_key": "${OPENAI_API_KEY}"}}
EOF
        chmod 600 secrets.json
        echo "[2/4] Wrote secrets.json from OPENAI_API_KEY env var."
    fi
fi

# 3. HuggingFace login (needed to pull migub/* checkpoint repos).
if [ -n "$HF_TOKEN" ]; then
    echo "[3/4] Logging into HuggingFace with HF_TOKEN..."
    python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"
else
    echo "[3/4] HF_TOKEN not set. If migub/* repos are private, run:"
    echo "      huggingface-cli login"
fi

# 4. Enable hf_transfer for faster downloads.
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> ~/.bashrc

echo ""
echo "[4/4] Setup done. Next:"
echo "      bash evaluations/download_v2_checkpoints.sh"
echo "      then: bash evaluations/run_v2_extra_gpu<N>.sh   (N = 1, 2, or 3)"
