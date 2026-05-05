#!/bin/bash
# Download ALL 7 LoRA adapters needed by the v2 multi-game eval.
# For the single-pod / multi-GPU layout (one pod runs all three slice scripts).
# Idempotent -- skips checkpoints already on disk.
set -e

cd "$(dirname "$0")/.."
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p output

download_ckpt() {
    local repo="$1"
    local ckpt="$2"
    local target="output/${repo}/${ckpt}"
    if [ -f "$target/adapter_model.safetensors" ]; then
        echo "[SKIP] ${repo}/${ckpt} already present"
    else
        echo "[DOWNLOAD] migub/${repo} -> ${ckpt}"
        hf download "migub/${repo}" --include "${ckpt}/*" --local-dir "output/${repo}"
        echo "[OK] ${repo}/${ckpt}"
    fi
}

# GRPO @ 560
download_ckpt "grpo-multigame-self-only"      "checkpoint-560"
download_ckpt "grpo-multigame-fair-only"      "checkpoint-560"
download_ckpt "grpo-multigame-all-equal"      "checkpoint-560"
download_ckpt "grpo-multigame-self-fair-equal" "checkpoint-560"

# LA-GRPO @ 760
download_ckpt "lagrpo-self-only-v2"           "checkpoint-760"
download_ckpt "lagrpo-multigame-all-equal"    "checkpoint-760"
download_ckpt "lagrpo-fair-only"              "checkpoint-760"

echo ""
echo "============================================"
echo "All v2 adapters ready (7 total)."
echo "============================================"
ls -la output/*/checkpoint-*/adapter_model.safetensors 2>/dev/null
