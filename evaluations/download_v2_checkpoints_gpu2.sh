#!/bin/bash
# Download only the LoRA adapters needed by run_v2_full_gpu2.sh:
#   - grpo-multigame-all-equal      @ checkpoint-560
#   - grpo-multigame-self-fair-equal @ checkpoint-560
#   - lagrpo-self-only-v2            @ checkpoint-760
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

download_ckpt "grpo-multigame-all-equal"       "checkpoint-560"
download_ckpt "grpo-multigame-self-fair-equal" "checkpoint-560"
download_ckpt "lagrpo-self-only-v2"            "checkpoint-760"

echo "[GPU 2 checkpoints ready]"
