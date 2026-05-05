#!/bin/bash
# Eval samples 10..27 (18 samples) for THREE models on GPU 2:
#   - grpo_all_equal_560
#   - grpo_self_fair_equal_560
#   - lagrpo_self_only_760
# Samples 0..9 are already in the existing <name>.json files.
#
# Per model: 18 samples x 20 reps = 360 games.
# Per pod: 3 models x 360 = 1080 games.
# Outputs: evaluations/results/negotiation/v2/<name>_samples10-27.json
set -e

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

OUTPUT_DIR="output"
RESULTS_DIR="evaluations/results/negotiation/v2"
REPS=20
GAME_TYPE="multi-game"
START_IDX=10
NUM_GAMES=18
SLICE_TAG="samples${START_IDX}-$((START_IDX + NUM_GAMES - 1))"

run_eval_slice() {
    local ckpt_path="$1"
    local out_name="$2"
    local lam_self="$3"
    local lam_welfare="$4"
    local lam_fair="$5"

    local target="$RESULTS_DIR/${out_name}_${SLICE_TAG}.json"
    if [ -f "$target" ]; then
        echo "[SKIP] $target already exists"
        return 0
    fi

    echo ""
    echo ">>> [GPU $CUDA_VISIBLE_DEVICES] ${out_name}: samples ${START_IDX}..$((START_IDX+NUM_GAMES-1))"
    python evaluations/run_negotiation_eval.py \
        --checkpoint "${ckpt_path}" \
        --repetitions ${REPS} \
        --game-type ${GAME_TYPE} \
        --num-games ${NUM_GAMES} \
        --sample-start-idx ${START_IDX} \
        --output-dir "$RESULTS_DIR" \
        --lambda-self ${lam_self} \
        --lambda-welfare ${lam_welfare} \
        --lambda-fair ${lam_fair}

    local base
    if [ "$ckpt_path" = "none" ]; then
        base="none"
    else
        base=$(basename "${ckpt_path%/}")
    fi
    local src="$RESULTS_DIR/${base}_${SLICE_TAG}.json"
    if [ -f "$src" ] && [ "$src" != "$target" ]; then
        mv "$src" "$target"
    fi
    echo "[DONE] $target"
}

run_eval_slice "output/grpo-multigame-all-equal/checkpoint-560"      "grpo_all_equal_560"       0.33 0.33 0.33
run_eval_slice "output/grpo-multigame-self-fair-equal/checkpoint-560" "grpo_self_fair_equal_560" 0.5  0.0  0.5
run_eval_slice "output/lagrpo-self-only-v2/checkpoint-760"           "lagrpo_self_only_760"     1.0  0.0  0.0

echo ""
echo "============================================"
echo "GPU 2 complete: 3 models x 18 samples x 20 reps = 1080 games."
echo "============================================"
ls -la "$RESULTS_DIR"/{grpo_all_equal_560,grpo_self_fair_equal_560,lagrpo_self_only_760}_${SLICE_TAG}.json 2>/dev/null
